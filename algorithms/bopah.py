import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from stable_baselines.common import set_global_seeds

from cont_evaluate import evaluate_policy
from algorithms.klac import ReplayBuffer, apply_squashing_func, Actor
from algorithms.bopah_single import BOPAHSingle

class BOPAH(BOPAHSingle):

    def _squared_exponential_kernel(self, input1, input2):
        diffs = (input1[:, None] - input2[None]) / np.sqrt(self.medians_heuristics) # N1, N2, Ds
        squared_diffs = tf.reduce_sum(tf.square(diffs), axis=2) # N1, N2
        return tf.cast(tf.exp(-0.5 * squared_diffs), tf.float64) # B, Nc

    def _calculate_c_functions(self, obs_inputs):
        k_sx = self._squared_exponential_kernel(obs_inputs, self.rbf_means) # B, N
        output = k_sx
        remainder = tf.maximum(1 - tf.reduce_sum(output, axis=1, keepdims=True), 0) # B, 1
        output = tf.concat([remainder, output], axis=1)
        output = tf.maximum(tf.cast(output, tf.float32), 0)
        output = output / tf.reduce_sum(output, axis=1, keepdims=True)
        return output

    def standardizer(self, ob):
        return (ob - self.obs_mean) / self.obs_std

    def _preprocess_trajectories(self, train_trajectory, valid_trajectory, max_action, num_critics, batch_size=64, traj_batch_size=5):
        train_obs, train_actions, train_rewards, train_next_obs, train_dones, train_ensemble_masks = \
            [np.array(out, dtype=np.float32) for out in ReplayBuffer.group_elements(train_trajectory, num_critics)]
        valid_obs, valid_actions, valid_rewards, valid_next_obs, valid_dones, valid_ensemble_masks = \
            [np.array(out, dtype=np.float32) for out in ReplayBuffer.group_elements(valid_trajectory, num_critics)]
        self.obs_mean = np.mean(train_obs, axis=0, keepdims=True)
        self.obs_std = np.std(train_obs, axis=0, keepdims=True) + 1e-3
        self.train_obs = train_obs

        train_iter = tf.data.Dataset.from_tensor_slices((
            self.standardizer(train_obs), train_actions / max_action, train_rewards[:, None], self.standardizer(train_next_obs), 
            train_dones[:, None], train_ensemble_masks)).shuffle(buffer_size=100).repeat().batch(batch_size).make_one_shot_iterator()
        valid_iter = tf.data.Dataset.from_tensor_slices((
            self.standardizer(valid_obs), valid_actions / max_action, valid_rewards[:, None], self.standardizer(valid_next_obs), 
            valid_dones[:, None], valid_ensemble_masks)).shuffle(buffer_size=100).repeat().batch(batch_size).make_one_shot_iterator()

        traj_valid_obs_padded, traj_valid_actions_padded, traj_valid_terminal_mask, valid_traj_maxlen \
            = ReplayBuffer.group_element_trajectory(valid_trajectory)
        traj_valid_obs_padded, traj_valid_actions_padded, traj_valid_terminal_mask = self.standardizer(traj_valid_obs_padded).astype('float32'), \
            (traj_valid_actions_padded / max_action).astype('float32'), traj_valid_terminal_mask.astype('float32')

        traj_valid_iter = tf.data.Dataset.from_tensor_slices((
            traj_valid_obs_padded, traj_valid_actions_padded, traj_valid_terminal_mask
            )).shuffle(buffer_size=100).repeat().batch(traj_batch_size).make_one_shot_iterator()

        return train_iter, valid_iter, traj_valid_iter, valid_traj_maxlen
        

    def __init__(self, train_trajectory, valid_trajectory, state_dim, action_dim, max_action, kl_coef, lamb=1, num_critics=5,
                 gradient_norm_panelty=0, gradient_norm_limit=30, hidden_dim=64, cluster_info=None, dependent_limit=0.05, seed=0, total_loss=None):
        set_global_seeds(seed)
        self.state_dim, self.action_dim, self.max_action, self.lamb, self.num_critics, self.hidden_dim \
            = state_dim, action_dim, max_action, lamb, num_critics, hidden_dim
        self.gradient_norm_panelty, self.gradient_norm_limit = gradient_norm_panelty, gradient_norm_limit

        self.rbf_means = cluster_info['representatives']
        squared_diffs = np.sum(np.square(self.rbf_means[:, None] - self.rbf_means[None]), axis=2)
        self.medians_heuristics = np.sqrt(np.median(squared_diffs[~np.eye(squared_diffs.shape[0], dtype=bool)]) / 2) / 2
        len_kl_coef = len(self.rbf_means) + 1

        self.sess = tf.keras.backend.get_session()
        train_iter, valid_iter, traj_valid_iter, traj_valid_maxlen = self._preprocess_trajectories(train_trajectory, valid_trajectory, max_action, num_critics)
        train_obs, train_actions, train_rewards, train_next_obs, train_terminals, train_ensemble_masks = train_iter.get_next()
        valid_obs, valid_actions, valid_rewards, valid_next_obs, valid_terminals, valid_ensemble_masks = valid_iter.get_next()
        traj_valid_obs, traj_valid_actions, traj_valid_mask = traj_valid_iter.get_next()

        self.state_dep_gradient_buffer_ph = tf.placeholder(dtype=tf.float32, shape=[len_kl_coef, 1])
        self.state_ind_gradient_buffer_ph = tf.placeholder(dtype=tf.float32, shape=[])

        self.gamma, self.tau, self.learning_rate = 0.99, 0.005, 3e-4

        initial_log_kl_coef = np.zeros([len_kl_coef, 1])
        log_state_dep_kl_coef = tf.Variable(initial_log_kl_coef, dtype=tf.float32)
        log_state_ind_kl_coef = tf.Variable(np.log(kl_coef), dtype=tf.float32)
        self.sess.run([log_state_dep_kl_coef.initializer, log_state_ind_kl_coef.initializer])
        state_dep_kl_coef = tf.exp(log_state_dep_kl_coef) # (Nc + 1) x 1
        state_ind_kl_coef = tf.exp(log_state_ind_kl_coef)
        kl_coef_vec = state_dep_kl_coef * state_ind_kl_coef

        # Actor, Critic
        actor = Actor(action_dim, max_action, hidden_dim=hidden_dim)
        action_pi, action_logp, dist = actor([train_obs])
        valid_action_pi, _, _ = actor([valid_obs])

        # Baseline policy
        kl, behavior_train_op, behavior_loss = self._build_baseline_policy_and_kl(dist, train_obs, train_actions)

        # Calculate c functions
        rbf_weights = self._calculate_c_functions(train_obs)

        # Critic training
        train_tensors = [train_obs, train_actions, train_next_obs, train_terminals, action_pi, train_ensemble_masks]
        self.critic_v, critic_q, mean_q_loss, qs_pi, critic_train_op, target_update_op = self._build_critic(
            train_tensors, v_bonus=0, q_bonus=train_rewards, hidden_dim=hidden_dim)
        self.kl_critic_v, kl_critic_q, kl_mean_q_loss, kl_qs_pi, kl_critic_train_op, kl_target_update_op = self._build_critic(
            train_tensors, v_bonus=-rbf_weights*kl, q_bonus=0, output_dim=len_kl_coef, hidden_dim=int(hidden_dim * np.sqrt(len_kl_coef)), total_loss=total_loss)
        
        valid_tensors = [valid_obs, valid_actions, valid_next_obs, valid_terminals, valid_action_pi, valid_ensemble_masks]
        self.valid_critic_v, valid_critic_q, valid_mean_q_loss, valid_qs_pi, valid_critic_train_op, valid_target_update_op = self._build_critic(
            valid_tensors, v_bonus=0, q_bonus=valid_rewards, hidden_dim=hidden_dim)

        critic_train_op_group = tf.group([critic_train_op, kl_critic_train_op, valid_critic_train_op])
        target_update_op_group = tf.group([target_update_op, kl_target_update_op, valid_target_update_op])

        # Actor training
        actor_loss = tf.matmul(rbf_weights * kl, kl_coef_vec) \
            - tf.matmul(tf.reduce_mean(kl_qs_pi, axis=0), kl_coef_vec) - tf.reduce_mean(qs_pi, axis=0)
        actor_loss = tf.reduce_mean(actor_loss)
        actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=actor.trainable_variables)
        self.sess.run(tf.variables_initializer(actor_optimizer.variables()))

        self.step_ops = [critic_train_op_group, actor_train_op, target_update_op_group, behavior_train_op]
        self.eval_ops = [tf.reduce_mean(kl), actor_loss, mean_q_loss + kl_mean_q_loss,
                        tf.reduce_mean(qs_pi), tf.reduce_mean(kl_qs_pi), tf.reduce_mean(kl_coef_vec), 
                        tf.reduce_mean(valid_mean_q_loss), tf.reduce_mean(valid_qs_pi)]
        self.eval_labels = ['kl', 'actor_loss', 'mean_q_loss', 'mean_q_r', 'mean_q_kl', 'kl_coef', 'valid_q_loss', 'valid_mean_q']

        # For hypergrad computation
        N = tf.shape(traj_valid_obs)[0]
        reshaped_obs = tf.reshape(traj_valid_obs, [-1, self.state_dim])  # traj_valid_obs: (episode batch) x (max_time_step + 1) x ds
        reshaped_actions = tf.reshape(traj_valid_actions, [-1, self.action_dim])
        
        reward_covariance = self._compute_negative_covariance(reshaped_obs, actor, critic_q, valid_critic_q)
        kl_covariance = self._compute_negative_covariance(reshaped_obs, actor, kl_critic_q, valid_critic_q) # B, (Nc+1)
        total_covariance = reward_covariance + tf.matmul(kl_covariance, kl_coef_vec)
        
        rbf_weights = self._calculate_c_functions(reshaped_obs) # B, (Nc + 1)
        denom = tf.matmul(rbf_weights, state_dep_kl_coef) # (B x D_beta) x (D_beta x 1) = B, 1
        state_dep_pi_grad = total_covariance * rbf_weights / denom / state_ind_kl_coef - kl_covariance
        state_dep_pi_grad = tf.reshape( state_dep_pi_grad / denom * tf.transpose(state_dep_kl_coef), [N, traj_valid_maxlen, len_kl_coef])

        state_ind_pi_grad = tf.reshape( reward_covariance / denom / state_ind_kl_coef, [N, traj_valid_maxlen, 1])

        _, _, behavior_dist = self.behavior_policy([reshaped_obs])

        _, _, target_dist = actor([reshaped_obs])
        pre_squash_actions = tf.math.atanh(tf.clip_by_value(reshaped_actions, -1 + 1e-6, 1 - 1e-6))
        log_is_ratios = target_dist.log_prob(pre_squash_actions) - behavior_dist.log_prob(pre_squash_actions) + tf.log(self.gamma)

        cum_log_is_ratios = tf.cumsum(tf.reshape(log_is_ratios, [N, traj_valid_maxlen, 1]), exclusive=True, axis=1)
        cum_log_is_ratios = tf.clip_by_value(cum_log_is_ratios, -7, 7)

        state_dep_v_grads_s0 = tf.reduce_sum(tf.exp(cum_log_is_ratios) * state_dep_pi_grad * traj_valid_mask, axis=1)
        state_ind_v_grads_s0 = tf.reduce_sum(tf.exp(cum_log_is_ratios) * state_ind_pi_grad * traj_valid_mask, axis=1)

        gradvar = [(-self.state_dep_gradient_buffer_ph * 1e-1, log_state_dep_kl_coef), 
                   (-self.state_ind_gradient_buffer_ph, log_state_ind_kl_coef)]
        coef_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        coef_train_op = coef_optimizer.apply_gradients(gradvar)
        with tf.control_dependencies([coef_train_op]):
            state_dep_coef_clip_op = tf.assign(log_state_dep_kl_coef, tf.clip_by_value(log_state_dep_kl_coef, -dependent_limit, dependent_limit))
            state_ind_coef_clip_op = tf.assign(log_state_ind_kl_coef, tf.clip_by_value(log_state_ind_kl_coef, np.log(0.75) + dependent_limit, np.log(500) - dependent_limit))
        self.sess.run(tf.variables_initializer(coef_optimizer.variables()))

        # For action selection
        self.obs_ph = tf.keras.layers.Input(state_dim)
        self.sampled_action, _, _ = actor([self.obs_ph])
        self.sampled_action_q = tf.reduce_mean(critic_q([self.obs_ph, self.sampled_action]), axis=0)
        self.rbf_weights_on_ph = self._calculate_c_functions(self.obs_ph)
        self.train_value = self.critic_v([self.obs_ph])
        self.valid_value = self.valid_critic_v([self.obs_ph])
        self.coef_to_plot = tf.matmul(self.rbf_weights_on_ph, kl_coef_vec)
        self.kl_coef_vec = kl_coef_vec

        # ops that are called outside
        self.coef_train_op = tf.group([coef_train_op, state_dep_coef_clip_op, state_ind_coef_clip_op])
        self.state_dep_v_grads_s0 = state_dep_v_grads_s0
        self.state_ind_v_grads_s0 = state_ind_v_grads_s0
        tf.get_default_graph().finalize()
        
    def batch_learn(self, train_trajectory, vec_env, total_timesteps, log_interval, seed, result_filepath=None, valid_trajectory=None):
        # Start...
        start_time = time.time()
        eval_timesteps = []
        evaluations = []
        infos_values = []
        hypergrad_lr = 1e-2
        for timestep in tqdm(range(total_timesteps), desc="BOPAH", ncols=70):
            step_result = self.sess.run(self.step_ops + self.eval_ops)
            infos_value = step_result[len(self.step_ops):]

            if (timestep + 1) % 500 == 0 and timestep > 100000:
                state_dep_grad_values = []
                state_ind_grad_values = []
                for i in range(40):
                    state_dep_output, state_ind_output = self.sess.run([self.state_dep_v_grads_s0, self.state_ind_v_grads_s0])
                    state_dep_grad_values += list(state_dep_output)
                    state_ind_grad_values += list(state_ind_output)
                self.sess.run(self.coef_train_op, feed_dict={
                    self.state_dep_gradient_buffer_ph: np.mean(state_dep_grad_values, axis=0, keepdims=True).T,
                    self.state_ind_gradient_buffer_ph: np.mean(state_ind_grad_values)})
                print('\n============================')
                for label, value in zip(self.eval_labels, infos_value):
                    if label == 'kl_coef':
                        print('%16s: %10.3f' % (label, value))
                        print('%16s: %10.3f' % ('log_kl_coef', np.log(value)))
                print('%16s: %10.3f' % ('total_dep_grad_value', np.mean(state_dep_grad_values)))
                print('%16s: %10.3f' % ('total_ind_grad_value', np.mean(state_ind_grad_values)))
                print(np.transpose(self.sess.run(self.kl_coef_vec)))
                print('============================\n')

            if timestep % log_interval == 0:
                evaluation = evaluate_policy(vec_env, self)
                eval_timesteps.append(timestep)
                evaluations.append(evaluation)
                infos_values.append(infos_value)
                print('t=%d: %f (elapsed_time=%f)' % (timestep, evaluation, time.time() - start_time))
                print('\n============================')
                for label, value in zip(self.eval_labels, infos_value):
                    print('%12s: %10.3f' % (label, value))
                print('============================\n')

                if result_filepath:
                    result = {'eval_timesteps': eval_timesteps, 'evals': evaluations, 'info_values': infos_values}
                    np.save(result_filepath + '.tmp.npy', result)

            if timestep % log_interval == 0 and self.state_dim == 3:
                from plot_alpha import plot_value_pendulum
                plot_value_pendulum(self.sess, self.standardizer, self.obs_ph,
                {'train_value': self.train_value, 'valid_value': self.valid_value,
                 'coef_log': self.coef_to_plot},
                           self.train_obs, "{}, {}".format(timestep, evaluations[-1]), 
                           self.rbf_means * self.obs_std + self.obs_mean)
            if timestep % log_interval == 0 and self.state_dim == 2:
                from plot_alpha import plot_value_mc
                plot_value_mc(self.sess, self.standardizer, self.obs_ph,
                {'train_value': self.train_value, 'valid_value': self.valid_value,
                 'coef_log': self.coef_to_plot},
                           self.train_obs, "{}, {}".format(timestep, evaluations[-1]), 
                           self.rbf_means * self.obs_std + self.obs_mean)

        return eval_timesteps, evaluations, infos_values
