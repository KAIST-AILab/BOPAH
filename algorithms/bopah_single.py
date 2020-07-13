import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from cont_evaluate import evaluate_policy
from algorithms.klac import ReplayBuffer, KLAC, apply_squashing_func


class BOPAHSingle(KLAC):

    def __init__(self, state_dim, action_dim, max_action, kl_coef, lamb=1, num_critics=5, gradient_norm_panelty=0,
                 gradient_norm_limit=30, hidden_dim=64):
        super(BOPAHSingle, self).__init__(state_dim, action_dim, max_action, kl_coef, lamb, num_critics, gradient_norm_panelty, 
                                       gradient_norm_limit, hidden_dim)

        self.valid_obs_ph, self.valid_action_ph, self.valid_reward_ph, self.valid_terminal_ph, self.valid_next_obs_ph \
            = [tf.keras.layers.Input(d) for d in [state_dim, action_dim, 1, 1, state_dim]]
        self.valid_ensemble_mask = tf.keras.layers.Input(self.num_critics)

        # For hyper-gradient
        self.traj_valid_obs_ph = tf.keras.layers.Input((None, self.state_dim), name='traj_valid_obs_ph')   # (maxlen) x |S|
        self.traj_valid_actions_ph = tf.keras.layers.Input((None, self.action_dim), name='traj_valid_actions_ph')  # (maxlen) x |A|
        self.mask_ph = tf.keras.layers.Input((None, 1), name='mask_ph')                # (maxlen) x 1
        self.gradient_buffer_ph = tf.placeholder(dtype=tf.float32, shape=[], name='gradient_buffer_ph')

        valid_action_pi, _, _ = self.actor([self.valid_obs_ph])

        valid_tensors = [self.valid_obs_ph, self.valid_action_ph, self.valid_next_obs_ph, self.valid_terminal_ph, valid_action_pi, self.valid_ensemble_mask]
        self.valid_critic_v, self.valid_critic_q, valid_mean_q_loss, valid_qs_pi, valid_critic_train_op, valid_target_update_op = self._build_critic(
            valid_tensors, v_bonus=0, q_bonus=self.valid_reward_ph, hidden_dim=self.hidden_dim)

        self.eval_ops.extend([tf.reduce_mean(valid_mean_q_loss), tf.reduce_mean(valid_qs_pi)])
        self.eval_labels.extend(['valid_q_loss', 'valid_mean_q'])

        self.step_ops[0] = tf.group([self.step_ops[0], valid_critic_train_op])  # TODO: hard-coded index
        self.step_ops[2] = tf.group([self.step_ops[2], valid_target_update_op])

    def _build_kl_coef(self, init_kl_coef):
        """
        overrides KLAC._build_kl_coef
        """
        self.log_kl_coef = tf.Variable(np.log(init_kl_coef), dtype=tf.float32)
        self.sess.run(self.log_kl_coef.initializer)
        return tf.exp(self.log_kl_coef)

    def _compute_negative_covariance(self, obs, actor, train_critic_q, valid_critic_q, num_samples=20):
        batch_size = tf.shape(obs)[0]
        tiled_obs = tf.tile(obs, [num_samples, 1])
        action_samples, _, actor_dist = actor([tiled_obs])
        action_samples_1, _, actor_dist = actor([tiled_obs])
        action_samples_2, _, actor_dist = actor([tiled_obs])

        q1s_pi = train_critic_q([tiled_obs, action_samples])
        q2s_pi = valid_critic_q([tiled_obs, action_samples])
        q1_pi = tf.reshape(self._reduce_q(q1s_pi), [num_samples, batch_size, -1])
        # handle multiple output dimension for train critic
        q2_pi = tf.reshape(self._reduce_q(q2s_pi), [num_samples, batch_size, 1])
        v1 = tf.reduce_mean(q1_pi, axis=0)
        v2 = tf.reduce_mean(q2_pi, axis=0)

        qq = tf.reduce_mean(q1_pi * q2_pi, axis=0)

        q1s_pi = train_critic_q([tiled_obs, action_samples_1])
        q2s_pi = valid_critic_q([tiled_obs, action_samples_2])
        q1_pi = tf.reshape(self._reduce_q(q1s_pi), [num_samples, batch_size, -1])
        q2_pi = tf.reshape(self._reduce_q(q2s_pi), [num_samples, batch_size, 1])
        v1 = tf.reduce_mean(q1_pi, axis=0)
        v2 = tf.reduce_mean(q2_pi, axis=0)

        return v1 * v2 - qq

    def _get_alpha_train_op(self, train_critic_q, valid_critic_q, maxlen):
        N = tf.shape(self.traj_valid_obs_ph)[0]
        reshaped_obs = tf.reshape(self.traj_valid_obs_ph, [-1, self.state_dim])
        reshaped_actions = tf.reshape(self.traj_valid_actions_ph, [-1, self.action_dim])

        negative_q_cov = self._compute_negative_covariance(reshaped_obs, self.actor, train_critic_q, valid_critic_q)
        pi_grad = tf.reshape(negative_q_cov / self.kl_coef, [N, maxlen, 1])

        _, _, behavior_dist = self.behavior_policy([reshaped_obs])

        _, _, target_dist = self.actor([reshaped_obs])
        pre_squash_actions = tf.math.atanh(tf.clip_by_value(reshaped_actions, -1 + 1e-6, 1 - 1e-6))
        log_is_ratios = target_dist.log_prob(pre_squash_actions) - behavior_dist.log_prob(pre_squash_actions) + tf.log(self.gamma)

        cum_log_is_ratios = tf.cumsum(tf.reshape(log_is_ratios, [N, maxlen, 1]), exclusive=True, axis=1)
        cum_log_is_ratios = tf.clip_by_value(cum_log_is_ratios, -7, 7)

        v_grads_s0 = tf.reduce_sum(tf.exp(cum_log_is_ratios) * pi_grad * self.mask_ph, axis=1)

        gradvar = [(-self.gradient_buffer_ph, self.log_kl_coef)]
        alpha_optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        alpha_train_op = alpha_optimizer.apply_gradients(gradvar)
        with tf.control_dependencies([alpha_train_op]):
            alpha_clip_op = tf.assign(self.log_kl_coef, tf.clip_by_value(self.log_kl_coef, np.log(0.75), np.log(500)))
        self.sess.run(tf.variables_initializer(alpha_optimizer.variables()))
        return tf.group([alpha_train_op, alpha_clip_op]), v_grads_s0

    #######################################
    # interfaces for cont_run.py
    #######################################

    def batch_learn(self, train_trajectory, vec_env, total_timesteps, log_interval, seed, result_filepath=None, valid_trajectory=None):
        np.random.seed(seed)

        train_replay_buffer = ReplayBuffer(train_trajectory, max_action=self.max_action, num_critic=self.num_critics)
        valid_replay_buffer = ReplayBuffer(valid_trajectory, max_action=self.max_action, num_critic=self.num_critics)
        valid_replay_buffer.standardizer = train_replay_buffer.standardizer
        self.standardizer = train_replay_buffer.standardizer

        # Zero-padding of valid (obs, actions)
        valid_obs_padded, valid_actions_padded, valid_terminal_mask, valid_traj_maxlen = ReplayBuffer.group_element_trajectory(valid_trajectory)
        valid_obs_padded, valid_actions_padded = self.standardizer(valid_obs_padded), valid_actions_padded / self.max_action
        valid_trajectory_indices = np.arange(len(valid_trajectory))
        num_updates = 10

        # Hyper-gradient ascent operation
        alpha_train_op, v_grads_s0 = self._get_alpha_train_op(self.critic_q, self.valid_critic_q, valid_traj_maxlen)

        # Start...
        saver = tf.train.Saver(max_to_keep=2)
        last_checkpoint = tf.train.latest_checkpoint(result_filepath + '_checkpoint')
        if last_checkpoint is not None:
            start_time = time.time()
            saver.restore(self.sess, last_checkpoint)
            loaded = np.load(result_filepath + '.tmp.npy', allow_pickle=True).item()
            eval_timesteps = loaded['eval_timesteps']
            evaluations = loaded['evals']
            infos_values = loaded['info_values']
            v_grad_list = []
            timestep = eval_timesteps[-1] + 1
            timesteps = range(timestep, total_timesteps)
            print('loaded', timestep)
            print(eval_timesteps)
            print(infos_values)
        else:
            start_time = time.time()
            eval_timesteps = []
            evaluations = []
            infos_values = []
            v_grad_list = []
            timesteps = range(total_timesteps)
        for timestep in tqdm(timesteps, desc="BOPAHSingle", ncols=70):
            obs, action, reward, next_obs, done, ensemble_mask = train_replay_buffer.sample(self.batch_size)
            valid_obs, valid_action, valid_reward, valid_next_obs, valid_done, valid_ensemble_mask = valid_replay_buffer.sample(self.batch_size)
            feed_dict = {
                self.obs_ph: obs, self.action_ph: action, self.reward_ph: reward,
                self.next_obs_ph: next_obs, self.terminal_ph: done,
                self.valid_obs_ph: valid_obs, self.valid_action_ph: valid_action, self.valid_reward_ph: valid_reward,
                self.valid_next_obs_ph: valid_next_obs, self.valid_terminal_ph: valid_done,
                self.obs_mean: train_replay_buffer.obs_mean, self.obs_std: train_replay_buffer.obs_std,
                self.ensemble_mask: ensemble_mask, self.valid_ensemble_mask: valid_ensemble_mask
            }
            step_result = self.sess.run(self.step_ops + self.eval_ops, feed_dict=feed_dict)
            infos_value = step_result[len(self.step_ops):]

            if (timestep + 1) % 500 == 0 and timestep > 100000:
                grad_values = []
                np.random.shuffle(valid_trajectory_indices)
                reshaped_indices = np.reshape(valid_trajectory_indices[:200], [num_updates, -1])
                for rind in reshaped_indices:
                    v_grads_s0_value = self.sess.run(v_grads_s0, feed_dict={
                        self.obs_mean: train_replay_buffer.obs_mean, self.obs_std: train_replay_buffer.obs_std,
                        self.traj_valid_obs_ph: valid_obs_padded[rind],
                        self.traj_valid_actions_ph: valid_actions_padded[rind],
                        self.mask_ph: valid_terminal_mask[rind]})
                    grad_values += list(v_grads_s0_value.flatten())
#                     print(negative_q_cov_value[:100, 0])
                v_grad_list.append(np.mean(grad_values))
                self.sess.run(alpha_train_op, feed_dict={self.gradient_buffer_ph: np.mean(grad_values)})
                
                print('t=%d: (elapsed_time=%f)' % (timestep, time.time() - start_time))
                print('\n============================')
                for label, value in zip(self.eval_labels, infos_value):
                    if label == 'kl_coef':
                        print('%16s: %10.3f' % (label, value))
                        print('%16s: %10.3f' % ('log_kl_coef', np.log(value)))
                print('%16s: %10.3f' % ('total_grad_value', np.mean(grad_values)))
                print('============================\n')
                
            if timestep % log_interval == 0:
                print('-----------saving----------------------')
                v_grad_mean = np.mean(v_grad_list)
                v_grad_list = []
                evaluation = evaluate_policy(vec_env, self)
                eval_timesteps.append(timestep)
                evaluations.append(evaluation)
                infos_values.append(infos_value + [v_grad_mean])
                print('t=%d: %f (elapsed_time=%f)' % (timestep, evaluation, time.time() - start_time))
                print('\n============================')
                for label, value in zip(self.eval_labels, infos_value):
                    print('%12s: %10.3f' % (label, value))
                print('============================\n')

                if result_filepath:
                    result = {'eval_timesteps': eval_timesteps, 'evals': evaluations, 'info_values': infos_values}
                    np.save(result_filepath + '.tmp.npy', result)
                    saver.save(self.sess, result_filepath + '_checkpoint/model')

        return eval_timesteps, evaluations, infos_values
