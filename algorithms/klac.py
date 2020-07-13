import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from cont_evaluate import evaluate_policy
from algorithms.sac import apply_squashing_func

# Simple replay buffer
class ReplayBuffer:

    @classmethod
    def group_elements(cls, trajectory, num_critic=None):
        flat_trajectory = [exp for traj_one in trajectory for exp in traj_one]
        obs, actions, rewards, next_obs, dones = map(list, zip(*flat_trajectory))
        N_samples = len(obs)
        ensemble_mask = list(np.random.multinomial(N_samples, np.ones(N_samples) / N_samples, size=num_critic).T) \
            if num_critic is not None else None
        return obs, actions, rewards, next_obs, dones, ensemble_mask

    @classmethod
    def group_element_trajectory(cls, trajectory):
        state_dim, action_dim = len(trajectory[0][0][0]), len(trajectory[0][0][1])
        N_trajectory = len(trajectory)
        traj_maxlen = max([len(traj) for traj in trajectory])
        obs_padded = np.zeros((N_trajectory, traj_maxlen, state_dim))
        actions_padded = np.zeros((N_trajectory, traj_maxlen, action_dim))
        mask_padded = np.zeros((N_trajectory, traj_maxlen, 1), dtype=np.bool)

        for traj_idx, trajectory in enumerate(trajectory):
            for t, (obs, action, reward, next_obs, done) in enumerate(trajectory):
                obs_padded[traj_idx, t, :] = obs
                actions_padded[traj_idx, t, :] = action
                mask_padded[traj_idx, t, 0] = True
        return obs_padded, actions_padded, mask_padded, traj_maxlen

    def __init__(self, trajectory, max_action, num_critic=None):
        self.num_critic = num_critic
        self.obs, self.action, self.reward, self.next_obs, self.done, self.ensemble_mask = \
            ReplayBuffer.group_elements(trajectory, num_critic)
        
        self.obs_mean = np.mean(self.obs, axis=0, keepdims=True)
        self.obs_std = np.std(self.obs, axis=0, keepdims=True) + 1e-3
        self.max_action = max_action

    def standardizer(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def sample(self, batch_size):
        obs, action, reward, next_obs, done, ensemble_mask = [], [], [], [], [], []
        indices = np.random.randint(0, len(self.obs), size=batch_size)
        for idx in indices:
            obs.append(self.obs[idx])
            action.append(self.action[idx])
            reward.append(self.reward[idx])
            next_obs.append(self.next_obs[idx])
            done.append(self.done[idx])
            if self.num_critic is not None:
                ensemble_mask.append(self.ensemble_mask[idx])
        obs, next_obs, action = np.array(obs), np.array(next_obs), np.array(action)

        obs, next_obs, action = self.standardizer(obs), self.standardizer(next_obs), action / self.max_action
        if self.num_critic is not None:
            return obs, action, np.array(reward)[:, None], next_obs, np.array(done)[:, None], np.array(ensemble_mask)
        else:
            return obs, action, np.array(reward)[:, None], next_obs, np.array(done)[:, None]


class Actor(tf.keras.layers.Layer):

    def __init__(self, action_dim, max_action, hidden_dim=64):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action

        # Actor parameters
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f0')
        self.l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f1')
        self.l3_mu = tf.keras.layers.Dense(action_dim, name='f2_mu')
        self.l3_log_std = tf.keras.layers.Dense(action_dim, name='f2_log_std')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.l1(obs)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)
        squahsed_action, squahsed_action_logp = apply_squashing_func(sampled_action, sampled_action_logp)

        return squahsed_action, squahsed_action_logp, dist


class VNetwork(tf.keras.layers.Layer):

    def __init__(self, output_dim=1, hidden_dim=64):
        super(VNetwork, self).__init__()

        self.v_l0 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f0')
        self.v_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='v/f1')
        self.v_l2 = tf.keras.layers.Dense(output_dim, name='v/f2')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.v_l0(obs)
        h = self.v_l1(h)
        v = self.v_l2(h)
        return v


class QNetwork(tf.keras.layers.Layer):

    def __init__(self, output_dim=1, num_critics=2, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f0' % i))
            self.qs_l1.append(tf.keras.layers.Dense(hidden_dim, activation='relu', name='q%d/f1' % i))
            self.qs_l2.append(tf.keras.layers.Dense(output_dim, name='q%d/f2' % i))

    def call(self, inputs, **kwargs):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)
        qs = []
        for i in range(self.num_critics):
            h = self.qs_l0[i](obs_action)
            h = self.qs_l1[i](h)
            q = self.qs_l2[i](h)
            qs.append(q)

        return qs


class KLAC:

    def _build_baseline_policy_and_kl(self, target_dist, obs_input, action_input):
        EPS = 1e-6
        self.behavior_policy = Actor(self.action_dim, self.max_action, hidden_dim=self.hidden_dim)
        _, behavior_action_logp, behavior_dist = self.behavior_policy([obs_input])
        before_squahed_action = tf.atanh(tf.clip_by_value(action_input, -1 + EPS, 1 - EPS))
        log_likelihood = behavior_dist.log_prob(before_squahed_action)
        log_likelihood -= tf.reduce_sum(tf.log(1 - action_input ** 2 + EPS), axis=1)
        behavior_loss = -tf.reduce_mean(log_likelihood)
        behavior_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        behavior_train_op = behavior_optimizer.minimize(behavior_loss, var_list=self.behavior_policy.trainable_variables)
        self.sess.run(tf.variables_initializer(behavior_optimizer.variables()))
        return target_dist.kl_divergence(behavior_dist)[:, None], behavior_train_op, behavior_loss

    def __init__(self, state_dim, action_dim, max_action, kl_coef, lamb=1, num_critics=5, gradient_norm_panelty=0, 
                 gradient_norm_limit=30, hidden_dim=64):
        self.state_dim, self.action_dim, self.max_action, self.lamb, self.num_critics, self.hidden_dim \
            = state_dim, action_dim, max_action, lamb, num_critics, hidden_dim
        self.gradient_norm_panelty, self.gradient_norm_limit = gradient_norm_panelty, gradient_norm_limit

        self.sess = tf.keras.backend.get_session()
        self.obs_ph, self.action_ph, self.reward_ph, self.terminal_ph, self.next_obs_ph \
            = [tf.keras.layers.Input(d) for d in [state_dim, action_dim, 1, 1, state_dim]]
        self.obs_mean, self.obs_std = tf.keras.layers.Input(state_dim), tf.keras.layers.Input(state_dim)
        self.ensemble_mask = tf.keras.layers.Input(self.num_critics)

        self.gamma, self.tau, self.learning_rate, self.batch_size = 0.99, 0.005, 3e-4, 64

        self.kl_coef = self._build_kl_coef(kl_coef)

        # Actor, Critic
        self.actor = actor = Actor(action_dim, max_action, hidden_dim=hidden_dim)
        self.action_pi, action_logp, dist = self.actor([self.obs_ph])

        # Baseline policy
        self.kl, behavior_train_op, behavior_loss = self._build_baseline_policy_and_kl(dist, self.obs_ph, self.action_ph)
        
        # Critic training
        train_tensors = [self.obs_ph, self.action_ph, self.next_obs_ph, self.terminal_ph, self.action_pi, self.ensemble_mask]
        self.critic_v, self.critic_q, mean_q_loss, qs_pi, critic_train_op, target_update_op = self._build_critic(
            train_tensors,  v_bonus=0,          q_bonus=self.reward_ph,     hidden_dim=self.hidden_dim)
        kl_critic_v, self.kl_critic_q, kl_mean_q_loss, kl_qs_pi, kl_critic_train_op, kl_target_update_op = self._build_critic(
            train_tensors,  v_bonus=-self.kl,   q_bonus=0,                  hidden_dim=self.hidden_dim)

        critic_train_op_group = tf.group([critic_train_op, kl_critic_train_op])
        target_update_op_group = tf.group([target_update_op, kl_target_update_op])

        # Actor training
        actor_loss = self.kl_coef * self.kl - self.kl_coef * tf.reduce_mean(kl_qs_pi, axis=0) - tf.reduce_mean(qs_pi, axis=0)
        actor_loss = tf.reduce_mean(actor_loss)
        actor_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=actor.trainable_variables)
        self.sess.run(tf.variables_initializer(actor_optimizer.variables()))

        self.step_ops = [critic_train_op_group, actor_train_op, target_update_op_group, behavior_train_op]

        self.eval_ops = [tf.reduce_mean(self.kl), actor_loss, mean_q_loss + kl_mean_q_loss,
                         tf.reduce_mean(qs_pi), tf.reduce_mean(kl_qs_pi), tf.reduce_mean(self.kl_coef), behavior_loss]
        self.eval_labels = ['kl', 'actor_loss', 'mean_q_loss', 'mean_q_r', 'mean_q_kl', 'kl_coef', 'behavior_loss']

        # For action selection
        self.sampled_action = self.action_pi
        self.sampled_action_q = tf.reduce_mean(qs_pi, axis=0)

    def _reduce_q(self, qs_pi):
        mean_q_pi = tf.reduce_mean(qs_pi, axis=0)
        var_q_pi = tf.reduce_mean([tf.square(x - mean_q_pi) for x in qs_pi], axis=0)
        std_q_pi = tf.sqrt(tf.maximum(var_q_pi, 1e-6))
        return mean_q_pi - self.lamb * std_q_pi

    def _build_critic(self, tensors, v_bonus, q_bonus, output_dim=1, hidden_dim=64, total_loss=False):
        """
        :param placeholders: [obs_ph, action_ph, next_obs_ph, terminal_ph, action_pi, ensemble_mask]
        :param v_bonus: is used to compute v_backup (for reward value: v_bonus=0, kl_value: v_bonus=-kl)
        :param q_bonus: is used to compute q_backup (for reward_value: q_bonus=r, kl_value: q_bonus=0)
        :param output_dim: integer
        :return: critic_v, critic_q, mean(q_losses), qs_pi, critic_train_op, target_update_op
        """
        obs_ph, action_ph, next_obs_ph, terminal_ph, action_pi, ensemble_mask = tensors

        # Define V and Q networks
        critic_v = VNetwork(output_dim, hidden_dim=hidden_dim)
        critic_q = QNetwork(output_dim, self.num_critics, hidden_dim=hidden_dim)
        critic_v_target = VNetwork(output_dim, hidden_dim=hidden_dim)

        # Critic training (V, Q)
        qs_pi = critic_q([obs_ph, action_pi])
        v = critic_v([obs_ph])
        qs = critic_q([obs_ph, action_ph])

        v_backup = tf.stop_gradient(self._reduce_q(qs_pi) + v_bonus)
        v_loss = tf.losses.mean_squared_error(v_backup, v)
        if total_loss and output_dim != 1:
            print('total_loss_added')
            v_loss = v_loss * output_dim + tf.losses.mean_squared_error(tf.reduce_sum(v_backup, axis=-1, keepdims=True), tf.reduce_sum(v, axis=-1, keepdims=True))

        # Gradient panelty (V)
        if self.gradient_norm_panelty > 0:
            v_grad_obs = tf.gradients(v, [obs_ph])[0]  # do not average, sum by the output dimension
            v_grad_norm = tf.sqrt(tf.reduce_sum(tf.square(v_grad_obs), axis=1) + 1e-8)
            v_grad_panelty_loss = tf.reduce_mean(tf.maximum(v_grad_norm - self.gradient_norm_limit * np.sqrt(self.state_dim), 0) ** 2)
            v_loss += self.gradient_norm_panelty * v_grad_panelty_loss

        value_target = critic_v_target([next_obs_ph])
        q_backup = tf.stop_gradient((1 - terminal_ph) * self.gamma * value_target + q_bonus)  # batch x 1
        q_losses = [tf.losses.mean_squared_error(q_backup, qs[k], weights=ensemble_mask[:, k:k+1]) for k in range(self.num_critics)]
        if total_loss and output_dim != 1:
            for k in range(self.num_critics):
                q_losses[k] = q_losses[k] * output_dim + tf.losses.mean_squared_error(
                    tf.reduce_sum(q_backup, axis=-1, keepdims=True), tf.reduce_sum(qs[k], axis=-1, keepdims=True), weights=ensemble_mask[:, k:k+1])

        # Gradient panelty (Q)
        if self.gradient_norm_panelty > 0:
            qs_grad_obs_action = [tf.concat(tf.gradients(q, [obs_ph, action_ph]), axis=-1) for q in qs]  # do not average, sum by the output dimension
            qs_grad_norm = [tf.sqrt(tf.reduce_sum(tf.square(q_grad_obs_action), axis=1) + 1e-8) for q_grad_obs_action in qs_grad_obs_action]
            qs_grad_panelty_loss = [tf.reduce_mean(tf.maximum(q_grad_norm - self.gradient_norm_limit * np.sqrt(self.state_dim + self.action_dim), 0) ** 2) for q_grad_norm in qs_grad_norm]
            for i, q_grad_panelty_loss in enumerate(qs_grad_panelty_loss):
                q_losses[i] += self.gradient_norm_panelty * q_grad_panelty_loss

        value_loss = v_loss + tf.reduce_sum(q_losses)
        critic_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        critic_train_op = critic_optimizer.minimize(value_loss, var_list=critic_v.trainable_variables + critic_q.trainable_variables)

        with tf.control_dependencies([critic_train_op]):
            # Update target network
            source_params = critic_v.trainable_variables
            target_params = critic_v_target.trainable_variables
            target_update_op = [
                tf.assign(target, (1 - self.tau) * target + self.tau * source)
                for target, source in zip(target_params, source_params)
            ]

        # Copy weights to target networks
        self.sess.run(tf.variables_initializer(critic_optimizer.variables()))
        critic_v_target.set_weights(critic_v.get_weights())

        return critic_v, critic_q, tf.reduce_mean(q_losses), qs_pi, critic_train_op, target_update_op

    def _build_kl_coef(self, init_kl_coef):
        return tf.constant(init_kl_coef, dtype=tf.float32)

    #######################################
    # interfaces for cont_run.py
    #######################################

    def batch_learn(self, trajectory, vec_env, total_timesteps, log_interval, seed, result_filepath=None, **kwargs):
        np.random.seed(seed)

        replay_buffer = ReplayBuffer(trajectory, max_action=self.max_action, num_critic=self.num_critics)
        self.standardizer = replay_buffer.standardizer

        # Start...
        start_time = time.time()
        eval_timesteps = []
        evaluations = []
        infos_values = []
        for timestep in tqdm(range(total_timesteps), desc="KLAC", ncols=70):
            obs, action, reward, next_obs, done, ensemble_mask = replay_buffer.sample(self.batch_size)
            feed_dict = {
                self.obs_ph: obs, self.action_ph: action, self.reward_ph: reward,
                self.next_obs_ph: next_obs, self.terminal_ph: done,
                self.obs_mean: replay_buffer.obs_mean, self.obs_std: replay_buffer.obs_std,
                self.ensemble_mask: ensemble_mask
            }
            step_result = self.sess.run(self.step_ops + self.eval_ops, feed_dict=feed_dict)
            infos_value = step_result[len(self.step_ops):]
            '''
            if timestep % log_interval == log_interval - 1:
                from plot_alpha import plot_value
                plot_value(self.sess, self.standardizer, self.obs_ph, 
                {'train_value': self.critic.v(self.obs_ph), 'density_estimation': self.denest([self.obs_ph])}, 
                replay_buffer.obs, timestep)
            '''
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

        return eval_timesteps, evaluations, infos_values

    def predict(self, obs, deterministic=False):
        assert len(obs.shape) == 2
        obs = self.standardizer(obs)
        obs_tile = np.tile(obs, (10, 1))
        action_tile, q1_tile = self.sess.run([self.sampled_action, self.sampled_action_q], feed_dict={
            self.obs_ph: obs_tile
        })
        actions = np.reshape(action_tile, (10, -1, self.action_dim))  # 10 x batch_size x da
        q1 = np.reshape(q1_tile, (10, -1, 1))  # 10 x batch_size x 1

        batch_size = obs.shape[0]
        indices = np.argmax(q1, axis=0)[:, 0]
        action = actions[indices, np.arange(batch_size), :]

        return action * self.max_action, None
