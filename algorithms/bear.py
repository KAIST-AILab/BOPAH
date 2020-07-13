import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from cont_evaluate import evaluate_policy
from algorithms.klac import ReplayBuffer
from algorithms.sac import apply_squashing_func


class RegularActor(tf.keras.layers.Layer):

    def __init__(self, action_dim, hidden_dim=64):
        super(RegularActor, self).__init__()
        self.action_dim = action_dim

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

    def sample_multiple(self, obs, num_sample=10):
        obs_tile = tf.tile(obs, (num_sample, 1))
        h = self.l1(obs_tile)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action_tile = dist.sample()
        # This trick stabilizes learning (clipping gaussian to a smaller range)
        sampled_action_tile = tf.clip_by_value(sampled_action_tile, mean - 0.5 * std, mean + 0.5 * std)

        sampled_action_tile_logp = dist.log_prob(sampled_action_tile)
        squahsed_action_tile, squahsed_action_logp = apply_squashing_func(sampled_action_tile, sampled_action_tile_logp)

        actions = tf.reshape(squahsed_action_tile, (num_sample, -1, self.action_dim))
        raw_actions = tf.reshape(sampled_action_tile, (num_sample, -1, self.action_dim))

        return actions, raw_actions


class EnsembleCritic(tf.keras.layers.Layer):

    def __init__(self, hidden_dim=64):
        super(EnsembleCritic, self).__init__()

        # Critic parameters
        self.q1_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q1/f1')
        self.q1_l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q1/f2')
        self.q1_l3 = tf.keras.layers.Dense(1, name='q1/f3')

        self.q2_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q2/f1')
        self.q2_l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q2/f2')
        self.q2_l3 = tf.keras.layers.Dense(1, name='q2/f3')

    def call(self, inputs, with_var=False, **kwargs):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)

        h = self.q1_l1(obs_action)
        h = self.q1_l2(h)
        q1 = self.q1_l3(h)

        h = self.q2_l1(obs_action)
        h = self.q2_l2(h)
        q2 = self.q2_l3(h)

        all_qs = [q1, q2]
        if with_var:
            std_q = tf.math.reduce_std(all_qs, axis=0)
            return all_qs, std_q
        else:
            return all_qs


class VAE(tf.keras.layers.Layer):

    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim=64):
        super(VAE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder parameters
        self.e1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='encoder/f1')
        self.e2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='encoder/f2')
        self.e_mean = tf.keras.layers.Dense(self.latent_dim, name='encoder/f3_mean')
        self.e_log_std = tf.keras.layers.Dense(self.latent_dim, name='encoder/f3_logstd')

        # Decoder parameters
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='decoder/f1')
        self.d2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='decoder/f2')
        self.d3 = tf.keras.layers.Dense(self.action_dim, name='decoder/f3')

    def encode(self, obs, action):
        h = self.e1(tf.concat([obs, action], axis=1))
        h = self.e2(h)
        mean = self.e_mean(h)
        std = tf.exp(self.e_log_std(h))
        encoder_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        z = encoder_dist.sample()
        return z, mean, std, encoder_dist

    def decode(self, obs, z=None):
        if z is None:
            batch_size = tf.shape(obs)[0]
            prior_dist = tfp.distributions.MultivariateNormalDiag(tf.zeros((batch_size, self.latent_dim)), tf.ones((batch_size, self.latent_dim)))
            # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
            z = tf.clip_by_value(prior_dist.sample(), -0.5, 0.5)

        h = self.d1(tf.concat([obs, z], axis=1))
        h = self.d2(h)
        action = tf.tanh(self.d3(h))
        return action

    def decode_multiple(self, obs, z=None, num_decode=10):
        """Decode 10 samples at least"""
        obs_tile = tf.tile(obs, (num_decode, 1))
        if z is None:
            tiled_batch_size = tf.shape(obs_tile)[0]
            prior_dist = tfp.distributions.MultivariateNormalDiag(tf.zeros((tiled_batch_size, self.latent_dim)), tf.ones((tiled_batch_size, self.latent_dim)))
            # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
            z_tile = tf.clip_by_value(prior_dist.sample(), -0.5, 0.5)

        h = self.d1(tf.concat([obs_tile, z_tile], axis=1))
        h = self.d2(h)
        raw_actions = tf.reshape(self.d3(h), (num_decode, -1, self.action_dim))
        actions = tf.tanh(raw_actions)  # num_decode x batch_size x da
        return actions, raw_actions

    def call(self, inputs, **kwargs):
        obs, action = inputs
        batch_size = tf.shape(obs)[0]
        prior_dist = tfp.distributions.MultivariateNormalDiag(tf.zeros((batch_size, self.latent_dim)), tf.ones((batch_size, self.latent_dim)))

        # Forward
        z, mean, std, encoder_dist = self.encode(obs, action)
        kl = encoder_dist.kl_divergence(prior_dist)
        action_recon = self.decode(obs, z)
        action_sampled = self.decode(obs)
        return z, kl, action_recon, action_sampled


class BEAR(object):
    def __init__(self, state_dim, action_dim, max_action, delta_conf=0.1, use_bootstrap=True, version=0, lamb=0.0,
                 threshold=0.05, mode='auto', num_samples_match=5, mmd_sigma=10.0, lagrange_thresh=10.0, use_ensemble=False, kernel_type='laplacian', hidden_dim=64):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        latent_dim = action_dim * 2
        self.delta_conf = delta_conf
        self.use_bootstrap = use_bootstrap
        self.version = version
        self.lamb = lamb
        self.threshold = threshold
        self.mode = mode
        self.num_samples_match = num_samples_match
        self.mmd_sigma = mmd_sigma
        self.lagrange_thresh = lagrange_thresh
        self.use_ensemble = use_ensemble
        self.kernel_type = kernel_type

        if self.mode == 'auto':
            # Use lagrange multipliers on the constraint if set to auto mode 
            # for the purpose of maintaing support matching at all times
            self.log_lagrange2 = tf.Variable(np.random.randn())

        self.obs_ph = tf.keras.layers.Input(state_dim, name='obs_ph')
        self.action_ph = tf.keras.layers.Input(action_dim, name='action_ph')
        self.reward_ph = tf.keras.layers.Input(1, name='reward_ph')
        self.terminal_ph = tf.keras.layers.Input(1, name='terminal_ph')
        self.next_obs_ph = tf.keras.layers.Input(state_dim, name='next_obs_ph')
        self.mask_ph = tf.keras.layers.Input(2, name='mask_ph')

        self.gamma = 0.99
        self.tau = 0.005

        self.step_number = tf.Variable(0.0)

        # Actor, Critic, VAE
        self.actor = RegularActor(action_dim, hidden_dim=hidden_dim)
        self.actor_target = RegularActor(action_dim, hidden_dim=hidden_dim)
        self.critic = EnsembleCritic(hidden_dim=hidden_dim)
        self.critic_target = EnsembleCritic(hidden_dim=hidden_dim)
        self.vae = VAE(state_dim, action_dim, latent_dim, hidden_dim=hidden_dim)

        # Train the Behaviour cloning policy to be able to take more than 1 sample for MMD
        z, kl, vae_action_recon, vae_action_sampled = self.vae([self.obs_ph, self.action_ph])
        vae_loss = tf.losses.mean_squared_error(self.action_ph, vae_action_recon) + 0.5 * tf.reduce_mean(kl)
        vae_optimizer = tf.train.AdamOptimizer(0.001)
        vae_train_op = vae_optimizer.minimize(vae_loss, var_list=self.vae.trainable_variables)

        # Critic training: In this step, we explicitly compute the actions
        # Duplicate sate 10 times (10 is a hyperparameter chosen by BCQ)
        next_state_tile = tf.tile(self.next_obs_ph, (10, 1))

        # Compute value of perturbed actions sampled from the VAE
        next_action_tile, _, _ = self.actor_target([next_state_tile])
        target_qs = self.critic_target([next_state_tile, next_action_tile])

        # Soft-convex combination for target values
        target_q = 0.75 * tf.reduce_min(target_qs, axis=0) + 0.25 * tf.reduce_max(target_qs, axis=0)
        target_q = tf.reduce_max(tf.reshape(target_q, (10, -1, 1)), axis=0)
        target_q = self.reward_ph + (1 - self.terminal_ph) * self.gamma * target_q

        current_qs = self.critic([self.obs_ph, self.action_ph], with_var=False)
        if self.use_bootstrap:
            critic_loss = tf.reduce_mean(tf.losses.mean_squared_error(current_qs[0], target_q, reduction='none') * self.mask_ph[:, 0:1]) +\
                          tf.reduce_mean(tf.losses.mean_squared_error(current_qs[1], target_q, reduction='none') * self.mask_ph[:, 1:2])
        else:
            critic_loss = tf.losses.mean_squared_error(current_qs[0], target_q) + tf.losses.mean_squared_error(current_qs[1], target_q)
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_train_op = critic_optimizer.minimize(critic_loss, var_list=self.critic.trainable_variables)

        # Action Training
        # If you take less samples (but not too less, else it becomes statistically inefficient), it is closer to a uniform support set matching
        num_samples = self.num_samples_match

        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(self.obs_ph, num_decode=num_samples)  # num_samples x batch x da
        actor_actions, raw_actor_actions = self.actor.sample_multiple(self.obs_ph, num_sample=num_samples)  # num_samples x batch x da

        # MMD done on raw actions (before tanh), to prevent gradient dying out due to saturation
        if self.kernel_type == 'gaussian':
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        # Update through TD3 style
        critic_qs = self.critic([tf.tile(self.obs_ph, (num_samples, 1)), tf.reshape(actor_actions, (-1, self.action_dim))])  # num_qs x (num_samples x batch_size) x 1
        critic_qs = [tf.reshape(critic_q, (num_samples, -1, 1)) for critic_q in critic_qs]  # num_qs x num_samples x batch_size x 1
        critic_qs = tf.reduce_mean(critic_qs, axis=1)  # num_qs x batch_size x 1
        std_q = tf.math.reduce_std(critic_qs, axis=0)  # batch_size x 1

        if not self.use_ensemble:
            std_q = tf.zeros_like(std_q)

        if self.version == 0:
            critic_qs = tf.reduce_min(critic_qs, axis=0)
        elif self.version == 1:
            critic_qs = tf.reduce_max(critic_qs, axis=0)
        elif self.version == 2:
            critic_qs = tf.reduce_mean(critic_qs, axis=0)

        # We do support matching with a warmstart which happens to be reasonable around epoch 20 during training
        if self.mode == 'auto':
            actor_loss = tf.cond(self.step_number >= 20000,
                                lambda: tf.reduce_mean(-critic_qs + self.lamb * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + tf.exp(self.log_lagrange2) * mmd_loss),
                                lambda: tf.reduce_mean(tf.exp(self.log_lagrange2) * mmd_loss))
        else:
            # This coefficient (100.0) is hardcoded, and is different for different tasks. I would suggest using auto, as that is the one used in the paper and works better.
            actor_loss = tf.cond(self.step_number >= 20000,
                                lambda: tf.reduce_mean(-critic_qs + self.lamb * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + 100.0 * mmd_loss),
                                lambda: 100.0 * tf.reduce_mean(mmd_loss))

        std_loss = self.lamb * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * tf.stop_gradient(std_q)  # only for log

        actor_optimizer = tf.train.AdamOptimizer(0.001)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.actor.trainable_variables)

        # Threshold for the lagrange multiplier

        if self.mode == 'auto':
            lagrange_loss = tf.reduce_mean(-critic_qs + self.lamb * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + tf.exp(self.log_lagrange2) * (mmd_loss - self.threshold))

            lagrange2_optimizer = tf.train.AdamOptimizer(0.001)
            lagrange_train_op = lagrange2_optimizer.minimize(-lagrange_loss, var_list=[self.log_lagrange2])
            with tf.control_dependencies([lagrange_train_op]):
                lagrange_clip_op = tf.assign(self.log_lagrange2, tf.clip_by_value(self.log_lagrange2, -5.0, self.lagrange_thresh))
            lagrange_train_op = tf.group([lagrange_train_op, lagrange_clip_op])


        # Update target network
        source_params = self.critic.trainable_variables + self.actor.trainable_variables
        target_params = self.critic_target.trainable_variables + self.actor_target.trainable_variables
        target_update_op = [
            tf.assign(target, (1 - self.tau) * target + self.tau * source)
            for target, source in zip(target_params, source_params)
        ]

        # Copy weights to target networks
        initializing_variables = vae_optimizer.variables() + critic_optimizer.variables() + actor_optimizer.variables() + lagrange2_optimizer.variables() + [self.log_lagrange2, self.step_number]
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.variables_initializer(initializing_variables))
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_target.set_weights(self.actor.get_weights())

        increase_step_number_op = tf.assign(self.step_number, self.step_number + 1)
        self.step_ops = [vae_train_op, critic_train_op, actor_train_op, lagrange_train_op, target_update_op, increase_step_number_op]

        # For action selection
        self.sampled_action, _, _ = self.actor([self.obs_ph])
        self.sampled_action_q1, _ = self.critic([self.obs_ph, self.sampled_action])
    
    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        """
        MMD constraint with Laplacian kernel for support matching

        samples1: 10 x batch_size x da
        samples2: 10 x batch_size x da
        """
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        samples1 = tf.transpose(samples1, (1, 0, 2))
        samples2 = tf.transpose(samples2, (1, 0, 2))

        diff_x_x = tf.expand_dims(samples1, 2) - tf.expand_dims(samples1, 1)  # B x N x N x d
        diff_x_x = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_x_x), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        diff_x_y = tf.expand_dims(samples1, 2) - tf.expand_dims(samples2, 1)
        diff_x_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_x_y), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        diff_y_y = tf.expand_dims(samples2, 2) - tf.expand_dims(samples2, 1)
        diff_y_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.abs(diff_y_y), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        overall_loss = tf.sqrt(diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6)
        return overall_loss
    
    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        """
        MMD constraint with Gaussian Kernel support matching

        samples1: 10 x batch_size x da
        samples2: 10 x batch_size x da
        """
        # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
        samples1 = tf.transpose(samples1, (1, 0, 2))
        samples2 = tf.transpose(samples2, (1, 0, 2))

        diff_x_x = tf.expand_dims(samples1, 2) - tf.expand_dims(samples1, 1)  # B x N x N x d
        diff_x_x = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.square(diff_x_x), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        diff_x_y = tf.expand_dims(samples1, 2) - tf.expand_dims(samples2, 1)
        diff_x_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.square(diff_x_y), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        diff_y_y = tf.expand_dims(samples2, 2) - tf.expand_dims(samples2, 1)
        diff_y_y = tf.reduce_mean(tf.exp(-tf.reduce_sum(tf.square(diff_y_y), axis=-1) / (2.0 * sigma)), axis=(1, 2))

        overall_loss = tf.sqrt(diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6)
        return overall_loss

    def train(self, replay_buffer, iterations, batch_size=64):

        for it in tqdm(range(iterations), desc="BEAR_iteration", ncols=70):
            obs, action, reward, next_obs, terminal, mask = replay_buffer.sample(batch_size)

            _, log_lagrange2_value = self.sess.run([self.step_ops, self.log_lagrange2], feed_dict={
                self.obs_ph: obs,
                self.action_ph: action,
                self.reward_ph: reward,
                self.next_obs_ph: next_obs,
                self.terminal_ph: terminal,
                self.mask_ph: mask
            })
            if it % 1000 == 0:
                print('=================')
                print('log_lagrange_value: {}'.format(log_lagrange2_value))

    #######################################
    # interfaces for cont_run.py
    #######################################

    def batch_learn(self, trajectory, vec_env, total_timesteps, log_interval, seed, result_filepath=None, **kwargs):
        np.random.seed(seed)

        replay_buffer = ReplayBuffer(trajectory, max_action=self.max_action, num_critic=2)
        self.standardizer = replay_buffer.standardizer

        # Start...
        start_time = time.time()
        timestep = 0
        eval_timesteps = []
        evaluations = []
        with tqdm(total=total_timesteps, desc="BEAR", ncols=70) as pbar:
            while timestep < total_timesteps:
                evaluation = evaluate_policy(vec_env, self)
                eval_timesteps.append(timestep)
                evaluations.append(evaluation)
                print('t=%d: %f (elapsed_time=%f)' % (timestep, evaluation, time.time() - start_time))

                self.train(replay_buffer, iterations=log_interval, batch_size=64)
                pbar.update(log_interval)
                timestep += log_interval

                if result_filepath:
                    result = {'eval_timesteps': eval_timesteps, 'evals': evaluations, 'info_values': []}
                    np.save(result_filepath + '.tmp.npy', result)

        return eval_timesteps, evaluations, []

    def predict(self, obs, deterministic=False):
        assert len(obs.shape) == 2
        obs = self.standardizer(obs)
        obs_tile = np.tile(obs, (10, 1))
        action_tile, q1_tile = self.sess.run([self.sampled_action, self.sampled_action_q1], feed_dict={
            self.obs_ph: obs_tile
        })
        actions = np.reshape(action_tile, (10, -1, self.action_dim))  # 10 x batch_size x da
        q1 = np.reshape(q1_tile, (10, -1, 1))  # 10 x batch_size x 1

        batch_size = obs.shape[0]
        indices = np.argmax(q1, axis=0)[:, 0]
        action = actions[indices, np.arange(batch_size), :]

        return action * self.max_action, None
