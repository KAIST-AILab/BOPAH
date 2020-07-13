import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from algorithms.klac import ReplayBuffer
from cont_evaluate import evaluate_policy


class Actor(tf.keras.layers.Layer):

    def __init__(self, action_dim, max_action, hidden_dim=64, perturb=0.05):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.perturb=0.05

        # Actor parameters
        self.l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f1')
        self.l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='f2')
        self.l3 = tf.keras.layers.Dense(action_dim, name='f3')

    def call(self, inputs, **kwargs):
        obs, action = inputs
        h = self.l1(tf.concat([obs, action], axis=1))
        h = self.l2(h)
        perturb = self.perturb * self.max_action * tf.tanh(self.l3(h))
        return tf.clip_by_value(action + perturb, -self.max_action, self.max_action)


class Critic(tf.keras.layers.Layer):

    def __init__(self, hidden_dim=64):
        super(Critic, self).__init__()

        # Critic parameters
        self.q1_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q1/f1')
        self.q1_l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q1/f2')
        self.q1_l3 = tf.keras.layers.Dense(1, name='q1/f3')

        self.q2_l1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q2/f1')
        self.q2_l2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='q2/f2')
        self.q2_l3 = tf.keras.layers.Dense(1, name='q2/f3')

    def call(self, inputs, **kwargs):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)

        h = self.q1_l1(obs_action)
        h = self.q1_l2(h)
        q1 = self.q1_l3(h)

        h = self.q2_l1(obs_action)
        h = self.q2_l2(h)
        q2 = self.q2_l3(h)

        return q1, q2


class VAE(tf.keras.layers.Layer):

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=64):
        super(VAE, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_action = max_action

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
        action = tf.tanh(self.d3(h)) * self.max_action
        return action

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


class BCQ(object):
    def __init__(self, state_dim, action_dim, max_action, trajectory, hidden_dim=64, perturb=0.05):
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim
        latent_dim = action_dim * 2

        self.obs_ph = tf.keras.layers.Input(state_dim)
        self.action_ph = tf.keras.layers.Input(action_dim)
        self.reward_ph = tf.keras.layers.Input(1)
        self.terminal_ph = tf.keras.layers.Input(1)
        self.next_obs_ph = tf.keras.layers.Input(state_dim)
        self.gamma = 0.99
        self.tau = 0.005

        # Actor, Critic, VAE
        self.actor = Actor(action_dim, max_action, hidden_dim=hidden_dim, perturb=perturb)
        self.actor_target = Actor(action_dim, max_action, hidden_dim=hidden_dim)
        self.critic = Critic(hidden_dim=hidden_dim)
        self.critic_target = Critic(hidden_dim=hidden_dim)
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=hidden_dim)

        # VAE
        z, kl, vae_action_recon, vae_action_sampled = self.vae([self.obs_ph, self.action_ph])
        vae_loss = tf.losses.mean_squared_error(self.action_ph, vae_action_recon) + 0.5 * tf.reduce_mean(kl)
        vae_optimizer = tf.train.AdamOptimizer(0.001)
        vae_train_op = vae_optimizer.minimize(vae_loss, var_list=self.vae.trainable_variables)

        # Critic training
        next_state_tile = tf.tile(self.next_obs_ph, (10, 1))
        target_action = self.actor_target([next_state_tile, self.vae.decode(next_state_tile)])
        target_q1, target_q2 = self.critic_target([next_state_tile, target_action])

        target_q = 0.75 * tf.reduce_min([target_q1, target_q2], axis=0) + 0.25 * tf.reduce_max([target_q1, target_q2], axis=0)
        target_q = tf.reduce_max(tf.reshape(target_q, (10, -1, 1)), axis=0)
        target_q = self.reward_ph + (1 - self.terminal_ph) * self.gamma * target_q

        current_q1, current_q2 = self.critic([self.obs_ph, self.action_ph])
        critic_loss = tf.reduce_sum([tf.losses.mean_squared_error(target_q, current_q1), tf.losses.mean_squared_error(target_q, current_q2)])
        critic_optimizer = tf.train.AdamOptimizer(0.001)
        critic_train_op = critic_optimizer.minimize(critic_loss, var_list=self.critic.trainable_variables)

        with tf.control_dependencies([critic_train_op]):
            # Actor (perturbation model) training
            perturbed_action = self.actor([self.obs_ph, vae_action_sampled])
            q1, _ = self.critic([self.obs_ph, perturbed_action])
            actor_loss = -tf.reduce_mean(q1)
            actor_optimizer = tf.train.AdamOptimizer(0.001)
            actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.actor.trainable_variables)

            with tf.control_dependencies([actor_train_op]):
                # Update target network
                source_params = self.critic.trainable_variables + self.actor.trainable_variables
                target_params = self.critic_target.trainable_variables + self.actor_target.trainable_variables
                target_update_op = [
                    tf.assign(target, (1 - self.tau) * target + self.tau * source)
                    for target, source in zip(target_params, source_params)
                ]

        # Copy weights to target networks
        optimizer_variables = vae_optimizer.variables() + critic_optimizer.variables() + actor_optimizer.variables()
        self.sess = tf.keras.backend.get_session()
        self.sess.run(tf.variables_initializer(optimizer_variables))
        self.critic_target.set_weights(self.critic.get_weights())
        self.actor_target.set_weights(self.actor.get_weights())

        self.step_ops = [vae_train_op, critic_train_op, actor_train_op, target_update_op]

        # For action selection
        self.sampled_action = self.actor([self.obs_ph, self.vae.decode(self.obs_ph)])
        self.sampled_action_q1, _ = self.critic([self.obs_ph, self.sampled_action])

    def train(self, replay_buffer, iterations, batch_size=64):

        for it in tqdm(range(iterations), desc="BCQ_iteration"):
            obs, action, reward, next_obs, terminal = replay_buffer.sample(batch_size)

            self.sess.run(self.step_ops, feed_dict={
                self.obs_ph: obs,
                self.action_ph: action,
                self.reward_ph: reward,
                self.next_obs_ph: next_obs,
                self.terminal_ph: terminal
            })

    #######################################
    # interfaces for cont_run.py
    #######################################

    def batch_learn(self, trajectory, vec_env, total_timesteps, log_interval, seed, result_filepath=None, **kwargs):
        np.random.seed(seed)

        replay_buffer = ReplayBuffer(trajectory, max_action=self.max_action)
        self.standardizer = replay_buffer.standardizer

        # Start...
        start_time = time.time()
        timestep = 0
        eval_timesteps = []
        evaluations = []
        with tqdm(total=total_timesteps, desc="VAEBC") as pbar:
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
