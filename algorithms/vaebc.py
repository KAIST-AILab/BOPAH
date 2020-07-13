import time

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tqdm import tqdm
from algorithms.klac import ReplayBuffer

from cont_evaluate import evaluate_policy


class VAEBC:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = action_dim * 2
        self.max_action = max_action

        self.obs_ph = tf.keras.layers.Input(state_dim)
        self.action_ph = tf.keras.layers.Input(action_dim)

        # Encoder parameters
        self.e1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='e1')
        self.e2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='e2')
        self.e_mean = tf.keras.layers.Dense(self.latent_dim, name='e_man')
        self.e_log_std = tf.keras.layers.Dense(self.latent_dim, name='e_log_std')

        # Decoder parameters
        self.d1 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='d1')
        self.d2 = tf.keras.layers.Dense(hidden_dim, activation='relu', name='d2')
        self.d3 = tf.keras.layers.Dense(self.action_dim, name='d3')

        # Forward
        self.z, mean, std, self.encoder_dist = self.encode(self.obs_ph, self.action_ph)
        self.prior_dist = tfp.distributions.MultivariateNormalDiag(tf.zeros((tf.shape(self.obs_ph)[0], self.latent_dim)), tf.ones((tf.shape(self.obs_ph)[0], self.latent_dim)))
        self.kl = self.encoder_dist.kl_divergence(self.prior_dist)
        self.action_recon = self.decode(self.obs_ph, self.z)
        self.action_sampled = self.decode(self.obs_ph)

        def loss(action_true, y_pred):
            recon_loss = tf.losses.mean_squared_error(action_true, self.action_recon)
            return recon_loss + 0.5 * tf.reduce_mean(self.kl, axis=0)

        self.model = tf.keras.models.Model(inputs=[self.obs_ph, self.action_ph], outputs=[self.action_sampled + 0 * tf.reduce_mean(self.z)])
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.model.compile(optimizer=self.optimizer, loss=loss)

    def encode(self, obs, action):
        h = self.e1(tf.concat([obs, action], axis=1))
        h = self.e2(h)
        mean = self.e_mean(h)
        std = tf.exp(self.e_log_std(h))
        encoder_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        z = encoder_dist.sample()
        return z, mean, std, encoder_dist

    def decode(self, obs, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = tf.clip_by_value(self.prior_dist.sample(), -0.5, 0.5)
        h = self.d1(tf.concat([obs, z], axis=1))
        h = self.d2(h)
        action = tf.tanh(self.d3(h)) * self.max_action
        return action

    def train(self, replay_buffer, iterations, batch_size=64):
        # Sample replay buffer / batch
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size * iterations)
        self.model.fit(x=[obs, action], y=[action], batch_size=batch_size, epochs=1)

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
        sess = tf.keras.backend.get_session()
        obs = self.standardizer(obs)
        action = sess.run(self.action_sampled, feed_dict={self.obs_ph: obs})
        return action * self.max_action, None
