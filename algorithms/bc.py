import pickle
import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from cont_evaluate import evaluate_policy
from algorithms.klac import ReplayBuffer


class BC(tf.keras.layers.Layer):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=64):
        super(BC, self).__init__()
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        # self.graph = tf.Graph()
        # with self.graph.as_default():
        # policy network parameters
        self.obs_ph = tf.keras.layers.Input(state_dim)
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.out = tf.keras.layers.Dense(action_dim * 2)

        # feed forward
        h = self.dense1(self.obs_ph)
        h = self.dense2(h)
        out = self.out(h)
        self.policy_mu = out[:, :action_dim]
        self.policy_std = tf.exp(out[:, action_dim:])

        self.policy_dist = tfp.distributions.MultivariateNormalDiag(self.policy_mu, self.policy_std)
        self.sampled_action = self.policy_dist.sample()
        self.squashed_action = tf.tanh(self.sampled_action) * self.max_action

        def loss(y_true, y_pred):
            EPS = 1e-6
            before_squahed_action = tf.atanh(tf.clip_by_value(y_true / self.max_action, -1 + EPS, 1 - EPS))
            log_likelihood = self.policy_dist.log_prob(before_squahed_action)
            log_likelihood -= tf.reduce_sum(tf.log(1 - (y_true / self.max_action) ** 2 + EPS), axis=1)
            return -tf.reduce_mean(log_likelihood)

        self.model = tf.keras.models.Model(inputs=[self.obs_ph], outputs=[self.squashed_action])
        self.optimizer = tf.keras.optimizers.Adam(0.001)
        self.model.compile(optimizer=self.optimizer, loss=loss)

    def train(self, replay_buffer, iterations, batch_size=64):
        # Sample replay buffer / batch
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size * iterations)
        self.model.fit(x=obs, y=action, batch_size=batch_size, epochs=1)

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
        with tqdm(total=total_timesteps, desc="BC") as pbar:
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
        action = self.model.predict(obs)
        return action * self.max_action, None

    def get_parameters(self):
        parameters = []
        weights = self.get_weights()
        for idx, variable in enumerate(self.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def save(self, filepath):
        parameters = self.get_parameters()
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
