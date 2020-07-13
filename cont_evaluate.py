import multiprocessing
import time

import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import SubprocVecEnv
from tqdm import tqdm

from util import precise_env_name


def make_vectorized_env(env_name, n_envs=multiprocessing.cpu_count()):
    def make_env(env_id, seed=0):
        def _init():
            env = gym.make(env_id)
            env.seed(seed)
            return env

        set_global_seeds(seed)
        return _init

    vec_env = SubprocVecEnv([make_env(precise_env_name(env_name), i) for i in range(n_envs)])
    return vec_env


def evaluate_policy(vec_env, agent, num_episodes=30, deterministic=False, render=False):
    episode_rewards = []

    with tqdm(total=num_episodes, desc="policy_evaluation", ncols=70) as pbar:
        episode_reward = np.zeros(vec_env.num_envs)
        obs = vec_env.reset()
        while len(episode_rewards) < num_episodes:
            action, _ = agent.predict(obs, deterministic=deterministic)
            next_obs, reward, done, _ = vec_env.step(action)
            # print(obs[0], action[0], reward[0])

            episode_reward = episode_reward + reward
            if np.count_nonzero(done) > 0:
                episode_rewards += list(episode_reward[done])
                episode_reward[done] = 0
                pbar.update(np.count_nonzero(done))

            obs = next_obs

            if render:
                vec_env.render()
                time.sleep(0.1)

        episode_rewards = np.array(episode_rewards)

    mu = np.mean(episode_rewards)
    ste = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
    print("\n%f +- %f" % (mu, ste))

    return np.mean(episode_rewards)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='halfcheetah')
    parser.add_argument("--trained_policy_seed", help="random seed of the trained policy", default=0, type=int)
    parser.add_argument("--trained_policy_step", help="time step of the trained policy", default=500000, type=int)
    args = parser.parse_args()

    from cont_train_policy import load_trained_agent
    vec_env = make_vectorized_env(args.env_name)
    trained_agent = load_trained_agent(args.env_name, args.trained_policy_seed, args.trained_policy_step)

    deterministic = False
    evaluation = evaluate_policy(vec_env, trained_agent, deterministic=deterministic)
    print('env: %s' % args.env_name)
    print('trained_policy_seed: %d' % args.trained_policy_seed)
    print('trained_policy_step: %d' % args.trained_policy_step)
    print('deterministic: %s' % deterministic)
    print('reward: %f' % evaluation)
