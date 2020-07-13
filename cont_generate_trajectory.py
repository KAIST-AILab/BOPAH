import os

import gym
import numpy as np
from stable_baselines.common import set_global_seeds
from tqdm import tqdm

from cont_train_policy import load_trained_agent
from util import precise_env_name


def generate_trajectory(env_name, trained_policy_seed, trained_policy_step, num_episodes, seed, hidden_dim=64):
    """
    :return: trajectory
    - list of [(obs, action, reward, next_obs, done), ... ]
    - len(trajectory): num_episodes
    - len(trajectory[0]): time steps of 0th episode
    """
    save_dir = "batch_trajectory/{}/seed_{}_hidden_{}/step_{}".format(env_name, trained_policy_seed, hidden_dim, trained_policy_step)
    os.makedirs(save_dir, exist_ok=True)

    trajectory_filepath = '%s/episode_%d_seed_%d.npy' % (save_dir, num_episodes, seed)
    if os.path.exists(trajectory_filepath):
        trajectory_result = np.load(trajectory_filepath, allow_pickle=True)
        print('Trajectory has already been generated: %s...' % trajectory_filepath)
    else:
        print('%s not exists... generate trajectories...' % trajectory_filepath)
        env = gym.make(precise_env_name(env_name))
        env.seed(seed)
        set_global_seeds(seed)
        if trained_policy_seed != 'uniform':
            trained_agent = load_trained_agent(env_name, trained_policy_seed, trained_policy_step, seed=seed, hidden_dim=hidden_dim)

        trajectory_result = []
        for episode in tqdm(range(num_episodes), desc='generate_trajectory', ncols=70):
            obs = env.reset()
            trajectory_one = []
            for t in range(10000):
                if trained_policy_seed != 'uniform':
                    action, _ = trained_agent.predict(obs, deterministic=False)
                else:
                    action = env.action_space.sample()
                next_obs, reward, done, info = env.step(action)

                terminal = done
                if info.get('TimeLimit.truncated'):
                    terminal = False
                trajectory_one.append((obs, action, reward, next_obs, terminal))
                if done:
                    break
                obs = next_obs
            trajectory_result.append(trajectory_one)
        trajectory_result = np.array(trajectory_result)
        np.save(trajectory_filepath, trajectory_result)

    return trajectory_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='halfcheetah')
    parser.add_argument("--trained_policy_seed", help="random seed of the trained policy", default=0, type=int)
    parser.add_argument("--trained_policy_step", help="time step of the trained policy", default=500000, type=int)
    parser.add_argument("--episode", help="the number of episodes to collect", default=1000, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    args = parser.parse_args()

    generate_trajectory(args.env_name, args.trained_policy_seed, args.trained_policy_step, args.episode, args.seed, hidden_dim=100)
