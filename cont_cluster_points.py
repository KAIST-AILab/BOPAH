import os

import gym
import numpy as np
import scipy
from sklearn import mixture
from stable_baselines.common import set_global_seeds
from tqdm import tqdm

from cont_train_policy import load_trained_agent
from cover_tree import CoverTree
from util import precise_env_name


def generate_cluster(env_name, trained_policy_seed, trained_policy_step, num_episodes, num_clusters, seed, hidden_dim=64):
    """
    :return: cluster infos
    - list of [(obs, action, reward, next_obs, done), ... ]
    - len(trajectory): num_episodes
    - len(trajectory[0]): time steps of 0th episode
    """
    save_dir = "batch_trajectory/{}/seed_{}_hidden_{}/step_{}".format(env_name, trained_policy_seed, hidden_dim, trained_policy_step)
    os.makedirs(save_dir, exist_ok=True)

    trajectory_filepath = '%s/episode_%d_seed_%d.npy' % (save_dir, num_episodes, seed)
    cluster_filepath = '%s/episode_%d_seed_%d_clusters_%d.npy' % (save_dir, num_episodes, seed, num_clusters)
    if os.path.exists(cluster_filepath):
        cluster_result = np.load(cluster_filepath, allow_pickle=True)[()]
        print('Clusters has already been generated: %s...' % cluster_filepath)
    else:
        print('%s not exists... generate clusters...' % cluster_filepath)
        env = gym.make(precise_env_name(env_name))
        env.seed(seed)
        set_global_seeds(seed)
        
        trajectory = np.load(trajectory_filepath, allow_pickle=True)
        obs = []
        for traj in trajectory:
            for (o, a, r, no, d) in traj:
                obs.append(o)
        obs_mean = np.mean(obs, axis=0, keepdims=True)
        obs_std = np.std(obs, axis=0, keepdims=True) + 1e-3
        stan_obs = (obs - obs_mean) / obs_std
        np.random.shuffle(stan_obs)

        import time
        startime = time.time()
        covertree = CoverTree(stan_obs[:10000], scipy.spatial.distance.euclidean, leafsize=10)
        print('used_time: {}'.format(time.time() - startime))
        print(covertree.root.ctr_idx)
        current_parents = [covertree.root]
        next_parents = []
        representatives = set([])
        candidates = []
        while len(representatives) < num_clusters:
            if not candidates:
                for child in current_parents[0].children:
                    if isinstance(child, CoverTree._LeafNode):
                        candidates.append(child)
                    else:
                        current_parents.append(child)
                representatives.add(current_parents.pop(0).ctr_idx)
            else:
                representatives.add(candidates.pop().ctr_idx)
        print(representatives)
        cluster_result = {'representatives': stan_obs[list(representatives)]}
        np.save(cluster_filepath, cluster_result)
    return cluster_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='ant')
    parser.add_argument("--trained_policy_seed", help="random seed of the trained policy", default=0, type=int)
    parser.add_argument("--trained_policy_step", help="time step of the trained policy", default=500000, type=int)
    parser.add_argument("--episode", help="the number of episodes to collect", default=1000, type=int)
    parser.add_argument("--clusters", help="the number of clusters to use", default=20, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    args = parser.parse_args()

    generate_cluster(args.env_name, args.trained_policy_seed, args.trained_policy_step, args.episode, args.clusters, args.seed, hidden_dim=100)
