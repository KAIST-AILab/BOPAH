import os

import gym
import numpy as np

from algorithms.bc import BC
from algorithms.bcq import BCQ
from algorithms.bear import BEAR
from algorithms.bopah import BOPAH
from algorithms.bopah_single import BOPAHSingle
from algorithms.klac import KLAC
from algorithms.vaebc import VAEBC
from cont_cluster_points import generate_cluster
from cont_evaluate import make_vectorized_env
from cont_generate_trajectory import generate_trajectory
from cont_train_policy import load_trained_agent
from util import precise_env_name

np.set_printoptions(precision=3, suppress=True, linewidth=250)


def run(env_name, trained_policy_seed, trained_policy_step, trajectory_episode, trajectory_seed, alg, total_timesteps, seed, alg_params={}):

    hidden_dim = 64 if alg_params.get('hidden_dim') is None else alg_params['hidden_dim']

    env = gym.make(precise_env_name(env_name))
    state_dim, action_dim, max_action = env.observation_space.shape[0], env.action_space.shape[0], float(env.action_space.high[0])
    trained_agent = load_trained_agent(env_name, trained_policy_seed, trained_policy_step, hidden_dim=hidden_dim)
    parameters = trained_agent.get_parameters()

    # Load trajectory & train/valid split
    split_ratio = 0.8
    trajectory_all = generate_trajectory(env_name, trained_policy_seed, trained_policy_step, trajectory_episode, trajectory_seed, hidden_dim=hidden_dim)
    trajectory_train = trajectory_all[:int(len(trajectory_all) * split_ratio)]
    trajectory_valid = trajectory_all[int(len(trajectory_all) * split_ratio):]

    log_interval = 10000  # max(100, total_timesteps // 300)

    alg_name, trajectory = alg, trajectory_all
    batch_trajectory = None
    # Load model
    if alg == 'bc':
        model = BC(state_dim, action_dim, max_action, hidden_dim=hidden_dim)
    elif alg == 'vaebc':
        model = VAEBC(state_dim, action_dim, max_action, hidden_dim=hidden_dim)
    elif alg == 'klac':
        kl_coef, gradient_norm_panelty, gradient_norm_limit = alg_params['kl_coef'], alg_params['gradient_norm_panelty'], alg_params['gradient_norm_limit']
        alg_name = 'klac_klcoef_{}_grad_norm_panelty_{}_grad_norm_limit_{}'.format(kl_coef, gradient_norm_panelty, gradient_norm_limit)
        model = KLAC(state_dim, action_dim, max_action, kl_coef=kl_coef, gradient_norm_panelty=gradient_norm_panelty, gradient_norm_limit=gradient_norm_limit, hidden_dim=hidden_dim)
        trajectory = trajectory_train
    elif alg == 'bopah_single':
        kl_coef, gradient_norm_panelty, gradient_norm_limit = alg_params['kl_coef'], alg_params['gradient_norm_panelty'], alg_params['gradient_norm_limit']
        alg_name = 'bopah_single_klcoef_{}_grad_norm_panelty_{}_grad_norm_limit_{}'.format(kl_coef, gradient_norm_panelty, gradient_norm_limit)
        model = BOPAHSingle(state_dim, action_dim, max_action, kl_coef=kl_coef, gradient_norm_panelty=gradient_norm_panelty, gradient_norm_limit=gradient_norm_limit, hidden_dim=hidden_dim)
        trajectory = trajectory_train
        batch_trajectory = trajectory_valid
    elif alg == 'bopah':
        kl_coef, gradient_norm_panelty, gradient_norm_limit, dependent_limit, num_clusters \
            = alg_params['kl_coef'], alg_params['gradient_norm_panelty'], alg_params['gradient_norm_limit'], alg_params['dependent_limit'], alg_params['num_clusters']
        alg_name = 'bopah_klcoef_{}_grad_norm_panelty_{}_grad_norm_limit_{}_dependent_limit_{}'.format(kl_coef, gradient_norm_panelty, gradient_norm_limit, dependent_limit)
        if alg_params.get('total_loss'):
            alg_name += '_total_loss'
        cluster_info = generate_cluster(env_name, trained_policy_seed, trained_policy_step, trajectory_episode, num_clusters, trajectory_seed, hidden_dim=hidden_dim)
        model = BOPAH(trajectory_train, trajectory_valid, state_dim, action_dim, max_action, kl_coef=kl_coef, gradient_norm_panelty=gradient_norm_panelty, 
                            gradient_norm_limit=gradient_norm_limit, hidden_dim=hidden_dim, cluster_info=cluster_info, 
                            dependent_limit=dependent_limit, seed=seed, total_loss=alg_params.get('total_loss'))
        trajectory = trajectory_train
        batch_trajectory = trajectory_valid
    elif alg == 'bcq':
        alg_name += '_perturb_{}'.format(alg_params['perturb'])
        model = BCQ(state_dim, action_dim, max_action, trajectory=trajectory_all, hidden_dim=hidden_dim, perturb=alg_params['perturb'])
    elif alg == 'bear':
        alg_name += '_thres_{}'.format(alg_params['thres'])
        model = BEAR(state_dim, action_dim, max_action, hidden_dim=hidden_dim, threshold=alg_params['thres'])
    else:
        raise NotImplementedError()

    # Set result path
    result_dir = "eval_results/%s/seed_%d/step_%d/trajectory_%d/seed_%d_hidden_%d/%s" % (env_name, trained_policy_seed, trained_policy_step, trajectory_episode, trajectory_seed, hidden_dim, alg_name)
    os.makedirs(result_dir, exist_ok=True)
    result_filepath = "%s/seed_%d.npy" % (result_dir, seed)
    if os.path.exists(result_filepath):
        print('Result file already exists: %s' % result_filepath)
        return np.load(result_filepath, allow_pickle=True)[()]

    # Run algorithm and save the result
    print('==============================================')
    print('Run: ', result_filepath)
    vec_env = make_vectorized_env(env_name)  # for policy evaluation
    eval_timesteps, evals, info_values = model.batch_learn(trajectory, vec_env, total_timesteps=total_timesteps, log_interval=log_interval, seed=seed,
                                                           result_filepath=result_filepath, valid_trajectory=batch_trajectory)
    result = {'eval_timesteps': eval_timesteps, 'evals': evals, 'info_values': info_values}
    np.save(result_filepath, result)
    os.remove(result_filepath + '.tmp.npy')

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='halfcheetah')
    parser.add_argument("--trained_policy_seed", help="random seed of the trained policy", default=0, type=int)
    parser.add_argument("--trained_policy_step", help="time step of the trained policy", default=500000, type=int)
    parser.add_argument("--trajectory_episode", help="the number episodes in batch data collected by the trained policy", default=1000, type=int)
    parser.add_argument("--trajectory_seed", help="random seed of trajectory generation", default=0, type=int)
    parser.add_argument("--total_timesteps", help="total timesteps", default=5000000, type=int)
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    args = parser.parse_args()

    gradient_norm_limit = 100 if args.env_name == 'walker' else 30
    alg_params = {'kl_coef': 1.0, 'gradient_norm_panelty': 0.02, 'gradient_norm_limit':gradient_norm_limit, 'hidden_dim':100,
                  'dependent_limit':np.log(2), 'num_clusters':20, 'total_loss':True, 'perturb':0.05, 'thres':0.05}

    run(args.env_name, args.trained_policy_seed, args.trained_policy_step, args.trajectory_episode, args.trajectory_seed, 'klac',
        total_timesteps=args.total_timesteps, seed=0, alg_params=alg_params)
