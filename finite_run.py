import time
from collections import defaultdict

from finite_mdp import *

np.set_printoptions(precision=3, suppress=True, linewidth=250)


def run(seed, optimality, trajectory_num_episodes, N_split):
    print('===================================', flush=True)
    print('seed={} / optimality={} / trajectory_num_episodes={} / N_split={}'.format(seed, optimality, trajectory_num_episodes, N_split))
    result_dir = 'result_random_mdp'
    os.makedirs(result_dir, exist_ok=True)
    result_filepath = "{}/optimality_{}_N_trajectory_{}_split_{}_{}.npy".format(result_dir, optimality, trajectory_num_episodes, N_split, seed % N_split)

    if os.path.exists(result_filepath):
        result = np.load(result_filepath, allow_pickle=True)[()]
    else:
        result = {}
    if result.get(seed) is None:
        result[seed] = {}

    if result[seed].get('completed') == 1:
        print('Completed... so skip!')
        return

    # Generate a random MDP
    start_time = time.time()
    mdp = generate_random_mdp(seed, S=50, A=4, gamma=0.95)
    pi_b = generate_baseline_policy(seed, mdp, optimality)
    print('MDP constructed ({:6.3f} secs)'.format(time.time() - start_time))

    # Construct MLE MDP
    start_time = time.time()
    trajectory_all = generate_trajectory(seed, mdp, pi_b, num_episodes=trajectory_num_episodes)
    mdp_all, N_all = compute_MLE_MDP(mdp.S, mdp.A, mdp.R, mdp.gamma, mdp.temperature, trajectory_all)
    print('MLE MDP constructed ({:6.3f} secs)'.format(time.time() - start_time))

    # In order to normalize performance
    if result[seed].get('V_opt') is None:
        start_time = time.time()
        V_opt = solve_MDP(mdp)[1][0]  # 1: optimal policy
        V_b = policy_evaluation(mdp, pi_b)[0][0]  # 0: baseline policy
        V_unif = policy_evaluation(mdp, np.ones((mdp.S, mdp.A)) / mdp.A)[0][0]
        result[seed]['V_opt'] = V_opt
        result[seed]['V_b'] = V_b
        result[seed]['V_unif'] = V_unif
        print('V_opt=%.3f / V_unif=%.3f / V_b=%.3f (ratio=%f) ({%6.3f} secs)' % (V_opt, V_unif, V_b, (V_b - V_unif) / (V_opt - V_unif), time.time() - start_time))

    # Baseline: solve (train + valid) MLE MDP without regularization
    if result[seed].get('V_mbrl') is None:
        start_time = time.time()
        pi_all, _, _ = solve_MDP(mdp_all)
        V_mbrl = policy_evaluation(mdp, pi_all)[0][0]
        result[seed]['V_mbrl'] = V_mbrl
        print('{:20s}: {:.3f} ({:6.3f} secs)'.format("MBRL", V_mbrl, time.time() - start_time))

    # SPIBB
    for N_threshold in [1, 2, 3, 5, 7, 10, 20]:
        if result[seed].get('V_spibb_{}'.format(N_threshold)) is None:
            start_time = time.time()
            pi_spibb = SPIBB(mdp_all, pi_b, N_all, N_threshold=N_threshold)
            V_spibb = policy_evaluation(mdp, pi_spibb)[0][0]
            result[seed]['V_spibb_{}'.format(N_threshold)] = V_spibb
            print('{:20s}: {:.3f} ({:6.3f} secs)'.format("SPIBB_{}".format(N_threshold), V_spibb, time.time() - start_time))

    # Robust MDP
    for delta in [0.001, 0.01, 0.1, 0.5, 1]:
        if result[seed].get('V_rmdp_{}'.format(delta)) is None:
            start_time = time.time()
            pi_rmdp = robust_MDP(mdp_all, N_all, delta=delta)
            V_rmdp = policy_evaluation(mdp, pi_rmdp)[0][0]
            result[seed]['V_rmdp_{}'.format(delta)] = V_rmdp
            print('{:20s}: {:.3f} ({:6.3f} secs)'.format("RobustMDP_{}".format(delta), V_rmdp, time.time() - start_time))
    
    # Reward-adjusted MDP (RAMDP)
    for kappa in [0.001, 0.003, 0.01, 0.1, 1]:
        if result[seed].get('V_ramdp_{}'.format(kappa)) is None:
            start_time = time.time()
            pi_ramdp = RAMDP(mdp_all, N_all, kappa=kappa)
            V_ramdp = policy_evaluation(mdp, pi_ramdp)[0][0]
            result[seed]['V_ramdp_{}'.format(kappa)] = V_ramdp
            print('{:20s}: {:.3f} ({:6.3f} secs)'.format("RAMDP_{}".format(kappa), V_ramdp, time.time() - start_time))

    # BOPAH
    for fold in [2]:  # [2, 5]:
        for dof in [1, 50]:  # [1, 2, 4, 20, 50]:
            if result[seed].get('V_bopah_{}_{}'.format(fold, dof)) is None:
                start_time = time.time()
                alpha = Alpha(S=mdp.S, D=dof, psi=np.clip(np.ones(dof) * 1.0 / len(trajectory_all), 0.001, np.inf))
                pi_bopah, _ = BOPAH(mdp.S, mdp.A, mdp.R, mdp.gamma, mdp.temperature, trajectory_all, pi_b, alpha, N_folds=fold)
                V_bopah = policy_evaluation(mdp, pi_bopah)[0][0]
                result[seed]['V_bopah_{}_{}'.format(fold, dof)] = V_bopah
                print('{:20s}: {:.3f} ({:6.3f} secs)'.format("BOPAH_{}_{}".format(fold, dof), V_bopah, time.time() - start_time))

    result[seed]['completed'] = 1

    # Save the result...
    np.save(result_filepath + '.tmp.npy', result)
    # rename the resultfile
    os.rename(result_filepath + '.tmp.npy', result_filepath)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", help="condor pid", default=None, type=int)
    args = parser.parse_args()

    print('pid: ', args.pid)
    num_nodes = 100  # the number of condor nodes

    if args.pid is None:
        settings = []
        for seed in range(10000):
            for trajectory_num_episodes in [10, 20, 50, 100, 200, 500, 1000, 2000]:
                for optimality in [0.9, 0.5]:
                    run(seed, optimality, trajectory_num_episodes=trajectory_num_episodes, N_split=num_nodes)
    else:
        print('condor')
        num_repeats = 100
        assert args.pid < num_nodes

        for repeat in range(num_repeats):
            seed = args.pid + repeat * num_nodes
            for trajectory_num_episodes in [10, 20, 50, 100, 200, 500, 1000, 2000]:
                for optimality in [0.9, 0.5]:
                    run(seed, optimality, trajectory_num_episodes=trajectory_num_episodes, N_split=num_nodes)
