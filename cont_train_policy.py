import glob
import os

import gym
from stable_baselines.common import set_global_seeds

from algorithms.sac import SAC
from util import precise_env_name


def train_and_save_policy(env_name, seed, save_period, total_timesteps, hidden_dim=100):
    save_dir = "trained_policy/%s/seed_%d_hidden_%d" % (env_name, seed, hidden_dim)
    os.makedirs(save_dir, exist_ok=True)
    if len(glob.glob("%s/step_*.pkl" % save_dir)) > 0:
        print("already trained: %s" % save_dir)
        return

    def callback(_locals, _globals):
        global n_steps
        model_filepath = "%s/step_%d.pkl" % (save_dir, n_steps + 1)

        if (n_steps + 1) % save_period == 0:
            print('Saving a model to %s' % model_filepath)
            model.save(model_filepath)

        n_steps += 1
        return True

    global n_steps
    n_steps = 0

    env = gym.make(precise_env_name(env_name))
    env.seed(seed)
    set_global_seeds(seed)
    model = SAC(env, ent_coef='auto', seed=seed, hidden_dim=hidden_dim)

    model.learn(total_timesteps=total_timesteps, log_interval=10, seed=seed, callback=callback)


def load_trained_agent(env_name, trained_policy_seed, trained_policy_step, bias_offset=0, seed=0, hidden_dim=64):
    env = gym.make(precise_env_name(env_name))
    trained_agent = SAC.load("trained_policy/%s/seed_%d_hidden_%d/step_%d.pkl" % (env_name, trained_policy_seed, hidden_dim, trained_policy_step), env, seed=seed, hidden_dim=hidden_dim)
    parameters = trained_agent.get_parameters()
    for i, parameter in enumerate(parameters):
        name, value = parameter
        if name == 'actor/f2_log_std/bias:0':
            parameters[i] = (name, value + bias_offset)
    trained_agent.load_parameters(parameters)
    return trained_agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", help="name of the env to train", default='halfcheetah')
    parser.add_argument("--seed", help="random seed", default=0, type=int)
    parser.add_argument("--save_period", help="save period", default=10000, type=int)
    parser.add_argument("--total_timesteps", help="total timesteps", default=500000, type=int)
    args = parser.parse_args()

    train_and_save_policy(args.env_name, args.seed, args.save_period, args.total_timesteps, hidden_dim=100)
