import numbers
import os
from copy import copy

import numpy as np
from scipy.special import logsumexp


class MDP:

    def __init__(self, S=50, A=4, T=None, R=None, gamma=0.95, temperature=0):
        """
        Create a random MDP

        :param S: the number of states
        :param A: the number of actions
        :param gamma: discount factor
        :param temperature: state-dependent temperature parameter for soft MDP
        """
        self.gamma = gamma
        if isinstance(temperature, numbers.Number):
            self.temperature = np.ones(S) * temperature
        else:
            self.temperature = temperature
        assert (len(self.temperature) == S)
        self.S = S
        self.A = A
        self.initial_state = 0
        self.absorbing_state = S - 1

        if T is None:
            self.T = np.zeros((self.S, self.A, self.S))
            for s in range(S):
                if s == self.absorbing_state:
                    self.T[s, :, s] = 1  # absorbing state: self-transition
                else:
                    for a in range(A):
                        p = np.r_[np.random.dirichlet([1, 1, 1, 1]), [0] * (S - 4 - 1)]
                        np.random.shuffle(p)
                        self.T[s, a, :] = np.r_[p, [0]]
        else:
            self.T = np.array(T)

        if R is None:
            min_value_state, min_value = -1, 1e10
            for s in range(S - 1):
                self.R = np.zeros((self.S, self.A))
                self.R[s, :] = 1
                T_tmp = np.array(self.T[s, :, :])
                self.T[s, :, :] = 0
                self.T[s, :, self.absorbing_state] = 1  # goal_state -> absorbing state
                _, V, _ = solve_MDP(self)
                if V[0] < min_value:
                    min_value = V[0]
                    min_value_state = s
                self.T[s, :, :] = T_tmp

            # Now, we determine the goal state: min_value_state
            self.goal_state = min_value_state
            self.R = np.zeros((self.S, self.A))
            self.R[self.goal_state, :] = 1
            self.T[self.goal_state, :, :] = 0
            self.T[self.goal_state, :, self.absorbing_state] = 1  # goal_state -> absorbing state
        else:
            self.R = np.array(R)

    def __copy__(self):
        mdp = MDP(S=self.S, A=self.A, T=self.T, R=self.R, gamma=self.gamma, temperature=self.temperature)
        return mdp


class Alpha:

    def __init__(self, S, D, psi=None, alpha_min=0.001):
        self.S = S
        self.D = D
        self.alpha_min = alpha_min
        self.psi = np.ones(self.D) * 0.1 if psi is None else np.clip(psi, self.alpha_min, np.inf)

        self.psi_grad = np.zeros((D, S))
        for d in range(D):
            self.psi_grad[d, np.arange(d, S, D)] = 1

    def forward(self):
        """
        :return: alpha(s) for each s
        """
        result = np.zeros(self.S)
        for d in range(self.D):
            result[np.arange(d, self.S, self.D)] = self.psi[d]
        return result

    def backward(self):
        """
        :return: \nabla_\psi \alpha(s) for each s
        """
        return np.array(self.psi_grad)

    def update(self, psi):
        self.psi = np.clip(psi, self.alpha_min, np.inf)

    def __copy__(self):
        new_alpha = Alpha(self.S, self.D)
        new_alpha.psi = np.array(self.psi)
        new_alpha.psi_grad = np.array(self.psi_grad)
        return new_alpha

    def __str__(self):
        return "[D=%d] %s" % (self.D, str(self.psi))

    __repr__ = __str__


def policy_evaluation(mdp, pi):
    if np.all(mdp.temperature == 0):
        # hard MDP
        r = np.sum(mdp.R * pi, axis=-1)
        P = np.sum(pi[:, :, None] * mdp.T, axis=1)

        if len(mdp.R.shape) == 3:
            V = np.tensordot(np.linalg.inv(np.eye(mdp.S) - mdp.gamma * P), r, axes=[-1, -1]).T
            Q = mdp.R + mdp.gamma * np.tensordot(mdp.T, V, axes=[-1, -1]).transpose([2, 0, 1])
        else:
            V = np.linalg.inv(np.eye(mdp.S) - mdp.gamma * P).dot(r)
            Q = mdp.R + mdp.gamma * mdp.T.dot(V)
        return V, Q
    else:
        # soft MDP
        r = np.sum((mdp.R - mdp.temperature[:, None] * np.log(pi + 1e-300)) * pi, axis=1)  # TODO
        P = np.sum(pi[:, :, None] * mdp.T, axis=1)

        V = np.linalg.inv(np.eye(mdp.S) - mdp.gamma * P).dot(r)
        Q = mdp.R + mdp.gamma * mdp.T.dot(V)

        return V, Q


def solve_MDP(mdp, method='PI'):
    if np.all(mdp.temperature == 0):
        if method == 'PI':
            pi = np.ones((mdp.S, mdp.A)) / mdp.A
            V_old = np.zeros(mdp.S)

            for _ in range(1000000):
                V, Q = policy_evaluation(mdp, pi)
                pi_new = np.zeros((mdp.S, mdp.A))
                pi_new[np.arange(mdp.S), np.argmax(Q, axis=1)] = 1.

                if np.all(pi == pi_new) or np.max(np.abs(V - V_old)) < 1e-8:
                    break
                V_old = V
                pi = pi_new

            return pi, V, Q
        elif method == 'VI':
            # perform value iteration
            V, Q = np.zeros(mdp.S), np.zeros((mdp.S, mdp.A))
            for _ in range(100000):
                Q_new = mdp.R + mdp.gamma * mdp.T.dot(V)
                V_new = np.max(Q_new, axis=1)

                if np.max(np.abs(V - V_new)) < 1e-8:
                    break

                V, Q = V_new, Q_new

            pi = np.zeros((mdp.S, mdp.A))
            pi[np.arange(mdp.S), np.argmax(Q, axis=1)] = 1.

            return pi, V, Q
        else:
            raise NotImplementedError('Undefined method: %s' % method)
    else:
        # soft MDP
        if method == 'PI':
            pi = np.ones((mdp.S, mdp.A)) / mdp.A

            for _ in range(1000000):
                V, Q = policy_evaluation(mdp, pi)
                pi_new = softmax(Q, mdp.temperature)

                if np.max(np.abs(pi - pi_new)) < 1e-8:
                    break
                pi = pi_new
            return pi, V, Q
        elif method == 'VI':
            V, Q = np.zeros(mdp.S), np.zeros((mdp.S, mdp.A))
            for _ in range(1000000):
                Q_new = mdp.R + mdp.gamma * mdp.T.dot(V)
                V_new = mdp.temperature * logsumexp(Q_new / mdp.temperature[:, None], axis=1)

                if np.max(np.abs(V - V_new)) < 1e-8:
                    break

                V, Q = V_new, Q_new

            pi = softmax(Q, mdp.temperature)

            return pi, V, Q


def generate_random_mdp(seed, S=50, A=4, gamma=0.95):
    np.random.seed(seed + 1)
    mdp = MDP(S, A, gamma=gamma)
    return mdp


def generate_trajectory(seed, mdp, pi, num_episodes=10, max_timesteps=50):
    if seed is not None:
        np.random.seed(seed + 1)
    trajectory = []
    for i in range(num_episodes):
        trajectory_one = []
        state = mdp.initial_state
        for t in range(max_timesteps):
            action = np.random.choice(np.arange(mdp.A), p=pi[state, :])
            reward = mdp.R[state, action]
            state1 = np.random.choice(np.arange(mdp.S), p=mdp.T[state, action, :])

            trajectory_one.append((i, t, state, action, reward, state1))
            if state == mdp.absorbing_state:
                break
            state = state1
        trajectory.append(trajectory_one)

    return trajectory


def compute_MLE_MDP(S, A, R, gamma, temperature, trajectory, absorb_unseen=True):
    N = np.zeros((S, A, S))
    for trajectory_one in trajectory:
        for episode, t, state, action, reward, state1 in trajectory_one:
            N[state, action, state1] += 1

    T = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            if N[s, a, :].sum() == 0:
                if absorb_unseen:
                    T[s, a, S - 1] = 1  # absorbing state
                else:
                    T[s, a, :] = 1. / S
            else:
                T[s, a, :] = N[s, a, :] / N[s, a, :].sum()

    mle_mdp = MDP(S, A, T, R, gamma, temperature)

    return mle_mdp, N


def compute_MLE_policy(S, A, trajectory):
    N = np.zeros((S, A)) + 1e-3
    for trajectory_one in trajectory:
        for episode, t, state, action, reward, state1 in trajectory_one:
            N[state, action] += 1
    pi = np.array(N)
    for s in range(S):
        pi[s, :] = pi[s, :] / pi[s, :].sum()
    return pi


def softmax(X, temperature):
    X = np.array(X)
    if len(X.shape) == 2:
        X = (X - np.max(X, axis=1)[:, None]) / (temperature[:, None] + 1e-20)  # TODO
        S = np.exp(X) / (np.sum(np.exp(X), axis=1) + 1e-20)[:, None]
        return S
    elif len(X.shape) == 1:
        X = (X - np.max(X)) / temperature
        S = np.exp(X) / np.sum(np.exp(X))
        return S
    else:
        raise NotImplementedError()


def generate_baseline_policy(seed, mdp, optimality=0.9):
    np.random.seed(seed + 1)
    pi_opt, _, Q_opt = solve_MDP(mdp)
    pi_unif = np.ones((mdp.S, mdp.A)) / mdp.A
    V_opt = policy_evaluation(mdp, pi_opt)[0][0]
    V_unif = policy_evaluation(mdp, pi_unif)[0][0]

    ##################################
    # following SPIBB paper
    ##################################
    V_final_target = V_opt * optimality + (1 - optimality) * V_unif
    V_softmax_target = 0.5 * V_opt + 0.5 * V_final_target
    softmax_reduction_factor = 0.9
    perturbation_reduction_factor = 0.9

    temperature = np.ones(mdp.S) / (2 * 10 ** 6)
    pi_soft = softmax(Q_opt, temperature)
    while policy_evaluation(mdp, pi_soft)[0][0] > V_softmax_target:
        temperature /= softmax_reduction_factor
        pi_soft = softmax(Q_opt, temperature)

    pi_b = pi_soft.copy()
    while policy_evaluation(mdp, pi_b)[0][0] > V_final_target:
        s = np.random.randint(mdp.S)
        a_opt = np.argmax(Q_opt[s, :])
        pi_b[s, a_opt] *= perturbation_reduction_factor
        pi_b[s, :] /= np.sum(pi_b[s, :])
    return pi_b


def compute_gradient_alpha(mdp_train, mdp_valid, pi_b, alpha, method='analytic'):
    S, A, R, gamma, temperature = mdp_train.S, mdp_train.A, mdp_train.R, mdp_train.gamma, mdp_train.temperature
    if method == 'analytic':
        new_mdp0 = MDP(S=S, A=A, T=mdp_train.T, R=R+alpha.forward()[:, None] * np.log(pi_b), gamma=gamma, temperature=alpha.forward()+temperature)
        pi_reg0, V_reg0, Q_reg0 = solve_MDP(new_mdp0)
        V0, Q0 = policy_evaluation(mdp_valid, pi_reg0)

        alpha_f = alpha.forward()
        alpha_b = alpha.backward()

        """ Compute dQ """
        """
        R_dQ = np.zeros((alpha.D, S, A))
        for d in range(alpha.D):
            for s in range(S):
                for a in range(A):
                    R_dQ[d, s, a] += alpha_b[d, s] * np.log(pi_b[s, a])
                    for s1 in range(S):
                        R_dQ[d, s, a] += gamma * mdp_train.T[s, a, s1] * alpha_b[d, s1] * V_reg0[s1] / (alpha.forward()[s1] + mdp_train.temperature[s1])
                        for a1 in range(A):
                            R_dQ[d, s, a] -= gamma / (alpha.forward()[s1] + mdp_train.temperature[s1]) * mdp_train.T[s, a, s1] * alpha_b[d, s1] * pi_reg0[s1, a1] * Q_reg0[s1, a1]
        """
        R_dQ = alpha_b[:, :, None] * np.log(pi_b) + gamma * np.tensordot(alpha_b * ((V_reg0 - np.sum(pi_reg0 * Q_reg0, axis=-1)) / (alpha_f + mdp_train.temperature)), mdp_train.T, axes=[-1, -1])
        _, dQ = policy_evaluation(MDP(S=S, A=A, T=mdp_train.T, R=R_dQ, gamma=gamma, temperature=0), pi_reg0)
        # dQ: D x S x A

        """ Compute dPi """
        beta = (dQ * (alpha_f + temperature)[None, :, None] - alpha_b[:, :, None] * Q_reg0[None, :, :]) / ((alpha_f + temperature) ** 2 + 1e-10)[None, :, None]  # TODO
        dPi = pi_reg0[None, :, :] * (beta - np.sum(beta * pi_reg0, axis=-1)[:, :, None])

        """ Compute dV """
        """
        R_dV = np.zeros((alpha.D, S, A))
        for d in range(alpha.D):
            for s in range(S):
                for a in range(A):
                    R_dV[d, s, a] = dPi[d, s, a] / pi_reg0[s, a] * (-temperature[s] * np.log(pi_reg0[s, a]) + Q0[s, a] - temperature[s])
        """
        R_dV = (dPi / (pi_reg0 + 1e-10)) * (-temperature[:, None] * np.log(pi_reg0 + 1e-10) + Q0 - temperature[:, None])  # TODO
        dV, _ = policy_evaluation(MDP(S=S, A=A, T=mdp_valid.T, R=R_dV, gamma=gamma, temperature=0), pi_reg0)

        return dV[:, 0]
    elif method in ['finite_difference', 'fd']:
        epsilon = 1e-7

        # Compute numerical gradient
        new_mdp0 = MDP(S=S, A=A, T=mdp_train.T, R=R + alpha[:, None] * np.log(pi_b), gamma=gamma, temperature=alpha + temperature)
        pi_reg0, V_reg0, Q_reg0 = solve_MDP(new_mdp0)
        V0 = policy_evaluation(mdp_valid, pi_reg0)[0][0]

        numerical_grad = np.zeros(S)
        for s in range(S):
            e = np.zeros(S); e[s] = epsilon
            alpha_e = alpha + e

            new_mdp1 = MDP(S=S, A=A, T=mdp_train.T, R=R + alpha_e[:, None] * np.log(pi_b), gamma=gamma, temperature=alpha_e + temperature)
            pi_reg1, _, _ = solve_MDP(new_mdp1)
            V1 = policy_evaluation(mdp_valid, pi_reg1)[0][0]
            numerical_grad[s] = (V1 - V0) / epsilon

        return numerical_grad
    else:
        raise NotImplementedError()


def BOPAH(S, A, R, gamma, temperature, trajectory_all, pi_b, alpha, N_folds=2, verbose=0):
    mdp_trains = []
    mdp_valids = []
    for fold_i in range(N_folds):
        trajectory_train = []
        trajectory_valid = []
        for fold_j in range(N_folds):
            if fold_i == fold_j:
                trajectory_train += trajectory_all[int(len(trajectory_all) * fold_j/N_folds):int(len(trajectory_all) * (fold_j + 1) / N_folds)]
            else:
                trajectory_valid += trajectory_all[int(len(trajectory_all) * fold_j/N_folds):int(len(trajectory_all) * (fold_j + 1) / N_folds)]
        mdp_trains.append(compute_MLE_MDP(S, A, R, gamma, temperature, trajectory_train)[0])
        mdp_valids.append(compute_MLE_MDP(S, A, R, gamma, temperature, trajectory_valid)[0])

    for grad_iter in range(3000):
        computed_alpha = alpha.forward()
        alpha_grads = []
        for fold_i in range(N_folds):
            mdp_train = mdp_trains[fold_i]
            mdp_valid = mdp_valids[fold_i]
            alpha_grad = np.clip(compute_gradient_alpha(mdp_train, mdp_valid, pi_b, alpha, method='analytic'), -10, 10)
            alpha_grads.append(alpha_grad)
        mean_alpha_grad = np.mean(alpha_grads, axis=0)
        if grad_iter % 200 == 0 and verbose:
            print('[bopah-%d] grad_iter: %5d' % (alpha.D, grad_iter))
            print('- alpha={}'.format(alpha))
            print('- alpha_grad={}'.format(mean_alpha_grad))
        if np.max(np.abs(mean_alpha_grad)) < 1e-6 or (np.all(alpha.forward() == alpha.alpha_min) and np.all(mean_alpha_grad < 0)):
            break
        alpha.update(np.clip(alpha.psi + 0.01 * mean_alpha_grad, alpha.alpha_min, np.inf))

    computed_alpha = alpha.forward()
    computed_alpha[computed_alpha == alpha.alpha_min] = 0
    pi_regs = []
    for fold_i in range(N_folds):
        mdp_train = mdp_trains[fold_i]
        pi_reg_i, _, _ = solve_MDP(MDP(S=S, A=A, T=mdp_train.T, R=R + computed_alpha[:, None] * np.log(pi_b), gamma=gamma, temperature=temperature + computed_alpha))
        pi_regs.append(pi_reg_i)
    pi_reg = np.mean(pi_regs, axis=0)
    if verbose:
        print('[bopah-%d] alpha=%s' % (alpha.D, computed_alpha))
    return pi_reg, computed_alpha


def SPIBB(mdp, pi_b, N, N_threshold=5):
    """
    :param mdp: MDP
    :param pi_b: baseline policy
    :param N: S x A x S table
    :param N_threshold: threshold (integer)
    """
    N = np.sum(N, axis=2)  # S x A
    pi = np.array(pi_b)
    V_old = np.zeros(mdp.S)
    while True:
        V, Q = policy_evaluation(mdp, pi)
        pi_new = np.array(pi_b)
        pi_new[N >= N_threshold] = 0
        for s in range(mdp.S):
            if np.any(N[s, :] >= N_threshold):
                remaining_prob = 1 - np.sum(pi_new[s, :])
                if np.all(mdp.temperature == 0):
                    p = np.zeros(pi_new[s, N[s, :] >= N_threshold].shape)
                    p[np.argmax(Q[s, N[s, :] >= N_threshold])] = remaining_prob
                    pi_new[s, N[s, :] >= N_threshold] = p
                else:
                    pi_new[s, N[s, :] >= N_threshold] = softmax(Q[s, N[s, :] >= N_threshold], mdp.temperature[s]) * remaining_prob

        if np.max(np.abs(pi - pi_new)) < 1e-8 or np.max(np.abs(V_old - V)) < 1e-8:
            break
        pi = pi_new
        V_old = V
    return pi


def robust_MDP(mdp, N, delta=None, c=None):
    assert (delta is None and c is not None) or (delta is not None and c is None)

    def solve_inner_problem(v, p_mean, e):
        # solve min_{p \in P_{plausible}} p * v
        p = np.array(p_mean)
        sorted_indices = [x for _, x in sorted(zip(v, np.arange(len(v))))]
        left, right = 0, len(p) - 1
        p[sorted_indices[right]] -= e / 2
        p[sorted_indices[left]] += e / 2

        while True:
            if p[sorted_indices[left]] > 1:
                p[sorted_indices[left + 1]] += p[sorted_indices[left]] - 1
                p[sorted_indices[left]] = 1
                left += 1
            elif p[sorted_indices[right]] < 0:
                p[sorted_indices[right - 1]] += p[sorted_indices[right]]
                p[sorted_indices[right]] = 0
                right -= 1
            else:
                break
        if not np.isclose(np.sum(p), 1):
            print(e)
            print(p)
        assert np.isclose(np.sum(p), 1)
        assert np.all(p >= 0)
        p /= np.sum(p)
        return p

    V, Q = np.zeros(mdp.S), np.zeros((mdp.S, mdp.A))

    for _ in range(1000000):
        # Compute robust transition
        T_tilde = np.zeros((mdp.S, mdp.A, mdp.S))
        for s in range(mdp.S):
            for a in range(mdp.A):
                if N[s, a].sum() == 0:
                    epsilon = 100
                else:
                    if delta is not None:
                        epsilon = np.sqrt(2. / N[s, a].sum() * (np.log(mdp.S * mdp.A / delta) + mdp.S * np.log(2)))
                    if c is not None:
                        epsilon = c / np.sqrt(N[s, a].sum())
                T_tilde[s, a, :] = solve_inner_problem(V, mdp.T[s, a, :], epsilon)

        # Robust value backup
        Q_new = mdp.R + mdp.gamma * T_tilde.dot(V)
        V_new = np.max(Q_new, axis=1)

        if np.max(np.abs(V - V_new)) < 1e-8:
            break

        V, Q = V_new, Q_new

    pi = np.zeros((mdp.S, mdp.A))
    pi[np.arange(mdp.S), np.argmax(Q, axis=1)] = 1.

    return pi


def RAMDP(mdp, N, kappa=0.003):
    N_sa = np.sum(N, axis=-1) + 0.00001
    ramdp = copy(mdp)
    ramdp.R = mdp.R - kappa / np.sqrt(N_sa)
    pi_ramdp, _, _ = solve_MDP(ramdp)
    return pi_ramdp


def kl_divergence_categorical(pi1, pi2):
    return np.sum(pi1 * (np.log(pi1) - np.log(pi2)), axis=1)


if __name__ == "__main__":
    print('mdp.py')
