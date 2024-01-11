import numpy as np
from MarkovApprox import Tauchen
from scipy.stats import norm
from dataclasses import dataclass
from tqdm import tqdm
from RandomProcess import FiniteMarkov
import matplotlib.pyplot as plt


@dataclass
class MenuCostVFIRes:
    a_grid: np.ndarray
    trans_mat_a: np.ndarray
    p_grid: np.ndarray
    trans_mat_p: np.ndarray
    policy_grid: np.ndarray
    value: np.ndarray
    policy: np.ndarray
    offset: int


class MenuCostEnv:
    def __init__(self, **kwargs):
        default_params = {
            'beta': 0.96 ** (1 / 12),
            'theta': 4,
            'mu': 0.0021,
            'sig_eta': 0.0032,
            'sig_ep': 0.0428,
            'k': 0.0245,
            'rho': 0.66
        }
        default_params.update(kwargs)
        self.beta = default_params['beta']
        self.theta = default_params['theta']
        self.mu = default_params['mu']
        self.sig_eta = default_params['sig_eta']
        self.sig_ep = default_params['sig_ep']
        self.rho = default_params['rho']
        # C = 1
        self.k = default_params['k']

    def approx_eta(self, eta_num, eta_precision):
        all_width = (eta_num - 1) / 2 / eta_precision * self.mu
        grid = np.linspace(-all_width, all_width, eta_num)
        w = grid[1] - grid[0]
        trans_prob = np.zeros(eta_num)

        for j in range(eta_num):
            upper = norm.cdf((grid[j] + w / 2) / self.sig_eta)
            lower = norm.cdf((grid[j] - w / 2) / self.sig_eta)
            if j == 0:
                trans_prob[j] = upper
            elif j == eta_num - 1:
                trans_prob[j] = 1 - lower
            else:
                trans_prob[j] = upper - lower
        return grid, trans_prob


    '''
               p grid
               |---| \           \ 
    policy     |---|  --> offset  \ 
     grid      |---| /             \ 
    |---| ---> |---|                 -> possible diffusion
    |---|      |---|               /
    |---|      |---|              /
    |---| ---> |---|             /
               |---|
               |---|
    state: p_t = log (p_{t-1}(z) / P_t), a_t = log(A_t)
    policy: the center of p_{t+1} = log (p_t(z) / P_t) - mu
    '''
    def vfi(self, a_num=5, p_num=201, eta_num=5, eta_precision=1):
        assert (eta_num - 1) % (2 * eta_precision) == 0
        assert (p_num - 1) % (2 * eta_precision) == 0
        a_grid, trans_mat_a = Tauchen(self.rho, self.sig_ep ** 2, 0).approx(n=a_num)
        p_width = (p_num - 1) / 2 / eta_precision * self.mu
        p_grid = np.linspace(-p_width, p_width, p_num)
        _, trans_prob_eta = self.approx_eta(eta_num, eta_precision)
        policy_num = p_num - eta_num + 1
        offset = int(eta_num / 2)
        policy_grid = p_grid[offset: -offset]

        profit = np.exp(-self.theta * (policy_grid + self.mu)) * (np.exp(policy_grid + self.mu)
                 - np.expand_dims((self.theta - 1) / self.theta * np.exp(-a_grid), axis=-1).repeat(policy_num, axis=-1))
        profit -= (self.theta - 1) / self.theta * self.k
        pi = np.tile(profit, (p_num, 1, 1))
        for i in range(policy_num):
            if offset + i + eta_precision >= p_num:
                break
            pi[offset + i + eta_precision, :, i] += (self.theta - 1) / self.theta * self.k

        value = np.zeros((p_num, a_num))
        trans_mat_p = np.zeros((policy_num, p_num))
        for i in range(policy_num):
            trans_mat_p[i, i:i + eta_num] = trans_prob_eta

        iter = 1
        old_v = np.random.normal(0, 1, (p_num, a_num))
        while not np.allclose(value, old_v):
            old_v = np.copy(value)
            value = np.max(pi + self.beta * trans_mat_a @ value.T @ trans_mat_p.T, axis=-1)
            print(f'iter {iter}, loss {np.max(value - old_v)}')
            iter += 1

        policy = np.argmax(pi + trans_mat_a @ value.T @ trans_mat_p.T, axis=-1)
        res = MenuCostVFIRes(a_grid, trans_mat_a, p_grid, trans_mat_p, policy_grid, value, policy, offset)
        self.res = res
        return res

    def plot_policy(self, res=None, figure_path=None, show=True, title=None):
        if res is None:
            res = self.res
        assert res is not None
        for idxa in range(res.value.shape[1]):
            plt.plot(res.p_grid, res.policy_grid[res.policy[:, idxa]] + self.mu, label=f'A={np.exp(res.a_grid[idxa]):.4f}')
        plt.plot(res.p_grid, res.p_grid)
        plt.title('Policy' if title is None else title)
        plt.xlabel('log[ p_{t-1}(z)/P_t ]')
        plt.xlabel('log[ p_t(z)/P_t ]')
        plt.legend()
        if figure_path:
            plt.savefig(figure_path)
        if show:
            plt.show()
        else:
            plt.clf()
    
    def simulate(n=10000, T=120, res=None):
        if res is None:
            res = self.res
        assert res is not None
        a_path, _ = FiniteMarkov(res.a_grid, res.trans_mat_a).simulate(T, 0)
        p_paths = np.ones((n, T), dtype=int) * int(res.value.shape[0] / 2)
        a_paths = np.ones((n, T), dtype=int) * int(res.value.shape[1] / 2)
        for i in tqdm(range(n)):
            a_path, _ = FiniteMarkov(res.a_grid, res.trans_mat_a).simulate(T - 1, 0)
            a_paths[i] =  a_path
            for t in range(1, T):
                p_paths[i, t] = np.random.choice(list(range(p_num)), p=res.trans_mat_p[res.policy[p_paths[i, t - 1], a_path[t - 1]]])
        return res.p_grid[p_paths], res.a_grid[a_paths], p_paths, a_paths
    
    