import numpy as np
from MarkovApprox import Rowenhorst, Tauchen, TauchenHussey
from FunctionApprox import ChebyshevApprox2D
import matplotlib.pyplot as plt
from RandomProcess import AR1Process
from tqdm import tqdm, trange
import warnings

default_params = {
    'rho': 0.979,
    'sigma': 0.0072,
    'gamma': 2,
    'beta': 0.984,
    'delta': 0.025,
    'alpha': 0.33,
    'tol': 1e-3
}


def compute_moments(a, k, c, i, y, burned=10000):
    sigma_a = np.sqrt(np.var(a[burned:]))
    sigma_k = np.sqrt(np.var(np.log(k[burned:])))
    sigma_ak = np.cov(a[burned:], np.log(k[burned:]))[0][1]
    sigma_y = np.sqrt(np.var(np.log(y[burned:])))
    sigma_c = np.sqrt(np.var(np.log(c[burned:])))
    sigma_i = np.sqrt(np.var(i[burned:]))
    rho_y = np.mean(y[burned:-1] * y[burned + 1:]) / np.mean(y[burned:-1] * y[burned:-1])
    return [sigma_k, sigma_ak, sigma_y, sigma_c, sigma_i, rho_y]


class OptimalGrowthEnv:
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.alpha = params.get('alpha', default_params['alpha'])
        self.beta = params.get('beta', default_params['beta'])
        self.rho = params.get('rho', default_params['rho'])
        self.sigma = params.get('sigma', default_params['sigma'])
        self.delta = params.get('delta', default_params['delta'])
        self.gamma = params.get('gamma', default_params['gamma'])
        self.tol = params.get('tol', default_params['tol'])

        self.k_ss = ((1 / self.beta - 1 + self.delta) / self.alpha) ** (1 / (self.alpha - 1))
        self.c_ss = self.k_ss ** self.alpha - self.delta * self.k_ss

        self.value_func = None
        self.policy_func = None
        self.k_grids = None
        self.a_grids = None
        self.trans_mat = None
        self.policy_func_cheby = None
        self.value_func_cheby = None

    def k_new(self, k_old, a, c):
        return np.exp(a) * k_old ** self.alpha + (1 - self.delta) * k_old - c

    def utility(self, c):
        u = np.zeros_like(c)
        u[c > 0] = c[c > 0] ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c[c > 0])
        u[c <= 0] = -np.inf
        return u

    def resource(self, k, a):
        return np.exp(a) * k ** self.alpha + (1 - self.delta) * k

    def grid_search(self, n_k, k_width, n_a, c_grids_width):
        a_grids, trans_mat = Rowenhorst(self.rho, self.sigma ** 2, 0).approx(n_a)
        self.a_grids, self.trans_mat = a_grids, trans_mat
        a_width = (np.max(a_grids) - np.min(a_grids)) / 2 + 1e-3
        a_center = (np.max(a_grids) + np.min(a_grids)) / 2
        k_prime_grids = np.arange(self.k_ss - k_width, self.k_ss + k_width, c_grids_width)

        value_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid2=a_grids)
        k_grids, a_grids = value_func.real_grids()
        resource = np.exp(a_grids).reshape((-1, 1)) * k_grids.reshape((1, -1)) ** self.alpha + (1 - self.delta) * k_grids
        resource = resource.T.reshape((n_k, n_a, 1))
        u_mat = self.utility(resource - k_prime_grids)

        old_weights = np.random.random(n_k * n_a)
        policy_func = np.zeros((n_k, n_a))
        new_values = np.zeros((n_k, n_a))
        t = trange(10000, desc='Bar desc', leave=True)
        while not np.allclose(value_func.weights, old_weights, atol=self.tol, rtol=self.tol):
            old_weights = np.copy(value_func.weights)
            v_prime = value_func(k_prime_grids, a_grids)
            new_values = u_mat + self.beta * trans_mat @ v_prime.T
            max_idx = np.argmax(new_values, axis=-1)
            new_values = np.take_along_axis(new_values, np.expand_dims(max_idx, axis=-1), axis=-1)
            policy_func = k_prime_grids[max_idx]
            value_func.approx(new_values)

            t.set_description(f'grid search: {np.sum(np.abs(value_func.weights - old_weights))}')
            t.update()
        self.policy_func = policy_func
        self.value_func = new_values
        self.k_grids = value_func.grid1 * k_width + self.k_ss

    def euler_method(self, n_k, k_width, n_a, markov_method='R'):
        if markov_method == 'R':
            a_grids, trans_mat = Rowenhorst(self.rho, self.sigma ** 2, 0).approx(n_a)
        elif markov_method == 'T':
            a_grids, trans_mat = Tauchen(self.rho, self.sigma ** 2, 0).approx(n_a)
        elif markov_method == 'TH':
            a_grids, trans_mat = TauchenHussey(self.rho, self.sigma ** 2, 0).approx(n_a)
        else:
            a_grids, trans_mat = Rowenhorst(self.rho, self.sigma ** 2, 0).approx(n_a)
        self.a_grids, self.trans_mat =  a_grids, trans_mat
        a_width = (np.max(a_grids) - np.min(a_grids)) / 2 + 1e-3
        a_center = (np.max(a_grids) + np.min(a_grids)) / 2

        policy_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid1=None, grid2=a_grids)
        value_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid1=None, grid2=a_grids)
        k_grids, a_grids = value_func.real_grids()

        old_weights = np.random.random(n_k * n_a)
        resource = np.exp(a_grids).reshape((-1, 1)) * k_grids.reshape((1, -1)) ** self.alpha + (1 - self.delta) * k_grids
        policy_func.approx(self.k_ss * np.ones((n_k, n_a)))

        t = trange(10000, desc='Bar desc', leave=True)

        while not np.allclose(policy_func.weights, old_weights, atol=self.tol):
            loss = np.sum(np.abs(policy_func.weights - old_weights))
            if not np.isfinite(loss):
                raise RuntimeError('not converge')
            t.set_description(f'policy loss: {np.sum(np.abs(policy_func.weights - old_weights))}')
            t.update()
            old_weights = np.copy(policy_func.weights)

            k_prime = policy_func(k_grids, a_grids)   # (n_k, n_a)
            k_2prime = (policy_func(k_prime, a_grids)) # (n_k, n_a, n_a')
            resource_ = (k_prime.reshape((n_k, n_a, 1)) ** self.alpha @ np.exp(a_grids).reshape((1, -1))).transpose(2, 0, 1) + (1 - self.delta) * k_prime
            resource_ = resource_.transpose(1, 2, 0)
            c_prime = resource_ - k_2prime
            mu_prime = c_prime ** (-self.gamma) * (self.alpha * np.expand_dims(k_prime, axis=-1) ** (self.alpha - 1) * np.exp(a_grids) + 1 - self.delta)
            mu = (mu_prime * trans_mat).sum(axis=-1) * self.beta
            new_policy = resource.T - mu ** (-1 / self.gamma)
            policy_func.approx(new_policy, lr=0.2)


        self.policy_func = policy_func(k_grids, a_grids)
        self.policy_func_cheby = policy_func
        self.k_grids = k_grids

        c = resource.T - self.policy_func
        u = self.utility(c)
        k_prime = self.policy_func

        old_weights = np.random.random((n_k * n_a))
        new_values = np.zeros((n_k, n_a))

        t = trange(10000, desc='Bar desc', leave=True)

        while not np.allclose(value_func.weights, old_weights, atol=self.tol, rtol=self.tol):
            t.set_description(f'value loss: {np.sum(np.abs(value_func.weights - old_weights))}')
            t.update()
            old_weights = np.copy(value_func.weights)
            v_prime = value_func(k_prime, a_grids)
            new_values = u + self.beta * (v_prime * trans_mat).sum(axis=-1)
            value_func.approx(new_values, lr=1.)
        self.value_func = new_values
        self.value_func_cheby = value_func
        #print(self.value_func_cheby.eval(self.k_ss, 0))

    def print_steady_state(self):
        print('Steady State:')
        print('K:', self.k_ss)
        print('C:', self.c_ss)

    def plot_policy(self, grid_num=100, offgrid=True):
        if self.policy_func is None:
            return None
        if self.policy_func_cheby is None:
            offgrid = False
        k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), grid_num, endpoint=True) if offgrid \
                else self.k_grids
        k_prime = self.policy_func_cheby(k_grids, self.a_grids) if self.policy_func_cheby else self.policy_func
        for idxa, a in enumerate(self.a_grids):
            plt.plot(k_grids, k_prime[:, idxa], label=str(round(a, 2)))
            plt.scatter(self.k_grids, self.policy_func[:, idxa], color='red')
        plt.plot(self.k_grids, self.k_grids, label='45 degree')
        plt.scatter([self.k_ss], [self.k_ss])
        plt.legend()
        plt.title('Capital Accumulation')
        plt.show()

    def plot_value(self, grid_num=100, offgrid=True):
        if self.value_func is None:
            return None
        if self.value_func_cheby is None:
            offgrid = False
        k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), grid_num, endpoint=True) if offgrid \
                else self.k_grids
        value = self.value_func_cheby(k_grids, self.a_grids) if self.value_func_cheby else self.value_func
        for idxa, a in enumerate(self.a_grids):
            plt.plot(k_grids, value[:, idxa], label=str(round(a, 2)))
            plt.scatter(self.k_grids, self.value_func[:, idxa], color='red')
        plt.legend()
        plt.title('Value Function')
        plt.show()

    def plot_consumption(self, grid_num=100, offgrid=True):
        if self.policy_func is None:
            return None
        if self.policy_func_cheby is None:
            offgrid = False
        k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), grid_num, endpoint=True) if offgrid \
                else self.k_grids
        k_prime = self.policy_func_cheby(k_grids, self.a_grids) if self.policy_func_cheby else self.policy_func
        resource = np.expand_dims(np.exp(self.a_grids), axis=-1) * k_grids ** self.alpha + (1 - self.delta) * k_grids
        c = resource.T - k_prime
        resource = np.expand_dims(np.exp(self.a_grids), axis=-1) * self.k_grids ** self.alpha + (1 - self.delta) * self.k_grids
        c_ = resource.T - self.policy_func
        for idxa, a in enumerate(self.a_grids):
            plt.plot(k_grids, c[:, idxa], label=str(round(a, 2)))
            plt.scatter(self.k_grids, c_[:, idxa], color='red')
        plt.scatter([self.k_ss], [self.c_ss])
        plt.legend()
        plt.title('Consumption')
        plt.show()

    def steady_distribution(self, k_grids):
        warnings.warn('not finished')
        if k_grids[0] > np.min(self.k_grids):
            k_grids = np.concatenate(([self.k_grids[0]], k_grids))
        k_num = k_grids.size
        k_max = np.concatenate((k_grids[1:], [self.k_grids[-1]]))
        k_values = (k_grids + k_max) / 2
        a_num = self.a_grids.size
        policy_func = np.zeros((k_num, a_num), dtype=int)
        for idxk, k in enumerate(k_values):
            for idxa, a in enumerate(self.a_grids):
                c = self.policy_func_cheby(k, a)
                k_prime = self.k_new(k, a, c)
                policy_func[idxk][idxa] = np.searchsorted(k_grids, k_prime)
        distribution = np.zeros((k_num, a_num))
        distribution[0][0] = 1
        old_distribution = np.zeros((k_num, a_num))
        while not np.allclose(old_distribution, distribution):
            old_distribution = np.copy(distribution)
            distribution = np.zeros((k_num, a_num))
            for idxk in range(k_num):
                for idxa in range(a_num):
                    for idxap in range(a_num):
                        distribution[policy_func[idxk][idxa]][idxap] += old_distribution[idxk][idxa] * self.trans_mat[idxa][idxap]
        return distribution, k_values

    # incomplete
    def compute_moments(self, grid_num=100):
        warnings.warn('not finished')
        k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), grid_num)
        dist, k_values = self.steady_distribution(k_grids)

        dist_a = np.sum(dist, axis=0)
        a_aprime = np.kron(self.a_grids.reshape((-1, 1)), self.a_grids.reshape((1, -1)))
        rho_hat = np.sum(a_aprime * (self.trans_mat * dist_a)) / np.sum(self.a_grids * self.a_grids * dist_a)

        epsilon = -rho_hat * self.a_grids.reshape((-1, 1)) + self.a_grids.reshape((1, -1))
        sigma_e_hat = np.sum(epsilon * epsilon * self.trans_mat)

        mean_a_hat = np.sum(self.a_grids * dist_a)
        sigma_a_hat = np.sum(self.a_grids * self.a_grids * dist_a) - mean_a_hat ** 2

        dist_k = np.sum(dist, axis=1)
        mean_k_hat = np.sum(k_values * dist_k)
        sigma_k_hat = np.sum(k_values * k_values * dist_k) - mean_k_hat ** 2

        sigma_ka_hat = np.sum((k_values.reshape((-1, 1)) * self.a_grids.reshape((1, -1))) * dist) - mean_a_hat * mean_k_hat

        c = np.zeros((k_values.size, self.a_grids.size))
        y = np.zeros((k_values.size, self.a_grids.size))
        i = np.zeros((k_values.size, self.a_grids.size))

        for idxk in range(k_values.size):
            for idxa in range(self.a_grids.size):
                pass

    def simulation(self, periods=5010000, a_series=None):
        if a_series is None:
            a_series = AR1Process(rho=self.rho, err_params={'sigma': self.sigma}).simulate(periods, init=0)
            a_series = np.clip(a_series, np.min(self.a_grids), np.max(self.a_grids))[1:]

        k_max = np.max(self.k_grids)
        k_min = np.min(self.k_grids)

        k_series = np.zeros(periods + 1)
        c_series = np.zeros(periods)
        i_series = np.zeros(periods)
        y_series = np.zeros(periods)
        k_series[0] = self.k_ss

        for t in trange(periods):
            k_series[t + 1] = self.policy_func_cheby(k_series[t], a_series[t])
            k_series[t + 1] = np.clip(k_series[t + 1], k_min, k_max)
            y_series[t] = np.exp(a_series[t]) * k_series[t] ** self.alpha
            i_series[t] = k_series[t + 1] - (1 - self.delta) * k_series[t]
            c_series[t] = y_series[t] - i_series[t]

        return a_series, k_series[:-1], c_series, i_series, y_series
