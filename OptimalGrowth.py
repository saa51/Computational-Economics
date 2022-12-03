import numpy as np
from MarkovApprox import Rowenhorst, Tauchen, TauchenHussey
from FunctionApprox import ChebyshevApprox2D
import matplotlib.pyplot as plt
from RandomProcess import AR1Process
from tqdm import tqdm
import tempfile, os
from joblib import Parallel, delayed
from itertools import product

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
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def resource(self, k, a):
        return np.exp(a) * k ** self.alpha + (1 - self.delta) * k

    def grid_search(self, n_k, k_width, n_a, c_grids_width, n_threads=1):
        a_grids, trans_mat = Rowenhorst(self.rho, self.sigma ** 2, 0).approx(n_a)
        self.a_grids, self.trans_mat = a_grids, trans_mat
        a_width = (np.max(a_grids) - np.min(a_grids)) / 2
        a_center = (np.max(a_grids) + np.min(a_grids)) / 2

        value_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid2=a_grids)

        old_weights = np.random.random((1, n_k * n_a))
        if n_threads == 1:
            policy_func = np.zeros((n_k, n_a))
            new_values = np.zeros((n_k, n_a))
        else:
            path = tempfile.mkdtemp()
            value_path = os.path.join(path, 'value.temp')
            new_values = np.memmap(value_path, dtype=float, shape=(n_k, n_a), mode='w+')
            policy_path = os.path.join(path, 'policy.temp')
            policy_func = np.memmap(policy_path, dtype=float, shape=(n_k, n_a), mode='w+')
        k_grids, a_grids = value_func.real_grids()

        def update_value(i, j):
            k = k_grids[i]
            a = a_grids[j]
            y = self.resource(k, a)
            k_primes = np.arange(self.k_ss - k_width, min(y, self.k_ss + k_width), c_grids_width)
            v_primes = np.zeros(k_primes.size)
            for idxk, k_prime in enumerate(k_primes):
                for idxa, a_prime in enumerate(a_grids):
                    v_primes[idxk] += value_func.eval(k_prime, a_prime) * trans_mat[j][idxa]
            v = self.utility(y - k_primes) + self.beta * v_primes
            new_values[i][j] = np.max(v)
            policy_func[i][j] = k_primes[np.argmax(v)]

        while not np.allclose(value_func.weights, old_weights, atol=self.tol, rtol=self.tol):
            print(np.sum(np.abs(value_func.weights - old_weights)))
            old_weights = np.copy(value_func.weights)
            if n_threads == 1:
                for i, j in product(range(len(k_grids)), range(len(a_grids))):
                    update_value(i, j)
            else:
                Parallel(n_jobs=n_threads)(delayed(update_value)(i, j) for i, j in product(range(len(k_grids)), range(len(a_grids))))
            value_func.approx(new_values)
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
        a_width = (np.max(a_grids) - np.min(a_grids)) / 2
        a_center = (np.max(a_grids) + np.min(a_grids)) / 2

        policy_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid2=a_grids)
        value_func = ChebyshevApprox2D(n_k, n_a, self.k_ss, a_center, k_width, a_width, grid2=a_grids)
        k_grids, a_grids = value_func.real_grids()

        old_weights = np.random.random((1, n_k * n_a))
        new_policy = np.zeros((n_k, n_a))
        policy_func.approx(self.c_ss * np.ones((n_k, n_a)))

        while not np.allclose(policy_func.weights, old_weights, atol=self.tol):
            new_policy = np.zeros((n_k, n_a))
            # print(np.sum(np.abs(policy_func.weights - old_weights)))
            old_weights = np.copy(policy_func.weights)

            for i, k in enumerate(k_grids):
                for j, a in enumerate(a_grids):
                    c = np.exp(policy_func.eval(k, a))
                    k_prime = self.k_new(k, a, c)

                    if k_prime < self.k_ss - k_width:
                        new_policy[i][j] = np.log(self.resource(k, a) - self.k_ss + k_width - 1e-6)
                        continue
                    if k_prime > self.k_ss + k_width:
                        new_policy[i][j] = np.log(self.resource(k, a) - self.k_ss - k_width + 1e-6)
                        continue

                    mu_prime = 0
                    for idxa, a_prime in enumerate(a_grids):
                        c_prime = np.exp(policy_func.eval(k_prime, a_prime))
                        k_return = np.exp(a_prime) * self.alpha * k_prime ** (self.alpha - 1) + 1 - self.delta
                        mu_prime += self.beta * c_prime ** (-self.gamma) * k_return * trans_mat[j][idxa]
                    new_policy[i][j] = - 1 / self.gamma * np.log(mu_prime)
            policy_func.approx(new_policy, lr=.5)

        self.policy_func = np.zeros((n_k, n_a))
        for i, k in enumerate(k_grids):
            for j, a in enumerate(a_grids):
                #self.policy_func[i][j] = self.resource(k_t * k_width + self.k_ss, a_t * a_width + a_center) - new_policy[i][j] ** (-1 / self.gamma)
                self.policy_func[i][j] = self.k_new(k, a, np.exp(new_policy[i][j]))
        self.policy_func_cheby = policy_func
        self.k_grids = k_grids

        old_weights = np.random.random((1, n_k * n_a))
        new_values = np.zeros((n_k, n_a))
        while not np.allclose(value_func.weights, old_weights, atol=self.tol, rtol=self.tol):
            # print(np.sum(np.abs(value_func.weights - old_weights)))
            old_weights = np.copy(value_func.weights)
            k_grids, a_grids = value_func.real_grids()
            for i, k in enumerate(k_grids):
                for j, a in enumerate(a_grids):
                    k_prime = self.policy_func[i][j]
                    c = self.resource(k, a) - k_prime
                    v_prime = 0
                    for idxa, a_prime in enumerate(a_grids):
                        v_prime += value_func.eval(k_prime, a_prime) * trans_mat[j][idxa]
                    new_values[i][j] = self.utility(c) + self.beta * v_prime
            value_func.approx(new_values, lr=0.2)
        self.value_func = new_values
        self.value_func_cheby = value_func
        #print(self.value_func_cheby.eval(self.k_ss, 0))

    def print_steady_state(self):
        print('Steady State:')
        print('K:', self.k_ss)
        print('C:', self.c_ss)

    def plot_policy(self, offgrid=False):
        if self.policy_func is None:
            return None
        if self.policy_func_cheby is None:
            offgrid = False
        for idxa in range(self.policy_func.shape[1]):
            a = self.a_grids[idxa]
            if offgrid:
                k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), 100, endpoint=True)
                c_grids = np.zeros(k_grids.size)
                for idxk, k in enumerate(k_grids):
                    c_grids[idxk] = np.exp(self.policy_func_cheby.eval(k, a))
                k_prime = self.k_new(k_grids, a, c_grids)
                plt.plot(k_grids, k_prime, label=str(round(a, 1)))
                plt.scatter(self.k_grids, self.policy_func[:, idxa], color='red')
            else:
                plt.plot(self.k_grids, self.policy_func[:, idxa], label=str(round(a, 1)))
        plt.plot(self.k_grids, self.k_grids, label='45 degree')
        plt.scatter([self.k_ss], [self.k_ss])
        plt.legend()
        plt.title('Capital Accumulation')
        plt.show()

    def plot_value(self, offgrid=False):
        if self.value_func is None:
            return None
        if self.value_func_cheby is None:
            offgrid = False
        for idxa in range(self.value_func.shape[1]):
            a = self.a_grids[idxa]
            if offgrid:
                k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), 100, endpoint=True)
                values = np.zeros(k_grids.size)
                for idxk, k in enumerate(k_grids):
                    values[idxk] = self.value_func_cheby.eval(k, a)
                plt.plot(k_grids, values, label=str(round(a, 2)))
                plt.scatter(self.k_grids, self.value_func[:, idxa], color='red')
            else:
                plt.plot(self.k_grids, self.value_func[:, idxa], label=str(round(a, 2)))
        plt.legend()
        plt.title('Value Function')
        plt.show()

    def plot_consumption(self, offgrid=True):
        if self.policy_func is None:
            return None
        if self.policy_func_cheby is None:
            offgrid = False
        for idxa in range(self.policy_func.shape[1]):
            a = self.a_grids[idxa]
            if offgrid:
                k_grids = np.linspace(np.min(self.k_grids), np.max(self.k_grids), 100, endpoint=True)
                consumptions = np.zeros(k_grids.size)
                for idxk, k in enumerate(k_grids):
                    consumptions[idxk] = np.exp(self.policy_func_cheby.eval(k, a))
                plt.plot(k_grids, consumptions, label=str(round(a, 2)))

            consumptions = np.zeros(self.k_grids.size)
            for idxk, k in enumerate(self.k_grids):
                consumptions[idxk] = self.resource(k, a) - self.policy_func[idxk][idxa]
            if offgrid:
                plt.scatter(self.k_grids, consumptions, color='red')
            else:
                plt.plot(self.k_grids, consumptions, label=str(round(a, 2)))
        plt.scatter([self.k_ss], [self.c_ss])
        plt.legend()
        plt.title('Consumption')
        plt.show()

    def steady_distribution(self, k_grids):
        if k_grids[0] > np.min(self.k_grids):
            k_grids = np.concatenate(([self.k_grids[0]], k_grids))
        k_num = k_grids.size
        k_max = np.concatenate((k_grids[1:], [self.k_grids[-1]]))
        k_values = (k_grids + k_max) / 2
        a_num = self.a_grids.size
        policy_func = np.zeros((k_num, a_num), dtype=int)
        for idxk, k in enumerate(k_values):
            for idxa, a in enumerate(self.a_grids):
                c = self.policy_func_cheby.eval(k, a)
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
            a_max = np.max(self.a_grids)
            a_min = np.min(self.a_grids)
            a_series = np.maximum(a_series, a_min)
            a_series = np.minimum(a_series, a_max)
            a_series = a_series[1:]
        k_series = np.zeros(periods + 1)
        c_series = np.zeros(periods)
        i_series = np.zeros(periods)
        y_series = np.zeros(periods)
        k_series[0] = self.k_ss

        k_max = np.max(self.k_grids)
        k_min = np.min(self.k_grids)
        for t in tqdm(range(periods)):
            c_series[t] = np.exp(self.policy_func_cheby.eval(k_series[t], a_series[t]))
            k_series[t + 1] = self.k_new(k_series[t], a_series[t], c_series[t])
            if k_series[t + 1] > k_max:
                k_series[t + 1] = k_max
            k_series[t + 1] = max(k_min, k_series[t + 1])
            c_series[t] = self.resource(k_series[t], a_series[t]) - k_series[t + 1]
            y_series[t] = np.exp(a_series[t]) * k_series[t] ** self.alpha
            i_series[t] = k_series[t + 1] - (1 - self.delta) * k_series[t]

        return a_series, k_series[:-1], c_series, i_series, y_series




