from scipy.optimize import root_scalar
from FunctionApprox import ExpLogLinearApprox
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from RandomProcess import AR1Process
from FunctionApprox import ChebyshevApprox2D
from joblib import Parallel, delayed
import tempfile, os

default_params = {
    'delta': 0.02,
    'gamma': 2,
    'beta': 0.99,
    'alpha': 0.5,
    'b': 0.01,
    'phi': 0.005
}


class GrowthWithHabitIACEnv:
    def __init__(self, params=None):
        if params is None:
            params = {}
        self.alpha = params.get('alpha', default_params['alpha'])
        self.beta = params.get('beta', default_params['beta'])
        self.gamma = params.get('gamma', default_params['gamma'])
        self.delta = params.get('delta', default_params['delta'])
        self.b = params.get('b', default_params['b'])
        self.phi = params.get('phi', default_params['phi'])
        self.k_ss, self.i_ss, self.c_ss, self.lamda_ss, self.mu_ss = 0, 0, 0, 0, 0

        self.policy_func = None
        self.value_func = None
        self.k_grids = None
        self.c_grids = None

    def compute_ss(self, k):
        i = self.delta * k
        c = k ** self.alpha - i - 1 / 2 * self.phi * i * i
        lamda = (1 - self.beta * self.b) * (c - self.b * c) ** (-self.gamma)
        mu = lamda * (1 + self.phi * i)
        return i, c, lamda, mu

    def foc_residual(self, k):
        i, c, lamda, mu = self.compute_ss(k)
        return mu * (1 / self.beta - 1 + self.delta) - lamda * self.alpha * k ** (self.alpha - 1)

    def solve_steady_state(self, x0=1, x1=2):
        result = root_scalar(self.foc_residual, x0=x0, x1=x1)
        if result.converged:
            self.k_ss = result.root
            self.i_ss, self.c_ss, self.lamda_ss, self.mu_ss = self.compute_ss(self.k_ss)
            print('Steady State (after', result.iterations, 'iterations):')
            print('Capital:', self.k_ss)
            print('Investment:', self.i_ss)
            print('Consumption:', self.c_ss)
            print('Consumption Shadow Price:', self.lamda_ss)
            print('Investment Shadow Price:', self.mu_ss)
        else:
            print('Fail to solve steady state:', result.flag)

    def pea(self, k_min, k_max, atol=1e-6, order=1, simulation=6000, buried=500):
        if self.k_ss == 0:
            self.solve_steady_state()
        k_width = max(self.k_ss - k_min, k_max - self.k_ss)
        c_width = self.c_ss / 2

        lamda_func = ExpLogLinearApprox(3, order)
        mu_func = ExpLogLinearApprox(3, order)

        k_grids = np.linspace(self.k_ss - k_width, self.k_ss + k_width, 50, endpoint=True)
        c_grids = np.linspace(self.c_ss - c_width, self.c_ss + c_width, 50, endpoint=True)
        a_grids = np.linspace(np.exp(-5), np.exp(5), 50, endpoint=True)
        x = np.array([[k, c, a] for k, c, a in product(k_grids, c_grids, a_grids)])

        lamda_func.approx(x, self.lamda_ss * np.ones(x.shape[0]) + np.random.uniform(-1, 1, x.shape[0]) * self.lamda_ss / 20 * 0)
        mu_func.approx(x, self.mu_ss * np.ones(x.shape[0]) + np.random.uniform(-1, 1, x.shape[0]) * self.mu_ss / 5)

        old_weight_mu = np.random.random(mu_func.get_weights().shape)
        old_weight_lamda = np.random.random(lamda_func.get_weights().shape)
        iter = 0
        while not (np.allclose(old_weight_mu, mu_func.get_weights(), atol=atol) and np.allclose(old_weight_lamda, lamda_func.get_weights(), atol=atol)):
            iter += 1
            old_weight_mu = np.copy(mu_func.get_weights())
            old_weight_lamda = np.copy(lamda_func.get_weights())

            c_series = np.zeros(simulation + 1)
            k_series = np.zeros(simulation + 1)
            i_series = np.zeros(simulation + 1)
            mu_series = np.zeros(simulation + 1)
            lamda_series = np.zeros(simulation + 1)
            a_series = AR1Process(0.8, err_params={'sigma': 0.1}).simulate(simulation, 0)

            c_series[0] = self.c_ss + (2 * np.random.rand() - 1) * c_width * 0
            k_series[1] = self.k_ss + (2 * np.random.rand() - 1) * k_width * 0
            for t in range(1, simulation):
                mu = mu_func.eval([[c_series[t - 1], k_series[t], np.exp(a_series[t])]])
                #print([c_series[t - 1], k_series[t], np.exp(a_series[t])])
                lamda = lamda_func.eval([[c_series[t - 1], k_series[t], np.exp(a_series[t])]])
                #print('mu', mu, 'lamda', lamda, (mu/lamda-1)/self.phi)
                i = (mu / lamda - 1) / self.phi
                k_prime = (1 - self.delta) * k_series[t] + i

                #k_prime = max(k_prime, self.k_ss - k_width)
                #k_prime = min(k_prime, self.k_ss + k_width)
                i = k_prime - k_series[t] * (1 - self.delta)
                c = np.exp(a_series[t]) * k_series[t] ** self.alpha - i - self.phi * i * i / 2
                while c < self.b * c_series[t - 1]+1:
                    i = i - 1
                    c = np.exp(a_series[t]) * k_series[t] ** self.alpha - i - self.phi * i * i / 2
                k_prime = (1 - self.delta) * k_series[t] + i
                i_series[t] = i
                c_series[t] = c
                mu_series[t] = mu
                lamda_series[t] = lamda
                k_series[t + 1] = k_prime

            print('Var c:', np.var(c_series[buried:-buried]))
            print('Var k:', np.var(k_series[buried:-buried]))
            if np.var(k_series[buried:-buried]) == np.nan:
                break
            #print(i_series)
            plt.plot(c_series / self.c_ss, label='c')
            plt.plot(i_series / self.i_ss, label='i')
            #plt.plot(np.exp(a_series), label='a')
            plt.legend()
            plt.title('Iterations ' + str(iter))
            plt.show()

            r_series = np.zeros(simulation + 1)
            for t in range(2, simulation - int(buried / 2)):
                lamda_series[t] = (c_series[t] - self.b * c_series[t - 1]) ** (-self.gamma) - self.beta * self.b * (c_series[t + 1] - self.b * c_series[t]) ** (-self.gamma)
                r_series[t + 1] = self.alpha * np.exp(a_series[t + 1]) * k_series[t + 1] ** (self.alpha - 1)
                mu_series[t] = self.beta * (r_series[t + 1] * lamda_series[t + 1] + mu_series[t + 1] * (1 - self.delta))
            plt.plot(mu_series / self.mu_ss, label='mu')
            plt.plot(lamda_series / self.lamda_ss, label='lamda')
            plt.plot(r_series, label='r')
            #plt.plot(np.exp(a_series), label='a')
            plt.legend()
            plt.title('Iterations ' + str(iter))
            plt.show()

            state_variables = np.stack([c_series[buried:-buried-1], k_series[buried + 1:-buried], np.exp(a_series[buried + 1:-buried])]).transpose()
            _, r2_mu = mu_func.approx(state_variables, mu_series[buried + 1:-buried], lr=0.1)
            _, r2_lamda = lamda_func.approx(state_variables, lamda_series[buried + 1:-buried], lr=0.1)
            print(mu_func.get_weights(), lamda_func.get_weights())

            print('Iterations', iter)
            print('R^2 of mu:', r2_mu)
            print('R^2 of lamda:', r2_lamda)
            print('Loss:', np.linalg.norm(old_weight_mu - mu_func.get_weights()) + np.linalg.norm(old_weight_lamda - lamda_func.get_weights()))

    def utility(self, c, c_last):
        return (c - self.b * c_last) ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c - self.b * c_last)

    def next_state(self, k, i, a=0):
        k_prime = (1 - self.delta) * k + i
        c = np.exp(a) * k ** self.alpha - i - self.phi * i * i / 2
        return k_prime, c

    def value_func_approx(self, k_width, c_width, n_k, n_c, atol=1e-5):
        if self.c_ss == 0:
            self.solve_steady_state()

        value_func = ChebyshevApprox2D(n_k, n_c, self.k_ss, self.c_ss, k_width, c_width)

        old_weights = np.random.random((1, n_k * n_c))
        #policy_func = np.zeros((n_k, n_c))
        #new_values = np.zeros((n_k, n_c))
        path = tempfile.mkdtemp()
        value_path = os.path.join(path, 'value.temp')
        new_values = np.memmap(value_path, dtype=float, shape=(n_k, n_c), mode='w+')
        policy_path = os.path.join(path, 'policy.temp')
        policy_func = np.memmap(policy_path, dtype=float, shape=(n_k, n_c), mode='w+')
        k_grids, c_grids = value_func.real_grids()

        iter = 1

        def update_value(i, j):
            k = k_grids[i]
            c_last = c_grids[j]
            y = k ** self.alpha
            invests = np.linspace(y - self.c_ss - c_width, y - self.c_ss + c_width, 500)
            invests = invests[y - invests - self.phi * invests * invests / 2 <= self.c_ss + c_width]
            invests = invests[y - invests - self.phi * invests * invests / 2 >= self.c_ss - c_width]
            invests = invests[y - invests - self.phi * invests * invests / 2 > c_last * self.b]
            invests = invests[(1 - self.delta) * k + invests >= self.k_ss - k_width]
            invests = invests[(1 - self.delta) * k + invests <= self.k_ss + k_width]
            k_primes, cs = self.next_state(k, invests)
            v_primes = np.zeros(k_primes.size)
            for idx in range(len(v_primes)):
                v_primes[idx] = value_func.eval(k_primes[idx], cs[idx])
                if not (-k_width <= k_primes[idx] - self.k_ss <= k_width and -c_width <= cs[
                    idx] - self.c_ss <= c_width):
                    print(k_primes[idx], cs[idx], invests[idx])
            v = self.utility(cs, c_last) + self.beta * v_primes
            new_values[i][j] = np.max(v)
            policy_func[i][j] = invests[np.argmax(v)]

        while not np.allclose(value_func.weights, old_weights, atol=atol):
            print('Iteration:', iter)
            iter += 1
            print('Loss:', np.sum(np.abs(value_func.weights - old_weights)))
            old_weights = np.copy(value_func.weights)
            Parallel(n_jobs=8)(delayed(update_value)(i, j) for i, j in product(range(len(k_grids)), range(len(c_grids))))
            '''
            for i, k in enumerate(k_grids):
                for j, c_last in enumerate(c_grids):
                    y = k ** self.alpha
                    invests = np.linspace(y - self.c_ss - c_width, y - self.c_ss + c_width, 250)
                    invests = invests[y - invests - self.phi * invests * invests / 2 <= self.c_ss + c_width]
                    invests = invests[y - invests - self.phi * invests * invests / 2 >= self.c_ss - c_width]
                    invests = invests[y - invests - self.phi * invests * invests / 2 > c_last * self.b]
                    invests = invests[(1 - self.delta) * k + invests >= self.k_ss - k_width]
                    invests = invests[(1 - self.delta) * k + invests <= self.k_ss + k_width]
                    k_primes, cs = self.next_state(k, invests)
                    v_primes = np.zeros(k_primes.size)
                    for idx in range(len(v_primes)):
                        v_primes[idx] = value_func.eval(k_primes[idx], cs[idx])
                        if not (-k_width<=k_primes[idx]-self.k_ss<=k_width and -c_width<=cs[idx]-self.c_ss<=c_width):
                            print(k_primes[idx], cs[idx], invests[idx])
                    v = self.utility(cs, c_last) + self.beta * v_primes
                    new_values[i][j] = np.max(v)
                    policy_func[i][j] = invests[np.argmax(v)]
            '''
            value_func.approx(new_values)
        self.policy_func = policy_func
        self.value_func = new_values
        self.k_grids, self.c_grids = value_func.real_grids()

    def plot_policy(self):
        if self.policy_func is None:
            return None
        for c in range(len(self.c_grids)):
            plt.plot(self.k_grids, self.policy_func[:, c], label=str(round(self.c_grids[c], 1)))
        plt.plot(self.k_grids, self.k_grids * self.delta, label='delta k')
        plt.scatter([self.k_ss], [self.i_ss])
        plt.legend()
        plt.show()

    def plot_value(self):
        if self.value_func is None:
            return None
        for c in range(len(self.c_grids)):
            plt.plot(self.k_grids, self.value_func[:, c], label=str(round(self.c_grids[c], 1)))
        plt.legend()
        plt.show()