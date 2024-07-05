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
    'phi': 0.005,
    'rho': 0.95,
    'sigma': 0.02
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
        self.rho = params.get('rho', default_params['rho'])
        self.sigma = params.get('sigma', default_params['sigma'])
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

    def pea(self, atol=1e-6, order=1, simulation=20500, buried=500, max_iter=5000, init_weight=None, lr=0.1, iri=False):
        if self.k_ss == 0:
            self.solve_steady_state()
        k_width = self.k_ss / 2
        c_width = self.c_ss / 2

        # mu_func = ExpLogLinearApprox(3, order)
        mu_func = ExpLogLinearApprox(2, order)
        if init_weight is None:
            k_grids = np.linspace(self.k_ss - k_width, self.k_ss + k_width, 50, endpoint=True)
            # c_grids = np.linspace(self.c_ss - c_width, self.c_ss + c_width, 50, endpoint=True)
            a_grids = np.linspace(np.exp(-5), np.exp(5), 50, endpoint=True)
            # x = np.array([[c, k, a] for k, c, a in product(k_grids, c_grids, a_grids)])
            x = np.array([[k, a] for k, a in product(k_grids, a_grids)])
            mu_ss = (self.c_ss - self.b * self.c_ss) ** (-self.gamma)
            mu_func.approx(x, mu_ss * np.ones(x.shape[0]))
        else:
            weight = mu_func.get_weights()
            for idx in range(len(init_weight)):
                weight[idx] = init_weight[idx]

        old_weight = np.random.random(mu_func.get_weights().shape)
        iter = 0
        r_squares = []
        weights = []
        losses = []
        c_series = np.zeros(simulation + 3)
        k_series = np.zeros(simulation + 3)
        i_series = np.zeros(simulation + 3)
        mu_series = np.zeros(simulation + 3)
        a_series = np.zeros(simulation + 3)

        eta_series = np.zeros(simulation + 3)
        while not (np.allclose(old_weight, mu_func.get_weights(), atol=atol)):
            if iter > max_iter:
                break
            iter += 1
            old_weight = np.copy(mu_func.get_weights())

            a_series = AR1Process(self.rho, err_params={'sigma': self.sigma}).simulate(simulation + 2, 0)
            c_series[0] = self.c_ss
            k_series[1] = self.k_ss
            for t in range(1, simulation + 2):
                y = np.exp(a_series[t]) * k_series[t] ** self.alpha
                # mu = mu_func.eval([[c_series[t - 1], k_series[t], np.exp(a_series[t])]])
                #print(t, [c_series[t - 1], k_series[t], np.exp(a_series[t])])
                mu = mu_func([[k_series[t], np.exp(a_series[t])]])
                c = mu ** (-1 / self.gamma) + self.b * c_series[t - 1]
                if k_series[t] > 1 / (self.phi * (1 - self.delta)):
                    #print('there', c, y)
                    c = min(c, y + 0.5 / self.phi - 0.5)
                else:
                    #print('here', t, k_series[t], c, y + k_series[t] * (1 - self.delta) - 0.5 * self.phi * (1 - self.delta) ** 2 * k_series[t] ** 2 - 1e-6)
                    c = min(c, y + k_series[t] * (1 - self.delta) - 0.5 * self.phi * (1 - self.delta) ** 2 * k_series[t] ** 2 - 0.5)
                if iri:
                    c = min(c, y)
                #print(t, self.phi, c, y, 1 + 2 * self.phi * (y - c))
                i = (np.sqrt(1 + 2 * self.phi * (y - c)) - 1) / self.phi
                k_prime = (1 - self.delta) * k_series[t] + i
                eta = (c - self.b * c_series[t - 1]) ** (-self.gamma) - mu
                mu = (c - self.b * c_series[t - 1]) ** (-self.gamma)
                i_series[t] = i
                c_series[t] = c
                mu_series[t] = mu
                k_series[t + 1] = k_prime
                eta_series[t] = eta

            # print('Var c:', np.var(c_series[buried:]))
            # print('Var k:', np.var(k_series[buried:]))
            if np.isnan(np.var(k_series[buried:])):
                break
            #print(i_series, k_series, c_series)
            plot = False
            if plot and iter % 500 == 0:
                plt.figure(1)
                plt.subplot(411)
                plt.plot(c_series[1:-1] / self.c_ss, label='c')
                plt.subplot(412)
                plt.plot(i_series[1:-1] / self.i_ss, label='i')
                plt.subplot(413)
                plt.plot(k_series[1:-1] / self.k_ss, label='k')
                plt.subplot(414)
                plt.plot(np.exp(a_series[1:-1]), label='a')
                #plt.legend()
                #plt.title('Iterations ' + str(iter))
                plt.show()
            #print(mu_series)
            '''
            for t in range(simulation - 1, 0, -1):
                r_prime = self.alpha * np.exp(a_series[t + 1]) * k_series[t + 1] ** (self.alpha - 1)
                mic = 1 + self.phi * i_series[t]    # marginal investment cost
                mic_prime = 1 + self.phi * i_series[t + 1]
                lamda_prime = mu_series[t + 1] - self.beta * self.b * mu_series[t + 2]  # shadow price of output
                mu_series[t] = self.beta * (self.b * mu_series[t + 1] + (r_prime + (1 - self.delta) * mic_prime) * lamda_prime / mic)
            '''
            for t in range(simulation):
                r_prime = self.alpha * np.exp(a_series[t + 1]) * k_series[t + 1] ** (self.alpha - 1)
                mic = 1 + self.phi * i_series[t]  # marginal investment cost
                mic_prime = 1 + self.phi * i_series[t + 1]
                lamda_prime = mu_series[t + 1] - self.beta * self.b * mu_series[t + 2]  # shadow price of output
                mu_series[t] = self.beta * (
                            self.b * mu_series[t + 1] + (r_prime + (1 - self.delta) * mic_prime) * lamda_prime / mic - (1 - self.delta) * eta_series[t + 1])

            # print(mu_series)
            '''
            if plot and iter % 200 == 0:
                plt.plot(mu_series, label='mu')
                plt.legend()
                plt.title('Iterations ' + str(iter))
                plt.show()
            '''
            if np.var(c_series[buried:simulation - 1]) < 1e-3:
                c_series += np.random.random(c_series.shape) * 0.1 - 0.05
            # state_variables = np.stack([c_series[buried:simulation - 1], k_series[buried + 1:simulation], np.exp(a_series[buried + 1:simulation])]).transpose()
            state_variables = np.stack([k_series[buried + 1:simulation], np.exp(a_series[buried + 1:simulation])]).transpose()
            _, r2 = mu_func.approx(state_variables, mu_series[buried + 1:simulation], lr=lr)
            #print(mu_func.get_weights())

            print('Iterations', iter)
            print('R^2 of mu:', r2)
            print('Loss:', np.linalg.norm(old_weight - mu_func.get_weights()))
            # print('cov', np.corrcoef(state_variables.transpose()))

            r_squares.append(r2)
            losses.append(np.linalg.norm(old_weight - mu_func.get_weights()))
            weights.append(mu_func.get_weights())

        plt.plot(np.maximum(r_squares, 0))
        plt.title('r^2')
        plt.show()

        plt.plot(np.log(losses))
        plt.title('log of loss')
        plt.show()

        weights = np.array(weights)
        for w in range(weights.shape[1]):
            plt.plot(weights[:, w])
            plt.title('Weight ' + str(w))
            plt.show()

        plt.figure(1)
        plt.subplot(411)
        plt.plot(c_series[1:-1] / self.c_ss, label='c')
        plt.gca().set_title('Consumption')
        plt.subplot(412)
        plt.plot(i_series[1:-1] / self.i_ss, label='i')
        plt.gca().set_title('Investment')
        plt.subplot(413)
        plt.plot(k_series[1:-1] / self.k_ss, label='k')
        plt.gca().set_title('Capital')
        plt.subplot(414)
        plt.plot(np.exp(a_series[1:-1]), label='a')
        plt.gca().set_title('TFP')
        # plt.legend()
        # plt.title('Iterations ' + str(iter))
        plt.show()

        plt.scatter(k_series[buried:simulation - 1], k_series[buried + 1:simulation])
        k_s = np.linspace(np.min(k_series[buried:simulation - 1]), np.max(k_series[buried:simulation - 1]), 100)
        plt.plot(k_s, k_s, label='45 degree line')
        plt.plot(k_s, (1 - self.delta) * k_s, label='remaining capital')
        plt.title('Decisions')
        plt.legend()
        plt.show()

        return mu_func.get_weights()

    def utility(self, c, c_last):
        u = np.zeros_like(c)
        u[c > self.b * c_last] = (c - self.b * c_last) ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 \
                                 else np.log(c - self.b * c_last)
        u[c <= self.b * c_last] = -np.inf
        return u

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
                v_primes[idx] = value_func(k_primes[idx], cs[idx])
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