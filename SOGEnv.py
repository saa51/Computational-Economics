from MarkovApprox import Tauchen
import numpy as np
from FunctionApprox import ChebyshevApprox, PolynomialApprox
from scipy.optimize import minimize_scalar, root_scalar
from RandomProcess import FiniteMarkov
from FunctionApprox import SplineApprox
import matplotlib.pyplot as plt


class SOGENV:
    _default_params = {
        'gamma': 2,
        'beta': 0.99,
        'delta': 0.025,
        'alpha': 0.36,
        'rho': 0.95,
        'sigma': 0.007,
        'tol': 1e-5
    }

    def __init__(self, normalized=True, params=None):
        # parameters
        if params is None:
            params = {}
        self.alpha = params.get('alpha', SOGENV._default_params['alpha'])
        self.beta = params.get('beta', SOGENV._default_params['beta'])
        self.rho = params.get('rho', SOGENV._default_params['rho'])
        self.sigma = params.get('sigma', SOGENV._default_params['sigma'])
        self.delta = params.get('delta', SOGENV._default_params['delta'])
        self.gamma = params.get('gamma', SOGENV._default_params['gamma'])
        self.tol = params.get('tol', SOGENV._default_params['tol'])

        if normalized:
            self.y_ss = 1
            self.k_ss = self.alpha / (1 / self.beta - 1 + self.delta)
            self.i_ss = self.k_ss * self.delta
            self.c_ss = self.y_ss - self.i_ss
            self.a_mean = -self.alpha * np.log(self.k_ss)
        else:
            self.k_ss = ((1 / self.beta - 1 + self.delta) / self.alpha) ** (1 / (self.alpha - 1))
            self.y_ss = self.k_ss ** self.alpha
            self.i_ss = self.k_ss * self.delta
            self.c_ss = self.y_ss - self.i_ss
            self.a_mean = 0

        # results
        self.a_grids = None
        self.trans_mat = None
        self.k_grids = None
        self.value_mat = None
        self.policy_mat = None
        self.value_func = None
        self.policy_func = None

    def utility(self, c):
        return c ** (1 - self.gamma) / (1 - self.gamma) if self.gamma != 1 else np.log(c)

    def value_func_iter(self, nk, na, width, m=3, **optimize_params):
        a_grids, trans_mat = Tauchen(self.rho, self.sigma ** 2, self.a_mean).approx(na, m)
        k_min = self.k_ss - width
        k_max = self.k_ss + width
        k_grids = np.linspace(k_min, k_max, nk, endpoint=True)
        self.a_grids, self.trans_mat, self.k_grids = a_grids, trans_mat, k_grids
        value_mat = np.zeros((nk, na))
        policy_mat = np.zeros((nk, na), dtype=int)

        if 'HPI' in optimize_params.get('methods', []):
            hpi_policy_iter = optimize_params.get('HPI_policy_iter', 5)
            hpi_value_iter = optimize_params.get('HPI_value_iter', 20)
            hpi = True
        else:
            hpi = False
            hpi_value_iter = 1e8
            hpi_policy_iter = 0
        mqp = 'MQP' in optimize_params.get('methods', [])

        u_mat = np.zeros((nk, na, nk))
        for idxk, k in enumerate(k_grids):
            for idxa, a in enumerate(a_grids):
                c = np.exp(a) * k ** self.alpha + (1 - self.delta) * k - k_grids
                c = np.maximum(c, np.min(c[c > 0]) / 100)
                u_mat[idxk][idxa] = self.utility(c)

        old_value_mat = np.random.random((nk, na))

        iter = 0
        while not np.allclose(old_value_mat, value_mat, atol=self.tol, rtol=0):
            iter += 1
            old_value_mat = np.copy(value_mat)
            if hpi and iter % hpi_policy_iter == 0:
                local_u = np.take_along_axis(u_mat, np.expand_dims(policy_mat, axis=-1), axis=-1).squeeze(axis=-1)
                local_vmat = np.copy(value_mat)
                for _ in range(hpi_value_iter):
                    value_mat = np.broadcast_to(trans_mat @ local_vmat.T, (nk, na, nk))
                    value_mat = local_u + self.beta * np.take_along_axis(value_mat, np.expand_dims(policy_mat, axis=-1), axis=-1).squeeze(axis=-1)
                    if mqp:
                        b_up = self.beta / (1 - self.beta) * np.max(value_mat - local_vmat)
                        b_low = self.beta / (1 - self.beta) * np.min(value_mat - local_vmat)
                        value_mat = value_mat + (b_up + b_low) / 2
            else:
                value_mat = np.max(u_mat + self.beta * np.dot(trans_mat, old_value_mat.transpose()), axis=-1)
                policy_mat = np.argmax(u_mat + self.beta * np.dot(trans_mat, old_value_mat.transpose()), axis=-1)

                if mqp:
                    b_up = self.beta / (1 - self.beta) * np.max(value_mat - old_value_mat)
                    b_low = self.beta / (1 - self.beta) * np.min(value_mat - old_value_mat)
                    value_mat = value_mat + (b_up + b_low) / 2
            
        policy_mat = np.argmax(u_mat + self.beta * np.dot(trans_mat, old_value_mat.transpose()), axis=-1)
        self.value_mat, self.policy_mat = value_mat, k_grids[policy_mat]
        return value_mat, k_grids[policy_mat]

    def value_approx(self, nk, na, width, deg, m, **optimize_params):
        a_grids, trans_mat = Tauchen(self.rho, self.sigma ** 2, self.a_mean).approx(na, m)
        k_min = self.k_ss - width
        k_max = self.k_ss + width
        self.a_grids, self.trans_mat = a_grids, trans_mat

        value_mat = np.zeros((nk, na))
        policy_mat = np.zeros((nk, na), dtype=float)
        k_grids = np.linspace(k_min, k_max, nk, endpoint=True)
        value_func = [ChebyshevApprox(self.k_ss, width + 1e-6, deg) for _ in range(na)]
        self.k_grids = k_grids

        if 'HPI' in optimize_params.get('methods', []):
            hpi_policy_iter = optimize_params.get('HPI_policy_iter', 5)
            hpi_value_iter = optimize_params.get('HPI_value_iter', 20)
            hpi = True
        else:
            hpi = False
            hpi_value_iter = 1e8
            hpi_policy_iter = 0
        mqp = 'MQP' in optimize_params.get('methods', [])

        k, a, idxa = 0, 0, 0

        def opposite_value(k_prime):
            return -(self.utility(np.exp(a) * k ** self.alpha + (1 - self.delta) * k - k_prime) +
                     self.beta * np.sum(np.array([value_func[i].eval(k_prime) for i in range(na)]) * trans_mat[idxa]))

        old_value_mat = np.random.random((nk, na))

        iter = 0
        while not np.allclose(old_value_mat, value_mat, atol=self.tol, rtol=0):
            iter += 1
            # print('iter {}: loss {}.'.format(iter, np.linalg.norm(old_value_mat - value_mat)))
            old_value_mat = np.copy(value_mat)

            if hpi and (iter % (hpi_value_iter + hpi_policy_iter)) >= hpi_value_iter:
                for idxk, k in enumerate(k_grids):
                    for idxa, a in enumerate(a_grids):
                        value_mat[idxk][idxa] = -opposite_value(policy_mat[idxk][idxa])
            else:
                for idxk, k in enumerate(k_grids):
                    for idxa, a in enumerate(a_grids):
                        cash = np.exp(a) * k ** self.alpha + (1 - self.delta) * k
                        res = minimize_scalar(opposite_value, bounds=(k_min, min(cash, k_max)), method='bounded')
                        value_mat[idxk][idxa] = -res.fun
                        policy_mat[idxk][idxa] = res.x

            if mqp:
                b_up = self.beta / (1 - self.beta) * np.max(value_mat - old_value_mat)
                b_low = self.beta / (1 - self.beta) * np.min(value_mat - old_value_mat)
                value_mat = value_mat + (b_up + b_low) / 2

            for idxa, func in enumerate(value_func):
                func.approx(value_mat[:, idxa], self.k_grids)
        self.value_mat, self.policy_mat = value_mat, policy_mat
        self.value_func = value_func
        return value_mat, policy_mat

    def modified_pea(self, nk, na, width, deg, m):
        a_grids, trans_mat = Tauchen(self.rho, self.sigma ** 2, self.a_mean).approx(na, m)
        k_min = self.k_ss - width
        k_max = self.k_ss + width
        self.a_grids, self.trans_mat = a_grids, trans_mat

        k_grids = np.linspace(k_min, k_max, nk, endpoint=True)
        self.k_grids = k_grids
        exp_func = [ChebyshevApprox(self.k_ss, width + 1e-6, deg) for _ in range(na)]

        for idxa in range(na):
            exp_func[idxa].approx([self.c_ss ** (-self.gamma) for _ in self.k_grids], grids=k_grids)

        old_weights = np.zeros(na * deg)
        new_weights = np.ones(na * deg)

        iter = 0
        while not np.allclose(old_weights, new_weights, rtol=0, atol=self.tol):
            iter += 1
            old_weights = np.copy(new_weights)
            new_weights = []

            for idxa, a in enumerate(self.a_grids):
                y = np.exp(a) * self.k_grids ** self.alpha + (1 - self.delta) * self.k_grids
                c = np.exp(exp_func[idxa].eval(self.k_grids)) ** (-1 / self.gamma)
                k_prime = y - c
                k_prime = np.maximum(k_prime, k_min)
                k_prime = np.minimum(k_prime, k_max)
                rhs = []
                for idxa_p, a_prime in enumerate(self.a_grids):
                    r_prime = self.alpha * np.exp(a_prime) * k_prime ** (self.alpha - 1) + 1 - self.delta
                    y_prime = np.exp(a_prime) * k_prime ** self.alpha + (1 - self.delta) * k_prime
                    sdf_prime = self.beta * np.exp(exp_func[idxa_p].eval(k_prime))
                    rhs.append(sdf_prime * r_prime)
                rhs = np.dot(trans_mat[idxa].reshape((1, -1)), np.array(rhs))
                new_weights.append(exp_func[idxa].approx(np.log(rhs), self.k_grids))

            new_weights = np.array(new_weights).ravel()

        policy_func = [SplineApprox() for _ in range(na)]
        for idxa, a in enumerate(self.a_grids):
            c = np.exp(exp_func[idxa].eval(self.k_grids)) ** (-1 / self.gamma)
            k_prime = np.exp(a) * self.k_grids ** self.alpha + (1 - self.delta) * self.k_grids - c
            policy_func[idxa].approx(self.k_grids, k_prime)
        self.policy_func = policy_func

    def simulate(self, periods=2500, tfp_series=None):
        if self.a_grids is None:
            return None
        na = self.a_grids.size
        if self.policy_func is None:
            policy_func = [SplineApprox() for _ in range(na)]
            for idxa in range(na):
                policy_func[idxa].approx(self.k_grids, self.policy_mat[:, idxa])
        else:
            policy_func = self.policy_func

        if tfp_series is None:
            idxa_series, a_series = FiniteMarkov(self.a_grids, self.trans_mat).simulate(periods, 0)
            tfp_series = (idxa_series, a_series)
        else:
            idxa_series, a_series = tfp_series
        k_series = np.zeros(periods + 2)
        k_series[0] = self.k_ss

        for t in range(0, periods + 1):
            k_series[t + 1] = policy_func[idxa_series[t]].eval(k_series[t])
        a_series = a_series[1:]
        k_series = k_series[1:]
        i_series = k_series[1:] - (1 - self.delta) * k_series[:-1]
        k_series = k_series[:-1]
        y_series = np.exp(a_series) * k_series ** self.alpha
        c_series = y_series - i_series
        return a_series, k_series, c_series, i_series, y_series, tfp_series

    def endo_grid(self, nk, na, width, deg, m, init=None):
        a_grids, trans_mat = Tauchen(self.rho, self.sigma ** 2, self.a_mean).approx(na, m)
        k_min = self.k_ss - width
        k_max = self.k_ss + width
        self.a_grids, self.trans_mat = a_grids, trans_mat

        value_mat = np.zeros((nk, na))

        k_grids = np.linspace(k_min, k_max, nk, endpoint=True)
        self.k_grids = k_grids
        future_vfunc = [ChebyshevApprox(self.k_ss, width + 1e-6, deg) for _ in range(na)]

        for idxa, a in enumerate(self.a_grids):
            if init is None:
                future_vfunc[idxa].approx(k_grids * self.c_ss ** (-self.gamma), grids=k_grids)
            else:
                v_mat, p_mat = init
                k_prime = p_mat[:, idxa]
                future_value = v_mat[:, idxa] - self.utility(np.exp(a) * self.k_grids ** self.alpha + (1 - self.delta) * self.k_grids - k_prime)
                future_vfunc[idxa].approx(future_value, grids=k_prime)

        old_weights = np.zeros(na * deg)
        new_weights = np.ones(na * deg)

        iter = 0
        while not np.allclose(old_weights, new_weights, rtol=0, atol=self.tol):
            iter += 1
            # print('iter {}: loss {}.'.format(iter, np.linalg.norm(old_weights - new_weights)))
            old_weights = np.copy(new_weights)
            #value_func = [PolynomialApprox(deg-3) for _ in range(na)]
            value_func = [SplineApprox() for _ in range(na)]
            for idxa, a in enumerate(self.a_grids):
                gradient = future_vfunc[idxa].gradient(self.k_grids)
                gradient = np.maximum(gradient, (self.c_ss * 5) ** (-self.gamma))
                c = gradient ** (-1 / self.gamma)
                y = c + self.k_grids
                y = np.maximum(y, np.exp(a) * k_min ** self.alpha + (1 - self.delta) * k_min)
                y = np.minimum(y, np.exp(a) * k_max ** self.alpha + (1 - self.delta) * k_max)
                c = y - self.k_grids
                value = self.utility(c) + future_vfunc[idxa].eval(self.k_grids)
                value_func[idxa].approx(y, value)

            for idxk, k in enumerate(self.k_grids):
                for idxa, a in enumerate(self.a_grids):
                    value_mat[idxk][idxa] = self.beta * np.dot(
                        [value_func[a_p].eval(np.exp(self.a_grids[a_p]) * k ** self.alpha + (1 - self.delta) * k) for a_p in range(na)], self.trans_mat[idxa])

            new_weights = []
            for idxa in range(na):
                new_weights.append(future_vfunc[idxa].approx(value_mat[:, idxa], self.k_grids))
            new_weights = np.array(new_weights).ravel()

        policy_func = [SplineApprox() for _ in range(na)]
        for idxa, a in enumerate(self.a_grids):
            c_vec = (future_vfunc[idxa].gradient(self.k_grids)) ** (-1 / self.gamma)
            y_vec = c_vec + self.k_grids
            k_vec = []
            for y in y_vec:
                def surplus(k, a, alpha, delta, y):
                    return np.exp(a) * k ** alpha + (1 - delta) * k - y

                def surplus_prime(k, a, alpha, delta, y):
                    return alpha * np.exp(a) * k ** (alpha - 1) + 1 - delta

                def surplus_2prime(k, a, alpha, delta, y):
                    return alpha * (alpha - 1) * np.exp(a) * k ** (alpha - 2)
                res = root_scalar(surplus, args=(a, self.alpha, self.delta, y), x0=self.k_ss, fprime=surplus_prime, fprime2=surplus_2prime, method='halley')
                k_vec.append(res.root)
            policy_func[idxa].approx(k_vec, self.k_grids)
        self.policy_func = policy_func

    def euler_error(self, grids):
        if self.a_grids is None:
            return None
        na = self.a_grids.size
        if self.policy_func is None:
            policy_func = [SplineApprox() for _ in range(na)]
            for idxa in range(na):
                policy_func[idxa].approx(self.k_grids, self.policy_mat[:, idxa])
        else:
            policy_func = self.policy_func
        grids = np.array(grids)
        k_list = grids[:, 0]
        idxa_list = np.argmin((grids[:, 1].reshape(-1, 1) - self.a_grids.reshape(1, -1)) ** 2, axis=1)
        res = []
        for k, idxa in zip(k_list, idxa_list):
            k_prime = policy_func[idxa].eval(k)
            c = np.exp(self.a_grids[idxa]) * k ** self.alpha + (1 - self.delta) * k - k_prime
            k_2primes = np.array([policy_func[i].eval(k_prime) for i in range(na)])
            c_primes = np.exp(self.a_grids) * k_prime ** self.alpha + (1 - self.delta) * k_prime - k_2primes
            r_primes = self.alpha * np.exp(self.a_grids) * k_prime ** (self.alpha - 1) + 1 - self.delta
            res.append(1 - (self.beta * np.sum(r_primes * self.trans_mat[idxa] * c_primes ** (-self.gamma))) ** (-1 / self.gamma) / c)
        return res

    def plot_value(self, title='', fname=None):
        if self.value_mat is None:
            return None
        for idxa in range(self.value_mat.shape[1]):
            a = self.a_grids[idxa]
            plt.plot(self.k_grids, self.value_mat[:, idxa], label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_policy(self, title='', fname=None):
        if self.policy_mat is None:
            return None
        for idxa in range(self.policy_mat.shape[1]):
            a = self.a_grids[idxa]
            plt.plot(self.k_grids, self.policy_mat[:, idxa], label='a=' + str(round(a, 2)))
        plt.plot(self.k_grids, self.k_grids, label='45 degree')
        plt.scatter([self.k_ss], [self.k_ss])
        plt.legend()
        plt.title('Policy Function: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_capital_diff(self, title='', fname=None):
        if self.policy_mat is None:
            return None
        for idxa in range(self.policy_mat.shape[1]):
            a = self.a_grids[idxa]
            plt.plot(self.k_grids, self.policy_mat[:, idxa] - self.k_grids, label='a=' + str(round(a, 2)))
        plt.plot()
        plt.legend()
        plt.title('Capital Difference: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_euler_err(self, title='', fname=None):
        na = self.a_grids.size
        errs = []
        for idxa in range(na):
            grids = [(self.k_grids[i], self.a_grids[idxa]) for i in range(self.k_grids.size)]
            err = self.euler_error(grids)
            err = np.log10(np.abs(err))
            plt.plot(self.k_grids, err, label='a=' + str(round(self.a_grids[idxa], 2)))
            errs.append(err)
        plt.legend()
        plt.title('Euler Error: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()
        return errs

    def plot_value_derivative(self, title='', fname=None):
        if self.value_mat is None:
            return None
        for idxa in range(self.value_mat.shape[1]):
            a = self.a_grids[idxa]
            plt.plot(self.k_grids[:-1], (self.value_mat[1:, idxa] - self.value_mat[:-1, idxa]) / (self.k_grids[1] - self.k_grids[0]), label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_value_2derivative(self, title='', fname=None):
        if self.value_mat is None:
            return None
        for idxa in range(self.value_mat.shape[1]):
            a = self.a_grids[idxa]
            gradient = (self.value_mat[1:, idxa] - self.value_mat[:-1, idxa]) / (self.k_grids[1] - self.k_grids[0])
            plt.plot(self.k_grids[:-2], (gradient[1:] - gradient[:-1]) / (self.k_grids[1] - self.k_grids[0]), label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Value Function 2nd Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_policy_derivative(self, title='', fname=None):
        if self.value_mat is None:
            return None
        for idxa in range(self.policy_mat.shape[1]):
            a = self.a_grids[idxa]
            plt.plot(self.k_grids[:-1], (self.policy_mat[1:, idxa] - self.policy_mat[:-1, idxa]) / (self.k_grids[1] - self.k_grids[0]), label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Policy Function Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

    def plot_policy_2derivative(self, title='', fname=None):
        if self.value_mat is None:
            return None
        for idxa in range(self.policy_mat.shape[1]):
            a = self.a_grids[idxa]
            gradient = (self.policy_mat[1:, idxa] - self.policy_mat[:-1, idxa]) / (self.k_grids[1] - self.k_grids[0])
            plt.plot(self.k_grids[:-2], (gradient[1:] - gradient[:-1]) / (self.k_grids[1] - self.k_grids[0]), label='a=' + str(round(a, 2)))
        plt.legend()
        plt.title('Policy Function 2nd Derivative: ' + title)
        if fname is not None:
            plt.savefig(fname)
        plt.show()

if __name__ == '__main__':
    SOGENV()