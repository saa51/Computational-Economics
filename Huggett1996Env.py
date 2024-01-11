import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import time, os
from FunctionApprox import SplineApprox


class Huggett1996Env:
    _default_params = {
        'beta': 0.994,
        'delta': 0.06,
        'alpha': 0.36,
        'sigma': 1.5,
        'tol': 1e-5,
        'n': 0.012,
        'N': 79,
        'A': 0.6,
        'a_bar': 0,
        'R': 54,
        'GY_ratio': 0.195,
        'theta': 0.1
    }

    def __init__(self, **kwargs):
        # parameters
        self.alpha = kwargs.get('alpha', Huggett1996Env._default_params['alpha'])
        self.beta = kwargs.get('beta', Huggett1996Env._default_params['beta'])
        self.sigma = kwargs.get('sigma', Huggett1996Env._default_params['sigma'])
        self.delta = kwargs.get('delta', Huggett1996Env._default_params['delta'])
        self.n = kwargs.get('n', Huggett1996Env._default_params['n'])
        self.N = kwargs.get('N', Huggett1996Env._default_params['N'])
        self.A = kwargs.get('A', Huggett1996Env._default_params['A'])
        self.a_bar = kwargs.get('a_bar', Huggett1996Env._default_params['a_bar'])
        self.R = kwargs.get('R', Huggett1996Env._default_params['R'])
        self.GY_ratio = kwargs.get('GY_ratio', Huggett1996Env._default_params['GY_ratio'])
        self.theta = kwargs.get('theta', Huggett1996Env._default_params['theta'])
        self.tol = kwargs.get('tol', Huggett1996Env._default_params['tol'])

        # data
        self.zvec = kwargs['zvec']
        self.pimat = kwargs['pimat']
        self.yvec = kwargs['yvec']
        self.z1prob = kwargs['z1probvec']
        self.znum = len(self.zvec)

        # stationary equilibrium
        self.mu_vec = None
        self.agg_labor = None
        self.avec = None
        self.anum = None
        self.policy_mat = None
        self.dists = None
        self.r, self.w, self.b, self.tau = 0, 0, 0, 0
        self.agg_k = None
        self.a_dist = None
        self.value_mat, self.consumption_mat, self.total_income_mat, self.labor_income_mat = None, None, None, None
        self.lorenz_curve = None

    def utility(self, c):
        return c ** (1 - self.sigma) / (1 - self.sigma) if self.sigma != 1 else np.log(c)

    def solve_decision_rule(self, amax, anum, w, r, b, tau, a_adj=0.1):
        # non-linear grids
        # a_grids = np.linspace(np.log(a_adj - self.a_bar), np.log(a_adj + amax), anum)
        # a_grids = np.exp(a_grids) - a_adj
        # linear grids
        a_grids = np.linspace(-self.a_bar, amax, anum)
        value_mat = np.zeros((self.N, anum, self.znum))
        policy_mat = np.zeros((self.N, anum, self.znum), dtype=int)
        consumption_mat = np.zeros((self.N, anum, self.znum))
        labor_income_mat = np.zeros((self.N, anum, self.znum))
        total_income_mat = np.zeros((self.N, anum, self.znum))

        # init final period
        total_income_mat[-1] = np.kron(a_grids.reshape((-1, 1)), np.ones((1, self.znum))) * (r * (1 - tau)) + b
        consumption_mat[-1] = np.kron(a_grids.reshape((-1, 1)), np.ones((1, self.znum))) * (1 + r * (1 - tau)) + b
        value_mat[-1] = np.kron(self.utility(a_grids * (1 + r * (1 - tau)) + b), np.ones(self.znum)).reshape((anum, self.znum))
        zero_index = np.argmin(a_grids ** 2)
        policy_mat[-1] = zero_index * np.ones((anum, self.znum), dtype=int)

        # backward induction
        for t in range(self.N - 2, -1, -1):
            b_t = 0 if t < self.R - 1 else b
            labor_income_mat[t] = (1 - self.theta - tau) * w * self.yvec[t] * np.kron(self.zvec.reshape((1, -1)), np.ones((anum, 1)))
            total_income_mat[t] = a_grids.reshape((-1, 1)) * r * (1 - tau) + labor_income_mat[t] + b_t    # a, z
            cash_on_hand = (a_grids * (1 - self.delta) + total_income_mat[t].T).T    # a, z
            future_value = np.dot(value_mat[t + 1], self.pimat.transpose())     # a', z
            c_mat = np.tile(np.expand_dims(cash_on_hand, axis=-1), (1, 1, anum)) - a_grids
            c_mat = np.maximum(c_mat, 1e-10)
            u_mat = self.utility(c_mat)     # a, z, a'
            value_mat[t] = np.max(u_mat + self.beta * future_value.T, axis=2)
            policy_mat[t] = np.argmax(u_mat + self.beta * future_value.T, axis=2)

            consumption_mat[t] = np.take_along_axis(c_mat, np.expand_dims(policy_mat[t], axis=-1), axis=-1).squeeze(axis=-1)

        return a_grids, policy_mat, value_mat, total_income_mat, labor_income_mat, consumption_mat

    def gen_stationary_distribution(self, avec, policy_mat):
        anum = len(avec)
        zero_idx = np.argmin(avec * avec)
        dists = np.zeros((self.N, anum, self.znum))
        dists[0][zero_idx] = self.z1prob

        for t in range(1, self.N):
            # p(a', z') = p(a, z) * g(a, z, a') * pi(z, z')
            for idxa, idxz in product(range(anum), range(self.znum)):
                dists[t][policy_mat[t - 1][idxa][idxz]] += dists[t - 1][idxa][idxz] * self.pimat[idxz]
        return dists

    def solve_stationary_equilibrium(self, amax, anum, r_init=-0.00016, max_iter=500, log=False):
        # cohort distribution
        self.mu_vec = (1 / (1 + self.n)) ** np.arange(self.N)
        self.mu_vec = self.mu_vec / np.sum(self.mu_vec)

        # labor supply
        if self.agg_labor is None:
            z_dists = []
            z_dist = self.z1prob
            for c in range(self.N):
                z_dists.append(z_dist)
                z_dist = np.dot(z_dist, self.pimat)
            z_dists = np.stack(z_dists)
            mean_z = np.dot(z_dists, self.zvec)
            self.agg_labor = np.sum(mean_z * self.yvec * self.mu_vec)

        r = r_init
        start_time = time.time()
        for iter in range(max_iter):
            agg_k = (r / self.alpha / self.A) ** (1 / (self.alpha - 1)) * self.agg_labor
            w = self.A * (1 - self.alpha) * (agg_k / self.agg_labor) ** self.alpha
            b = self.theta * w * self.agg_labor / np.sum(self.mu_vec[self.R - 1:])
            y = self.A * agg_k ** self.alpha * self.agg_labor ** (1 - self.alpha)
            g = self.GY_ratio * y
            tau = g / (y - self.delta * agg_k)

            avec, policy_mat, value_mat, total_income_mat, labor_income_mat, consumption_mat = \
                self.solve_decision_rule(amax, anum, w, r, b, tau)

            total_dists = self.gen_stationary_distribution(avec, policy_mat)

            dists = np.sum(total_dists, axis=2)
            agg_a = np.sum(np.dot(self.mu_vec, np.dot(dists, avec)))
            r_supply = self.A * self.alpha * (self.agg_labor / agg_a) ** (1 - self.alpha) - self.delta

            if log:
                print(f'Time {time.time() - start_time:.4f}, Iter {iter}: demand r {r:.4f}, supply r {r_supply:.4f}.\nAggregate capital {agg_k:.4f}, aggregate savings {agg_a:.4f}.')
                print(f'output {y:.4f}, wage {w:.4f}, transfer {b:.4f}, tax rate {tau:.4f}, aggregate labor {self.agg_labor:.4f}.')

            if np.abs(r - r_supply) < self.tol:
                self.avec = avec
                self.anum = len(avec)
                self.policy_mat = policy_mat
                self.dists = total_dists
                self.r, self.w, self.b, self.tau = r, w, b, tau
                self.agg_k = agg_a
                self.value_mat, self.consumption_mat, self.total_income_mat, self.labor_income_mat = \
                    value_mat, consumption_mat, total_income_mat, labor_income_mat
                break
            else:
                r = (r + r_supply) / 2

    def gen_wealth_distribution(self):
        if self.a_dist is None:
            self.a_dist = np.dot(self.mu_vec, np.sum(self.dists, axis=2))
        return self.a_dist

    def gini_index(self):
        a_dist = self.gen_wealth_distribution()
        a_cdf = np.cumsum(a_dist)
        a_cum = np.cumsum(a_dist * self.avec)
        a_cum = a_cum / a_cum[-1]
        gini = 0
        for a in range(self.anum - 1):
            gini += (a_cdf[a + 1] - a_cum[a + 1] + a_cdf[a] - a_cum[a]) * (a_cdf[a + 1] - a_cdf[a])
        return gini

    def percentile(self, q):
        a_dist = self.gen_wealth_distribution()
        a_cdf = np.cumsum(a_dist)
        idx = np.searchsorted(a_cdf, q)
        return self.avec[idx - 1]

    def plot_Lorenz_curve(self, figure_path=None):
        a_dist = self.gen_wealth_distribution()
        a_cdf = np.cumsum(a_dist)
        a_cum = np.cumsum(a_dist * self.avec)
        a_cum = a_cum / a_cum[-1]
        a_cdf = a_cdf.tolist()
        a_cdf = [0] + a_cdf
        a_cum = a_cum.tolist()
        a_cum = [0] + a_cum
        self.lorenz_curve = SplineApprox()
        self.lorenz_curve.approx(np.array(a_cdf), np.array(a_cum))
        plt.plot(a_cdf, a_cum, label='Lorenz Curve')
        plt.plot(a_cdf, a_cdf, label='45 degree line')
        plt.title('Lorenz Curve')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, 'Lorenz.pdf'))
        plt.show()

    def plot_policy(self, zidx=None, cidx=None, figure_path=None):
        if zidx is None:
            zidx = [0, 9, 17]
        if cidx is None:
            cidx = [0, self.R - 1, int(self.N / 3)]

        for c in cidx:
            for z in zidx:
                plt.plot(self.avec, self.avec[self.policy_mat[c, :, z]], label='z={}'.format(z + 1))
            plt.legend()
            plt.title('Policy Function: Age {}'.format(c + 1))
            if figure_path is not None:
                plt.savefig(os.path.join(figure_path, 'policy_{}.pdf'.format(c + 1)))
            plt.show()

    def plot_variance(self, figure_path=None):
        dists = self.dists
        dists = dists.reshape((self.N, -1))

        c_mat = self.consumption_mat
        c_mat = c_mat.reshape((self.N, -1))
        c_mean = np.sum(np.log(c_mat) * dists, axis=1)
        c_var = np.sum((np.log(c_mat) - c_mean.reshape((-1, 1))) ** 2 * dists, axis=1)

        ti_mat = self.total_income_mat
        ti_mat = ti_mat.reshape((self.N, -1))
        ti_mean = np.sum(np.log(ti_mat) * dists, axis=1)
        ti_var = np.sum((np.log(ti_mat) - ti_mean.reshape((-1, 1))) ** 2 * dists, axis=1)

        li_mat = self.labor_income_mat
        li_mat = li_mat.reshape((self.N, -1))
        li_mean = np.sum(np.log(li_mat) * dists, axis=1)
        li_var = np.sum((np.log(li_mat) - li_mean.reshape((-1, 1))) ** 2 * dists, axis=1)

        a_dists = np.sum(self.dists, axis=2)
        a_mean = np.dot(a_dists, self.avec)
        # a_var = np.sum((self.avec.reshape((1, -1)) - a_mean.reshape((-1, 1))) ** 2 * a_dists, axis=1)

        plt.plot(a_mean, label='asset holding')
        plt.plot(c_mean, label='consumption')
        plt.plot(ti_mean, label='total income')
        plt.plot(li_mean, label='labor income')
        plt.legend()
        plt.title('Mean')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, 'mean.pdf'))
        plt.show()

        # plt.plot(a_var, label='asset holding')
        plt.plot(c_var, label='consumption')
        plt.plot(ti_var, label='total income')
        plt.plot(li_var, label='labor income')
        plt.legend()
        plt.title('Variance')
        if figure_path is not None:
            plt.savefig(os.path.join(figure_path, 'variance.pdf'))
        plt.show()


