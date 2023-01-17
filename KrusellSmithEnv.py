import numpy as np
import logging, os, tempfile
from FunctionApprox import LinearApprox, PolynomialApprox
from itertools import product
from RandomProcess import FiniteMarkov
from scipy.interpolate import interp2d, interp1d, CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed


class KrusellSmithEnv:
    _default_params = {
        'beta': 0.99,
        'delta': 0.025,
        'alpha': 0.36,
        'sigma': 1,
        'tol': 1e-5,
        'h': 0.3271,
        'duration_unemp_good': 1.5,
        'duration_unemp_bad': 2.5,
        'u_good': 0.04,
        'u_bad': 0.1,
        'duration_good': 8,
        'duration_bad': 8,
        'z_good': 1.01,
        'z_bad': 0.99,
        'transfer': 1e-2
    }
    z_name = ['good', 'bad']
    e_name = ['employed', 'unemployed']

    def __init__(self, params=None):
        # parameters
        if params is None:
            params = {}
        self.alpha = params.get('alpha', KrusellSmithEnv._default_params['alpha'])
        self.beta = params.get('beta', KrusellSmithEnv._default_params['beta'])
        self.sigma = params.get('sigma', KrusellSmithEnv._default_params['sigma'])
        self.delta = params.get('delta', KrusellSmithEnv._default_params['delta'])
        self.h = params.get('h', KrusellSmithEnv._default_params['h'])
        self.duration_unemp_good = params.get('duration_unemp_good', KrusellSmithEnv._default_params['duration_unemp_good'])
        self.duration_unemp_bad = params.get('duration_unemp_bad', KrusellSmithEnv._default_params['duration_unemp_bad'])
        self.u_good = params.get('u_good', KrusellSmithEnv._default_params['u_good'])
        self.u_bad = params.get('u_bad', KrusellSmithEnv._default_params['u_bad'])
        self.duration_good = params.get('duration_good', KrusellSmithEnv._default_params['duration_good'])
        self.duration_bad = params.get('duration_bad', KrusellSmithEnv._default_params['duration_bad'])
        self.zg = params.get('z_good', KrusellSmithEnv._default_params['z_good'])
        self.zb = params.get('z_bad', KrusellSmithEnv._default_params['z_bad'])
        self.transfer = params.get('transfer', KrusellSmithEnv._default_params['transfer'])

        self.logger = logging.getLogger('FinalProject.KrusellSmithEnv')

        self.trans_mat = None
        self.z_trans_mat = None
        self.e_trans_mat = None

        self.policy_mat = None
        self.k_grids = None
        self.agg_k_grids = None
        self.k_motion = None

        self.agg_lg = self.h * (1 - self.u_good)
        self.agg_lb = self.h * (1 - self.u_bad)
        self.gen_transmat()

    def utility_func(self, c):
        return c ** (1 - self.sigma) / (1 - self.sigma) if self.sigma != 1 else np.log(c)

    def utility(self, c):
        return np.piecewise(c, [c <= 0, c > 0], [lambda x: 1e10 * (x - 1), self.utility_func])

    def gen_transmat(self):
        # copy transmat.m
        self.logger.info('Calculating transition matrix.')
        durug = self.duration_unemp_good
        unempg = self.u_good
        durgd = self.duration_good
        unempb = self.u_bad
        durbd = self.duration_bad
        durub = self.duration_unemp_bad

        pgg00 = (durug - 1.0) / durug
        pbb00 = (durub - 1.0) / durub
        pbg00 = 1.25 * pbb00
        pgb00 = 0.75 * pgg00
        pgg01 = (unempg - unempg * pgg00) / (1.0 - unempg)
        pbb01 = (unempb - unempb * pbb00) / (1.0 - unempb)
        pbg01 = (unempb - unempg * pbg00) / (1.0 - unempg)
        pgb01 = (unempg - unempb * pgb00) / (1.0 - unempb)
        pgg = (durgd - 1.0) / durgd
        pgb = 1.0 - (durbd - 1.0) / durbd
        pgg10 = 1.0 - (durug - 1.0) / durug
        pbb10 = 1.0 - (durub - 1.0) / durub
        pbg10 = 1.0 - 1.25 * pbb00
        pgb10 = 1.0 - 0.75 * pgg00
        pgg11 = 1.0 - (unempg - unempg * pgg00) / (1.0 - unempg)
        pbb11 = 1.0 - (unempb - unempb * pbb00) / (1.0 - unempb)
        pbg11 = 1.0 - (unempb - unempg * pbg00) / (1.0 - unempg)
        pgb11 = 1.0 - (unempg - unempb * pgb00) / (1.0 - unempb)
        pbg = 1.0 - (durgd - 1.0) / durgd
        pbb = (durbd - 1.0) / durbd

        pi = np.zeros((4, 4))
        pi[0][0] = pgg * pgg11
        pi[1][0] = pbg * pbg11
        pi[2][0] = pgg * pgg01
        pi[3][0] = pbg * pbg01

        pi[0][1] = pgb * pgb11
        pi[1][1] = pbb * pbb11
        pi[2][1] = pgb * pgb01
        pi[3][1] = pbb * pbb01

        pi[0][2] = pgg * pgg10
        pi[1][2] = pbg * pbg10
        pi[2][2] = pgg * pgg00
        pi[3][2] = pbg * pbg00

        pi[0][3] = pgb * pgb10
        pi[1][3] = pbb * pbb10
        pi[2][3] = pgb * pgb00
        pi[3][3] = pbb * pbb00

        self.trans_mat = np.transpose(pi)
        self.z_trans_mat = np.array([[pgg, pgb], [pbg, pbb]])   # Pr{z'|z} z, z'
        # Pr{e'|e, z, z'}   z, z', e, e'
        self.e_trans_mat = np.array([[[[pgg11, pgg01], [pgg10, pgg00]],
                                      [[pbg11, pbg01], [pbg10, pbg00]]],
                                     [[[pgb11, pgb01], [pgb10, pgb00]],
                                      [[pbb11, pbb01], [pbb10, pbb00]]]
                                     ])
        self.logger.debug('Transition Matrix:\n' + str(self.trans_mat))
        self.logger.debug('Z Transition Matrix:\n' + str(self.z_trans_mat))
        self.logger.debug(
            'e Transition Matrix\nGood to Good:\n{}\nGood to Bad:\n{}\nBad to Good:\n{}\nBad to Bad:\n{}'.format(
                self.e_trans_mat[0][0], self.e_trans_mat[0][1], self.e_trans_mat[1][0], self.e_trans_mat[1][1]))

    def no_agg_risk_VFI(self, r, w, trans_mat, k_max=100, k_num=500, tol=1e-5, **optimize_params):

        k_adj = 0.5
        k_min = 0
        #k_grids = np.linspace(np.log(k_min + k_adj), np.log(k_max + k_adj), k_num)
        #k_grids = np.exp(k_grids) - k_adj
        k_grids = np.linspace(k_min, k_max, k_num, endpoint=True)

        e_grids = np.array([0, 1])
        e_num = 2
        value_mat = np.zeros((k_num, e_num))
        policy_mat = np.zeros((k_num, e_num), dtype=int)

        if 'HPI' in optimize_params.get('methods', []):
            hpi_policy_iter = optimize_params.get('HPI_policy_iter', 5)
            hpi_value_iter = optimize_params.get('HPI_value_iter', 20)
            hpi = True
        else:
            hpi = False
            hpi_value_iter = 1e8
            hpi_policy_iter = 0
        mqp = 'MQP' in optimize_params.get('methods', [])

        cash_in_hand = (1 + r - self.delta) * k_grids.reshape((-1, 1)) + self.h * w * e_grids.reshape((1, -1)) + self.transfer
        c_mat = np.kron(cash_in_hand, np.ones(k_num)).reshape((k_num, e_num, k_num)) - k_grids
        u_mat = self.utility(c_mat)

        old_value_mat = np.random.random((k_num, e_num))

        iter = 0
        while not np.allclose(old_value_mat, value_mat, atol=tol, rtol=0):
            iter += 1
            old_value_mat = np.copy(value_mat)
            future_value = np.dot(old_value_mat, trans_mat.transpose())
            if hpi and (iter % (hpi_value_iter + hpi_policy_iter)) >= hpi_value_iter:
                for idxk, k in enumerate(k_grids):
                    for idxe, e in enumerate(e_grids):
                        value_mat[idxk][idxe] = u_mat[idxk][idxe][policy_mat[idxk][idxe]] + self.beta * np.dot(
                            trans_mat[idxe], old_value_mat.transpose()[:, policy_mat[idxk][idxe]])
            else:
                value_mat = np.max(u_mat + self.beta * future_value.transpose(), axis=2)
                policy_mat = np.argmax(u_mat + self.beta * future_value.transpose(), axis=2)

            if mqp:
                b_up = self.beta / (1 - self.beta) * np.max(value_mat - old_value_mat)
                b_low = self.beta / (1 - self.beta) * np.min(value_mat - old_value_mat)
                value_mat = value_mat + (b_up + b_low) / 2
        # self.logger.debug(policy_mat[-10:, :])
        return value_mat, policy_mat, k_grids

    def no_agg_risk_aggregation(self, policy_mat, k_grids, trans_mat, tol=1e-5):
        k_num = len(k_grids)
        k_init = int(k_num / 2)
        pi = np.zeros((k_num, 2))
        pi[k_init][0] = pi[k_init][1] = 0.5
        pi = pi.reshape((1, -1))
        transition = np.zeros((k_num * 2, k_num * 2))
        for idxk in range(len(k_grids)):
            for idxe in range(2):
                transition[idxk * 2 + idxe][policy_mat[idxk][idxe] * 2 + 0] = trans_mat[idxe][0]
                transition[idxk * 2 + idxe][policy_mat[idxk][idxe] * 2 + 1] = trans_mat[idxe][1]
        pi_old = np.random.random((1, k_num * 2))
        while not np.allclose(pi, pi_old, atol=tol, rtol=0):
            pi_old = np.copy(pi)
            pi = np.dot(pi, transition)
        pi = pi.reshape((k_num, 2))
        agg_k = np.dot(k_grids, np.sum(pi, axis=1))
        return pi, agg_k

    def no_agg_risk_equilibrium(self, state, r_init=0.03, max_iter=200, tol=1e-5, lr=0.5):
        self.logger.info('Calculating the stationary equilibrium without aggregate uncertainty when state is {}.'.format(state))
        if state == 'good':
            trans_mat = np.array([[self.trans_mat[0][0], self.trans_mat[0][2]],
                                 [self.trans_mat[2][0], self.trans_mat[2][2]]])
            agg_l = self.agg_lg
            z = self.zg
        else:
            trans_mat = np.array([[self.trans_mat[1][1], self.trans_mat[1][3]],
                                 [self.trans_mat[3][1], self.trans_mat[3][3]]])
            agg_l = self.agg_lb
            z = self.zb
        trans_mat = trans_mat / np.sum(trans_mat, axis=1).reshape((-1, 1))

        optimize_dict = {
            'methods': [# 'HPI',
                        'MQP'],
            'HPI_policy_iter': 20,
            'HPI_value_iter': 10
        }
        r = r_init
        agg_k, w = 0, 0
        large_r, small_r = [1], [0]
        solved = False
        for iter in range(max_iter):
            agg_k = agg_l * (r / z / self.alpha) ** (1 / (self.alpha - 1))
            w = (1 - self.alpha) * z * (agg_k / agg_l) ** self.alpha
            value_mat, policy_mat, k_grids = self.no_agg_risk_VFI(r, w, trans_mat, **optimize_dict)
            dist, k_supply = self.no_agg_risk_aggregation(policy_mat, k_grids, trans_mat)
            r_supply = self.alpha * z * (agg_l / k_supply) ** (1 - self.alpha)
            self.logger.debug('Iteration {}: r demand {}, r supply {}, k demand {}, k supply {}, labor {}, wage {}.'.
                              format(iter + 1, r, r_supply, agg_k, k_supply, agg_l, w))
            if np.abs(agg_k - k_supply) < tol or (np.min(large_r) - np.max(small_r) < tol):
                solved = True
                self.logger.info(
                    'Solution: interest rate {}, capital {}, labor {}, wage {}.'.format(r, agg_k, agg_l, w))
                break
            large_r.append(max(r_supply, r))
            small_r.append(min(r_supply, r))
            r = (np.min(large_r) + np.max(small_r)) / 2
        if not solved:
            self.logger.warning(
                'Cannot solve the stationary equilibrium without aggregate uncertainty when state is {}.'.format(state))

    def KS_vfi(self, k_motion, k_max=100, k_num=500, agg_k_max=15, agg_k_min=5, agg_k_num=4, tol=1e-2, max_iter=2000, value_init=None, **optimize_params):
        self.logger.info('Solving Krusell-Smith agent problem.')
        e_grids = np.array([1, 0])
        e_num = 2
        z_grids = np.array([self.zg, self.zb])
        l_grids = np.array([self.agg_lg, self.agg_lb])
        k_adj = 0.5
        k_min = 0
        k_grids = np.linspace(np.log(k_min + k_adj), np.log(k_max + k_adj), k_num)
        k_grids = np.exp(k_grids) - k_adj

        #k_grids = np.linspace(k_min, k_max, k_num)
        agg_k_grids = np.linspace(agg_k_min, agg_k_max, agg_k_num)
        if value_init is None:
            value_mat = np.zeros((agg_k_num, 2, 2, k_num))  # K, e, z, k
        else:
            value_mat = value_init
        # on the grid policy
        index_mat = np.ones((agg_k_num, 2, 2, k_num), dtype=int) * int(k_num / 2)

        policy_mat = np.zeros((agg_k_num, 2, 2, k_num))
        old_value_mat = np.random.random((agg_k_num, 2, 2, k_num))

        if 'HPI' in optimize_params.get('methods', []):
            hpi_policy_iter = optimize_params.get('HPI_policy_iter', 5)
            hpi_value_iter = optimize_params.get('HPI_value_iter', 20)
            hpi = True
        else:
            hpi = False
            hpi_value_iter = 1e8
            hpi_policy_iter = 0
        mqp = 'MQP' in optimize_params.get('methods', [])

        for iter in range(max_iter):
            old_value_mat = np.copy(value_mat)
            for idxz, z in enumerate(z_grids):
                for idx_aggk, agg_k in enumerate(agg_k_grids):
                    # aggregate variables
                    r = self.alpha * z * (l_grids[idxz] / agg_k) ** (1 - self.alpha)
                    w = (1 - self.alpha) * z * (agg_k / l_grids[idxz]) ** self.alpha
                    agg_k_prime = np.exp(k_motion[idxz].eval(np.log(agg_k)))

                    # right hand side
                    cash_in_hand = w * self.h * e_grids.reshape((1, -1)) + (1 - self.delta + r) * k_grids.reshape((-1, 1)) + self.transfer
                    c_mat = np.kron(cash_in_hand, np.ones(k_num)).reshape((k_num, e_num, k_num)) - k_grids
                    u_mat = self.utility(c_mat)     # k, e, k'
                    local_value_mat = np.zeros((2, 2, k_num))   # e, z, k
                    for idxe, idxz_p, idxk in product(range(2), range(2), range(k_num)):
                        coeff = np.polyfit(agg_k_grids, old_value_mat[:, idxe, idxz_p, idxk], len(agg_k_grids) - 1)
                        local_value_mat[idxe][idxz_p][idxk] = np.poly1d(coeff)(agg_k_prime)
                    # self.logger.debug(local_value_mat)
                    local_value_mat = local_value_mat.reshape((4, -1))  # (ez)' k'
                    future_value = np.dot(self.trans_mat, local_value_mat)  # (ez) k'
                    if idxz == 0:
                        # good state
                        future_value = future_value[[0, 2], :]
                    else:
                        # bad state
                        future_value = future_value[[1, 3], :]
                    rhs = u_mat + self.beta * future_value
                    # self.logger.debug(rhs[-1, 1])

                    # on the grid policy
                    policy_mat[idx_aggk, :, idxz, :] = k_grids[np.argmax(rhs, axis=2).transpose()]
                    value_mat[idx_aggk, :, idxz, :] = np.max(rhs, axis=2).transpose()

                    # cubic interpolation, which is used in K-S paper, not work
                    '''
                    for idxe, idxk in product(range(e_num), range(k_num)):
                        if len(k_grids[k_grids <= cash_in_hand[idxk][idxe]]) < 10:
                            policy_mat[idx_aggk][idxe][idxz][idxk] = k_grids[np.argmax(rhs[idxk][idxe])]
                            value_mat[idx_aggk][idxe][idxz][idxk] = np.max(rhs[idxk][idxe])
                        else:
                            func = CubicSpline(k_grids[k_grids <= cash_in_hand[idxk][idxe]], rhs[idxk][idxe][k_grids <= cash_in_hand[idxk][idxe]])
                            if (iter + 1) % (hpi_value_iter + hpi_policy_iter) < hpi_value_iter:
                                derivative = func.derivative()
                                k_vec = np.append(derivative.roots(), [k_min, min(k_max, cash_in_hand[idxk][idxe])])
                                k_vec = k_vec[k_vec <= cash_in_hand[idxk][idxe]]
                                value_vec = func(k_vec)
                                # self.logger.debug('{}, {}, {}.'.format(cash_in_hand[idxk][idxe], k_vec, value_vec))
                                idx = np.argmax(value_vec)
                                policy_mat[idx_aggk][idxe][idxz][idxk] = k_vec[idx]
                                value_mat[idx_aggk][idxe][idxz][idxk] = value_vec[idx]
                            else:
                                value_mat[idx_aggk][idxe][idxz][idxk] = func(policy_mat[idx_aggk][idxe][idxz][idxk])
                    '''
            if mqp and iter > 5:
                b_up = self.beta / (1 - self.beta) * np.max(value_mat - old_value_mat)
                b_low = self.beta / (1 - self.beta) * np.min(value_mat - old_value_mat)
                value_mat = value_mat + (b_up + b_low) / 2
            # self.logger.debug(policy_mat)
            loss = np.max(np.abs(value_mat - old_value_mat))
            if iter % 10 == 0:
                self.logger.debug('VFI Iteration {}: loss {}.'.format(iter, loss))
            if loss < tol and (not hpi or (iter + 1) % (hpi_value_iter + hpi_policy_iter) < hpi_value_iter):
                break
        # self.logger.debug(value_mat)
        self.logger.info('Finish VFI.')
        return value_mat, policy_mat, k_grids, agg_k_grids

    def refine_policy(self, k_motion, value_mat, k_grids, agg_k_grids, k_num=200, agg_k_num=50):
        self.logger.info('Refine policy.')

        old_k_num = len(k_grids)
        old_k_grids = k_grids
        old_aggk_grids = agg_k_grids

        e_grids = np.array([1, 0])
        e_num = 2
        z_grids = np.array([self.zg, self.zb])
        l_grids = np.array([self.agg_lg, self.agg_lb])
        k_adj = 0.5
        k_min, k_max = np.min(k_grids), np.max(k_grids)
        k_grids = np.linspace(np.log(k_min + k_adj), np.log(k_max + k_adj), k_num)
        k_grids = np.exp(k_grids) - k_adj
        # k_grids = np.linspace(k_min, k_max, k_num)
        agg_k_min, agg_k_max = np.min(agg_k_grids), np.max(agg_k_grids)
        agg_k_grids = np.linspace(agg_k_min, agg_k_max, agg_k_num)

        # on the grid policy

        policy_mat = np.zeros((agg_k_num, 2, 2, k_num))

        for idxz, z in enumerate(z_grids):
            for idx_aggk, agg_k in enumerate(agg_k_grids):
                # aggregate variables
                r = self.alpha * z * (l_grids[idxz] / agg_k) ** (1 - self.alpha)
                w = (1 - self.alpha) * z * (agg_k / l_grids[idxz]) ** self.alpha
                agg_k_prime = np.exp(k_motion[idxz].eval(np.log(agg_k)))

                # right hand side
                cash_in_hand = w * self.h * e_grids.reshape((1, -1)) + (1 - self.delta + r) * k_grids.reshape(
                        (-1, 1)) + self.transfer
                c_mat = np.kron(cash_in_hand, np.ones(k_num)).reshape((k_num, e_num, k_num)) - k_grids
                u_mat = self.utility(c_mat)  # k, e, k'
                local_value_mat = np.zeros((2, 2, old_k_num))  # e, z, k
                for idxe, idxz_p, idxk in product(range(2), range(2), range(old_k_num)):
                    coeff = np.polyfit(old_aggk_grids, value_mat[:, idxe, idxz_p, idxk], len(old_aggk_grids) - 1)
                    local_value_mat[idxe][idxz_p][idxk] = np.poly1d(coeff)(agg_k_prime)
                # self.logger.debug(local_value_mat)
                local_value_mat = local_value_mat.reshape((4, -1))  # (ez)' k'
                future_value = np.dot(self.trans_mat, local_value_mat)  # (ez) k'
                if idxz == 0:
                    # good state
                    future_value = future_value[[0, 2], :]  # e, k'
                else:
                    # bad state
                    future_value = future_value[[1, 3], :]  # e, k'
                refine_future_value = np.zeros((e_num, k_num))
                for idxe in range(e_num):
                    func = interp1d(old_k_grids, future_value[idxe])
                    refine_future_value[idxe] = func(k_grids)
                rhs = u_mat + self.beta * refine_future_value
                # self.logger.debug(rhs[-1, 1])

                # on the grid policy
                policy_mat[idx_aggk, :, idxz, :] = k_grids[np.argmax(rhs, axis=2).transpose()]

        self.logger.info('Finish refining policy.')
        return policy_mat, k_grids, agg_k_grids

    def simulate_aggk(self, policy_mat, k_grids, agg_k_grids, z_series=None, length=6000, agent_num=5000):
        self.logger.info('Simulating aggregate capital path.')
        if z_series is None:
            idxz_series, z_series = FiniteMarkov(np.array([self.zg, self.zb]), self.z_trans_mat).simulate(length, self.zg)
        else:
            idxz_series, z_series = z_series
        # self.logger.debug('{} {} {}'.format(agg_k_grids.size, k_grids.size, k_grids[policy_mat[:, 0, 0, :]].shape))

        # on the grid policy
        '''
        policy_func = [[interp2d(agg_k_grids, k_grids, k_grids[policy_mat[:, 0, 0, :]].transpose()),
                        interp2d(agg_k_grids, k_grids, k_grids[policy_mat[:, 0, 1, :]].transpose())],
                       [interp2d(agg_k_grids, k_grids, k_grids[policy_mat[:, 1, 0, :]].transpose()),
                        interp2d(agg_k_grids, k_grids, k_grids[policy_mat[:, 1, 1, :]].transpose())]]   # e, z
        '''
        # off the grid policy
        policy_func = [[interp2d(agg_k_grids, k_grids, policy_mat[:, 0, 0, :].transpose()),
                        interp2d(agg_k_grids, k_grids, policy_mat[:, 0, 1, :].transpose())],
                       [interp2d(agg_k_grids, k_grids, policy_mat[:, 1, 0, :].transpose()),
                        interp2d(agg_k_grids, k_grids, policy_mat[:, 1, 1, :].transpose())]]   # e, z

        # initialize
        kvec = np.random.random(agent_num) * (np.max(agg_k_grids) - np.min(agg_k_grids)) + np.min(agg_k_grids)
        evec = np.zeros(agent_num, dtype=int)
        evec[np.random.random(agent_num) < self.u_good] = 1     # 1 = unemployed, 0 = employed, state index
        agg_k = np.zeros(length)
        for t in range(length):
            agg_k[t] = np.mean(kvec)
            kvec = (1 - evec) * policy_func[0][idxz_series[t]](agg_k[t], kvec).ravel() + \
                   evec * policy_func[1][idxz_series[t]](agg_k[t], kvec).ravel()
            prob = self.e_trans_mat[idxz_series[t], idxz_series[t + 1], :, 0]
            ran_vec = np.random.random(agent_num)
            thres = evec * prob[1] + (1 - evec) * prob[0]
            evec = np.where(ran_vec < thres, 0, 1)
        plt.show()
        return agg_k, idxz_series, z_series

    def solve_approximated_equilibrium(self, order=1, max_iter=200, simulation_length=11000, agent_num=5000, burned=1000, tol=1e-3):
        vfi_tol = 1e-2
        if order != 1:
            raise NotImplementedError
        self.logger.info('Solving K-S approximated equilibrium.')
        k_motion = [LinearApprox(order) for _ in range(2)]

        for z in range(2):
            k_motion[z].weights[0] = np.log(11.5) * 0.05
            k_motion[z].weights[1] = .95

        old_weights = np.random.random((order + 1) * 2)
        new_weights = np.concatenate([k_motion[i].weights for i in range(2)])

        optimize_dict = {
            'methods': ['HPI', 'MQP'],
            'HPI_policy_iter': 20,
            'HPI_value_iter': 10
        }
        value_mat = None
        for iter in range(max_iter):
            old_weights = np.copy(new_weights)
            value_mat, policy_mat, k_grids, agg_k_grids = self.KS_vfi(k_motion, value_init=value_mat, tol=vfi_tol,
                                                                      **optimize_dict)
            vfi_tol = max(vfi_tol * 0.9, 1e-4)
            # self.plot_policy(policy_mat, k_grids, agg_k_grids)
            # policy_mat, k_grids, agg_k_grids = self.refine_policy(k_motion, value_mat, k_grids, agg_k_grids)
            agg_k, idxz, z = self.simulate_aggk(policy_mat, k_grids, agg_k_grids, length=simulation_length,
                                                agent_num=agent_num)
            # plt.plot(agg_k[burned:])
            # plt.show()
            agg_k, idxz, z = agg_k[burned:], idxz[burned:-1], z[burned:-1]
            y = agg_k[1:]
            x = agg_k[:-1]

            plt_grid = np.linspace(np.min(agg_k), np.max(agg_k), 100)
            for idxz_value in [0, 1]:
                local_y = np.log(y[idxz[:-1] == idxz_value])
                local_x = np.log(x[idxz[:-1] == idxz_value])
                plt.scatter(np.exp(local_x), np.exp(local_y), label=KrusellSmithEnv.z_name[idxz_value], s=1)
                weights, r2 = k_motion[idxz_value].approx(local_x, local_y, lr=0.1)
                plt.plot(np.exp(local_x), np.exp(k_motion[idxz_value].eval(local_x.reshape(-1, 1))),
                         label=KrusellSmithEnv.z_name[idxz_value])
                self.logger.info("Regression for {} state: log(K') = {} + {} * log(K), r2 = {}.".format(
                    KrusellSmithEnv.z_name[idxz_value], round(weights[0], 3), round(weights[1], 3), r2))
            plt.plot(plt_grid, plt_grid, label='45 degree')
            plt.legend()
            plt.title('Iter {}: aggregate capital motion'.format(iter))
            plt.show()

            new_weights = np.concatenate([k_motion[i].weights for i in range(2)])
            iter = iter + 1
            loss = np.max(np.abs(old_weights - new_weights))
            self.logger.info('K-S Iteration {}: loss {}.'.format(iter, loss))
            if loss < tol:
                self.logger.info('Model solved after {} iterations.'.format(iter))
                self.policy_mat = policy_mat
                self.k_grids = k_grids
                self.agg_k_grids = agg_k_grids
                self.k_motion = k_motion
                break

    def plot_policy(self, policy_mat=None, k_grids=None, agg_k_grids=None, max_k=25, figure_path=None):
        if self.policy_mat is None and policy_mat is None:
            return None
        elif policy_mat is None:
            policy_mat = self.policy_mat
            k_grids = self.k_grids
            agg_k_grids = self.agg_k_grids
        for idx_aggk in range(len(agg_k_grids)):
            for idxz in [0, 1]:
                for idxe in [0, 1]:
                    plt.plot(k_grids[k_grids < max_k], policy_mat[idx_aggk][idxe][idxz][k_grids < max_k],
                             label='idiosyncratic state {}'.format(KrusellSmithEnv.e_name[idxe]))
                plt.plot(k_grids[k_grids < max_k], k_grids[k_grids < max_k], label='45 degree')
                plt.legend()
                plt.title('Decision Rule: aggregate state {}, aggregate K = {}, '.
                          format(KrusellSmithEnv.z_name[idxz], round(agg_k_grids[idx_aggk], 2)))
                if figure_path is not None:
                    plt.savefig(os.path.join(figure_path, 'policy_{}_{}.pdf'.format(idx_aggk, idxz)))
                    plt.savefig(os.path.join(figure_path, 'policy_{}_{}.png'.format(idx_aggk, idxz)))
                plt.show()

    def full_simulation(self, z_series=None, length=11000, agent_num=5000):
        if z_series is None:
            idxz_series, z_series = FiniteMarkov(np.array([self.zg, self.zb]), self.z_trans_mat).simulate(length, self.zg)
        else:
            idxz_series, z_series = z_series
        policy_func = [[interp2d(self.agg_k_grids, self.k_grids, self.policy_mat[:, 0, 0, :].transpose()),
                        interp2d(self.agg_k_grids, self.k_grids, self.policy_mat[:, 0, 1, :].transpose())],
                       [interp2d(self.agg_k_grids, self.k_grids, self.policy_mat[:, 1, 0, :].transpose()),
                        interp2d(self.agg_k_grids, self.k_grids, self.policy_mat[:, 1, 1, :].transpose())]]   # e, z

        # initialize
        kvec = np.zeros((length + 1, agent_num))    # T, N
        # uniform distribution
        kvec[0] = np.random.random(agent_num) * (np.max(self.k_grids) - np.min(self.k_grids)) + np.min(self.k_grids)
        evec = np.zeros((length + 1, agent_num), dtype=int)
        evec[0][np.random.random(agent_num) < self.u_good] = 1     # 1 = unemployed, 0 = employed, state index
        agg_k = np.zeros(length + 1)
        agg_l = self.agg_lg * (1 - idxz_series) + self.agg_lb * idxz_series

        for t in range(length):
            agg_k[t] = np.mean(kvec[t])
            kvec[t + 1] = (1 - evec[t]) * policy_func[0][idxz_series[t]](agg_k[t], kvec[t]).ravel() + \
                          evec[t] * policy_func[1][idxz_series[t]](agg_k[t], kvec[t]).ravel()
            prob = self.e_trans_mat[idxz_series[t], idxz_series[t + 1], :, 0]
            ran_vec = np.random.random(agent_num)
            thres = evec[t] * prob[1] + (1 - evec[t]) * prob[0]
            evec[t + 1] = np.where(ran_vec < thres, 0, 1)

        agg_k[-1] = np.mean(kvec[-1])

        rvec = self.alpha * z_series * (agg_l / agg_k) ** (1 - self.alpha)
        wvec = (1 - self.alpha) * z_series * (agg_k / agg_l) ** self.alpha
        yvec = z_series * agg_k ** self.alpha * agg_l ** (1 - self.alpha)
        ivec = agg_k[1:] - (1 - self.delta) * agg_k[:-1]
        agg_c = yvec[:-1] - ivec
        income_vec = (rvec - self.delta) * kvec.transpose() + wvec * self.h * (1 - evec.transpose())  # N, T
        income_vec = income_vec.transpose()

        return agg_k[:-1], yvec[:-1], agg_c, ivec, income_vec[:-1], kvec[:-1], idxz_series, z_series

    def forecast_aggk(self, k_init, idxz_series, periods=1):
        if self.k_motion is None:
            return None

        if periods == 1 and len(idxz_series.shape) == 1:
            idxz_series = idxz_series.reshape((1, -1))

        k = np.log(k_init).reshape((-1, 1))
        for t in range(periods):
            k = idxz_series[t] * self.k_motion[1].eval(k.reshape(-1, 1)) + (1 - idxz_series[t]) * self.k_motion[0].eval(k.reshape(-1, 1))
        return np.exp(k)