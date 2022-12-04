import numpy as np
from scipy.stats import norm
from numpy.polynomial import hermite

epsilon = 1e-8


class MarkovApprox:
    def __init__(self, rho, var_e, mean_z):
        self.rho = rho
        self.var_e = var_e
        self.mean_z = mean_z
        self.var_z = var_e / (1 - rho ** 2)
        self.sigma_e = np.sqrt(self.var_e)
        self.sigma_z = np.sqrt(self.var_z)
        self.grids = None
        self.trans_mat = None


class Rowenhorst(MarkovApprox):
    def approx(self, n):
        p = (1 + self.rho) / 2
        nu = np.sqrt((n - 1) * self.var_z)
        trans_mat = np.array([[p, 1 - p], [1 - p, p]])
        for i in range(3, n + 1):
            trans_zero = np.concatenate((trans_mat, np.zeros((i - 1, 1))), axis=1)
            zero_trans = np.concatenate((np.zeros((i - 1, 1)), trans_mat), axis=1)
            upper_left = np.concatenate((trans_zero, np.zeros((1, i))), axis=0)
            upper_right = np.concatenate((zero_trans, np.zeros((1, i))), axis=0)
            lower_left = np.concatenate((np.zeros((1, i)), trans_zero), axis=0)
            lower_right = np.concatenate((np.zeros((1, i)), zero_trans), axis=0)
            trans_mat = p * (upper_left + lower_right) + (1 - p) * (upper_right + lower_left)
            normalize_mat = np.eye(i) / 2
            normalize_mat[0][0] = 1
            normalize_mat[i - 1][i - 1] = 1
            trans_mat = np.dot(normalize_mat, trans_mat)
        self.grids = np.linspace(self.mean_z - nu, self.mean_z + nu, n, endpoint=True)
        self.trans_mat = trans_mat
        return self.grids, trans_mat


class Tauchen(MarkovApprox):
    def approx(self, n, m=None):
        __tauchen_omega_dict = {
            5: 1.6425,
            25: 2.5107,
            10: 1.9847
        }
        if m is None:
            m = __tauchen_omega_dict.get(n, 3)
        grids = np.linspace(self.mean_z - m * self.sigma_z, self.mean_z + m * self.sigma_z, n, endpoint=True)
        w = 2 * m * self.sigma_z / (n - 1)
        trans_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                upper = norm.cdf((grids[j] + w / 2 - (1 - self.rho) * self.mean_z - self.rho * grids[i]) / self.sigma_e)
                lower = norm.cdf((grids[j] - w / 2 - (1 - self.rho) * self.mean_z - self.rho * grids[i]) / self.sigma_e)
                if j == 0:
                    trans_mat[i][j] = upper
                elif j == n - 1:
                    trans_mat[i][j] = 1 - lower
                else:
                    trans_mat[i][j] = upper - lower
        self.grids = grids
        self.trans_mat = trans_mat
        return grids, trans_mat


class TauchenHussey(MarkovApprox):
    def approx(self, n):
        y_grids, weights = hermite.hermgauss(n)
        grids = np.sqrt(2) * self.sigma_z * y_grids + self.mean_z
        trans_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                prob_mu = norm.pdf((grids[j] - self.mean_z) / self.sigma_e)
                prob_z = norm.pdf((grids[j] - (1 - self.rho) * self.mean_z - self.rho * grids[i]) / self.sigma_e)
                trans_mat[i][j] = weights[j] / (prob_mu + epsilon) * prob_z + epsilon
        sum_vec = np.sum(trans_mat, axis=1)
        for i in range(n):
            for j in range(n):
                trans_mat[i][j] = trans_mat[i][j] / sum_vec[i]
        self.grids = grids
        self.trans_mat = trans_mat
        return grids, trans_mat


def markov_moments(grids, trans_mat):
    dist = np.ones(grids.size) / grids.size
    old_dist = np.zeros(grids.size)
    while not np.allclose(dist, old_dist):
        old_dist = np.copy(dist)
        dist = np.dot(dist.reshape(1, -1), trans_mat).ravel()
    mean_a = np.sum(dist * grids)
    sigma_a = np.sqrt(np.sum(grids * grids * dist) - mean_a * mean_a)

    a_aprime = np.kron(grids.reshape((-1, 1)), grids.reshape((1, -1)))
    rho = np.sum(a_aprime * (trans_mat * dist)) / np.sum(grids * grids * dist)

    epsilon = -rho * grids.reshape((-1, 1)) + grids.reshape((1, -1))
    sigma_e = np.sqrt(np.sum(epsilon * epsilon * (trans_mat.transpose() * dist).transpose()))
    return [rho, sigma_e, sigma_a]


if __name__ == '__main__':
    rho = 0.95
    mean_z = 1
    var_e = 1
    grids_num = 5

    print('Rowenhorst')
    print(Rowenhorst(rho, var_e, mean_z).approx(grids_num))
    print('Tauchen')
    print(Tauchen(rho, var_e, mean_z).approx(grids_num, 3))
    print('TauchenHussey')
    print(TauchenHussey(rho, var_e, mean_z).approx(grids_num))
