import numpy as np
from numpy.linalg import inv
from scipy.special import comb
from sklearn.metrics import r2_score


class FunctionApprox:
    pass


class ChebyshevApprox2D(FunctionApprox):
    def __init__(self, n1, n2, mean1, mean2, width1, width2, grid1=None, grid2=None):
        self.n1 = n1
        self.n2 = n2
        self.mean1 = mean1
        self.mean2 = mean2
        self.width1 = width1
        self.width2 = width2
        self.grid1 = (grid1 - mean1) / width1 if grid1 is not None else np.cos(np.pi * np.arange(1, 2 * n1, 2) / (2 * n1))
        self.grid2 = (grid2 - mean2) / width2 if grid2 is not None else np.cos(np.pi * np.arange(1, 2 * n2, 2) / (2 * n2))
        phi_mat = []
        for i in range(n1):
            for j in range(n2):
                row_vec = np.cos(j * np.arccos(self.grid2)).reshape((1, n2))
                col_vec = np.cos(i * np.arccos(self.grid1)).reshape((n1, 1))
                func_value = np.tensordot(col_vec, row_vec, 1)
                phi_mat.append(func_value.ravel())
        self.phi_mat = np.array(phi_mat)
        self.weights = np.zeros((1, n1 * n2))

    def real_grids(self):
        return self.grid1 * self.width1 + self.mean1, self.grid2 * self.width2 + self.mean2

    def eval(self, x, y):
        x = (x - self.mean1) / self.width1
        y = (y - self.mean2) / self.width2
        row_vec = np.cos(np.arange(0, self.n2) * np.arccos(y)).reshape((1, self.n2))
        col_vec = np.cos(np.arange(0, self.n1) * np.arccos(x)).reshape((self.n1, 1))
        values = np.tensordot(col_vec, row_vec, 1).reshape((-1, 1))
        return np.dot(self.weights, values).ravel()[0]

    def approx(self, values, lr=1, update=True):
        values = np.reshape(values, (1, -1))
        new_weights = np.dot(values, inv(self.phi_mat))
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights


class PolinomialApprox(FunctionApprox):
    def __init__(self, deg):
        self.deg = deg
        self.weights = np.zeros(deg)

    def eval(self, x):
        y = np.zeros(shape=x.shape)
        for d in range(self.deg):
            y = y * x + self.weights[d]
        return y

    def approx(self, grids, values, lr=1, update=True):
        basis_values = [np.ones(shape=grids.shape)]
        for d in range(self.deg - 1):
            basis_values.append(basis_values[-1] * grids)
        basis_values.reverse()
        basis_values = np.stack(basis_values, axis=1)
        new_weights = np.dot(inv(basis_values), values.reshape((-1, 1))).ravel()
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights


class SplineApprox(FunctionApprox):
    def __init__(self):
        self.grids = None
        self.values = None

    def eval(self, x):
        idx_h = np.searchsorted(self.grids[:-1], x, 'left')
        idx_l = idx_h - 1
        return (self.values[idx_h] - self.values[idx_l]) / (self.grids[idx_h] - self.grids[idx_l]) * (x - self.grids[idx_l]) + self.values[idx_l]

    def approx(self, grids, values, lr=1):
        idx = np.argsort(grids)
        self.grids = grids[idx]
        values = values[idx]
        if self.values is None:
            self.values = np.zeros(shape=values.shape)
        self.values = lr * values + (1 - lr) * self.values


class ChebyshevApprox(FunctionApprox):
    def __init__(self, mean, width, n):
        self.mean = mean
        self.width = width
        self.deg = n
        self.weights = np.zeros(n)
        self.default_grids = np.cos(np.pi * np.arange(1, 2 * n, 2) / (2 * n)) * width + mean

    def eval(self, x):
        x_g = (x - self.mean) / self.width
        basis_values = np.cos(np.kron(np.arange(0, self.deg).reshape((-1, 1)), np.arccos(x_g).reshape((1, -1))))
        y = np.dot(basis_values.transpose(), self.weights.reshape((-1, 1)))
        return y.reshape(x.shape)

    def approx(self, values, grids=None, lr=1, update=True):
        if grids is None:
            grids = self.default_grids
        g_g = (grids - self.mean) / self.width
        basis_values = np.cos(np.kron(np.arange(0, self.deg).reshape((-1, 1)), np.arccos(g_g).reshape((1, -1))))
        values = np.reshape(values, (1, -1))
        new_weights = np.dot(values, inv(basis_values))
        new_weights = new_weights.ravel()
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights


class LinearApprox(FunctionApprox):
    def __init__(self, var_num, order=1, constant=True):
        self.v_num = var_num
        self.order = order
        self.cons = constant

        self.param_num = int(self.cons)
        for d in range(1, order + 1):
            self.param_num += int(comb(d + self.v_num - 1, self.v_num - 1))
        self.weights = np.zeros(self.param_num)

    def regressor_construction(self, x):
        regressors = np.copy(x)
        regressors = np.reshape(regressors, (-1, self.v_num))
        idx = np.arange(self.v_num)
        for d in range(2, self.order + 1):
            end = regressors.shape[1]
            for v in range(self.v_num):
                start = regressors.shape[0]
                regressors = np.concatenate((regressors, regressors[:, idx[v]:end] * x[:, v].reshape((-1, 1))), axis=1)
                idx[v] = start
        if self.cons:
            regressors = np.concatenate((np.ones((regressors.shape[0], 1)), regressors), axis=1)
        return regressors

    def eval(self, x):
        shape_x = x.shape[:-1]
        regressors = self.regressor_construction(x)
        return np.dot(regressors, self.weights.reshape((-1, 1))).reshape(shape_x)

    def approx(self, x, y, lr=1, update=True):
        regressors = self.regressor_construction(x)
        dependent = np.copy(y).reshape((-1, 1))
        new_weights = np.dot(inv(np.dot(regressors.transpose(), regressors)), np.dot(regressors.transpose(), dependent))
        new_weights = new_weights.ravel()
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        prediction = np.dot(regressors, new_weights.reshape((-1, 1))).ravel()
        #r_square = r2_score(dependent.ravel(), prediction)
        z = dependent.ravel() - prediction
        u = np.sum(z * z) / len(z)
        #print(self.weights[0], np.mean(z))
        #print(np.linalg.norm(inv(np.dot(regressors.transpose(), regressors))))
        r_square = 1 - u / np.var(dependent)
        return new_weights, r_square


class ExpLogLinearApprox(FunctionApprox):
    def __init__(self, var_num, order=1, constant=True):
        self.linear_approx = LinearApprox(var_num, order, constant)

    def eval(self, x):
        return np.exp(self.linear_approx.eval(np.log(x)))

    def approx(self, x, y, lr=1, update=True):
        return self.linear_approx.approx(np.log(x), np.log(y), lr, update)

    def get_weights(self):
        return self.linear_approx.weights


if __name__ == '__main__':
    x = np.random.normal(0, 1, 1000)
    e = np.random.normal(0, 1, 1000)
    beta_0 = 1
    beta_1 = 1
    #y = beta_0 + beta_1 * x + e
    y=e+beta_0
    approx = ExpLogLinearApprox(1,constant=False)
    print(approx.approx(np.exp(x).reshape((-1, 1)), np.exp(y)))
