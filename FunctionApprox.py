import numpy as np
from numpy.linalg import inv
from scipy.special import comb
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

class FunctionApprox:
    def __init__(self, **kwargs) -> None:
        pass

    def __call__(self, x):
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
        value1 = np.cos(np.expand_dims(np.arccos(self.grid1), axis=-1) * np.arange(n1)).reshape((n1, 1, n1, 1))
        value2 = np.cos(np.expand_dims(np.arccos(self.grid2), axis=-1) * np.arange(n2)).reshape((1, n2, 1, n2))
        value1 = np.tile(value1, (1, n2, 1, n2))
        value2 = np.tile(value2, (n1, 1, n1, 1))
        value = value1 * value2
        self.phi_mat = value.reshape((n1 * n2, n1 * n2))
        self.weights = np.zeros(n1 * n2)

    def real_grids(self):
        return self.grid1 * self.width1 + self.mean1, self.grid2 * self.width2 + self.mean2

    def __call__(self, x, y):
        x = (x - self.mean1) / self.width1
        y = (y - self.mean2) / self.width2
        '''print(x)
        print(y)'''
        value1 = np.cos(np.expand_dims(np.arccos(x), axis=-1) * np.arange(self.n1)).reshape(x.shape + (1, self.n1, 1))
        value2 = np.cos(np.expand_dims(np.arccos(y), axis=-1) * np.arange(self.n2)).reshape((1,) + y.shape + (1, self.n2))
        value1 = np.tile(value1, tuple([1] * len(x.shape)) + y.shape + (1, self.n2))
        value2 = np.tile(value2, x.shape + tuple([1] * len(y.shape)) + (self.n1, 1))
        values = (value1 * value2).reshape(x.shape + y.shape + (-1,))
        return values @ self.weights

    def approx(self, values, lr=1, update=True):
        new_weights = inv(self.phi_mat) @ values.reshape((-1,))
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights


class PolynomialApprox(FunctionApprox):
    def __init__(self, deg):
        self.deg = deg
        self.weights = np.zeros(self.deg)

    def __call__(self, x):
        x = np.array(x)
        ans = np.expand_dims(x, axis=-1) ** np.arange(self.deg) * self.weights
        return ans.sum(axis=-1)

    def approx(self, grids, values, lr=1, update=True):
        basis_values = grids.reshape((-1, 1)) ** np.arange(self.deg)
        new_weights = inv(basis_values.T @ basis_values) @ basis_values.T @ values.reshape((-1,))
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights

    def gradient(self, x):
        x = np.array(x)
        ans = np.expand_dims(x, axis=-1) ** np.arange(self.deg - 1) @ (np.arange(1, self.deg) * self.weights[1:])
        return ans


class SplineApprox(FunctionApprox):
    def __init__(self):
        self.grids = None
        self.values = None

    def __call__(self, x):
        idx_h = np.searchsorted(self.grids[:-1], x, 'left')
        idx_l = idx_h - 1
        return (self.values[idx_h] - self.values[idx_l]) / (self.grids[idx_h] - self.grids[idx_l]) * (x - self.grids[idx_l]) + self.values[idx_l]

    def approx(self, grids, values, lr=1):
        idx = np.argsort(grids)
        self.grids = np.array(grids)[idx]
        values = np.array(values)[idx]
        if self.values is None:
            self.values = np.zeros(shape=values.shape)
        self.values = lr * values + (1 - lr) * self.values


class ChebyshevApprox(FunctionApprox):
    def __init__(self, mean, width, n, grids=None):
        self.mean = mean
        self.width = width
        self.deg = n
        self.weights = np.zeros(n)
        if grids is None:
            self.default_grids = np.cos(np.pi * np.arange(1, 2 * n, 2) / (2 * n)) * width + mean
        else:
            self.default_grids = grids

    def __call__(self, x):
        x_g = (x - self.mean) / self.width
        basis_values = np.cos(np.expand_dims(np.arccos(x_g), axis=-1) * np.arange(self.deg))
        y = basis_values @ self.weights
        return y

    def approx(self, values, grids=None, lr=1, update=True):
        if grids is None:
            grids = self.default_grids
        g_g = (grids - self.mean) / self.width
        basis_values = np.cos(np.expand_dims(np.arccos(g_g), axis=-1) * np.arange(self.deg))
        new_weights = inv(basis_values.T @ basis_values) @ basis_values.T @ values
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights

    def gradient(self, x):
        x_g = (x - self.mean) / self.width
        basis_values = np.sin(np.expand_dims(np.arccos(x_g), axis=-1) * np.arange(self.deg - 1)) / \
                        np.sin(np.expand_dims(np.arccos(x_g), axis=-1) * np.ones(self.deg - 1))
        basis_values *= np.arange(1, self.deg)
        return basis_values @ self.weights / self.width


class LinearApprox(FunctionApprox):
    def __init__(self, var_num, order=1, constant=True):
        self.v_num = var_num
        self.order = order + 1
        self.cons = constant

        self.param_num = int(self.cons) 
        self.param_num += int(comb(np.arange(order) + self.v_num, (self.v_num - 1) * np.ones(order)).sum())
        self.weights = np.zeros(self.param_num)

        self.feature_transform = PolynomialFeatures(order, include_bias=constant)
        self.feature_transform.fit(np.arange(var_num).reshape(1, -1))

    def __call__(self, x):
        regressors = self.feature_transform.transform(x)
        return regressors @ self.weights

    def approx(self, x, y, lr=1, update=True):
        regressors = self.feature_transform.transform(x)
        dependent = np.copy(y).reshape((-1, 1))
        new_weights = inv(regressors.T @ regressors) @ regressors.T @ dependent
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights


class ExpLogLinearApprox(FunctionApprox):
    def __init__(self, var_num, order=1, constant=True):
        self.linear_approx = LinearApprox(var_num, order, constant)

    def __call__(self, x):
        return np.exp(self.linear_approx(np.log(x)))

    def approx(self, x, y, lr=1, update=True):
        return self.linear_approx.approx(np.log(x), np.log(y), lr, update)

    def get_weights(self):
        return self.linear_approx.weights


if __name__ == '__main__':
    # test 1
    '''
    x = np.random.normal(0, 1, 1000)
    e = np.random.normal(0, 1, 1000)
    beta_0 = 1
    beta_1 = 1
    #y = beta_0 + beta_1 * x + e
    y=e+beta_0
    approx = ExpLogLinearApprox(1,constant=False)
    print(approx.approx(np.exp(x).reshape((-1, 1)), np.exp(y)))
    '''

    x = np.linspace(0, 1, 1000)
    e = np.random.normal(0, 1, 1000)
    y = np.sqrt(x + 1)
    approx = ChebyshevApprox(0.5, 0.5, 5)
    print(approx.approx(y, x))
    y_hat = approx.eval(x)
    import matplotlib.pyplot as plt
    #plt.plot(x, y, label='real')
    #plt.plot(x, y_hat, label='approx')
    plt.plot(x, approx.gradient(x), label='computed_g')
    plt.plot(x[:-1]+0.0005, (y_hat[1:] - y_hat[:-1]) * 1000, label='cut')
    plt.plot(x, 0.5 / y, label='real_g')
    plt.legend()
    plt.show()
