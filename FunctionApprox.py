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


class PolynomialApprox(FunctionApprox):
    def __init__(self, deg):
        self.deg = deg
        self.weights = np.zeros(deg)

    def eval(self, x):
        z = np.array(x)
        ans = np.dot(z.reshape((-1, 1)) ** np.arange(self.deg), self.weights.reshape(-1, 1))
        if hasattr(x, 'shape'):
            ans = ans.reshape(x.shape)
        return ans

    def approx(self, grids, values, lr=1, update=True):
        basis_values = grids.reshape((-1, 1)) ** np.arange(self.deg)
        new_weights = np.dot(inv(np.dot(basis_values.transpose(),basis_values)), np.dot(basis_values.transpose(), values.reshape((-1, 1)))).ravel()
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights

    def gradient(self, x):
        ans = np.dot(x.reshape((-1, 1)) ** np.arange(self.deg), (np.arange(self.deg) * self.weights).reshape(-1, 1))
        if hasattr(x, 'shape'):
            ans = ans.reshape(x.shape)
        return ans


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

    def eval(self, x):
        x_g = (x - self.mean) / self.width
        basis_values = np.cos(np.kron(np.arange(0, self.deg).reshape((-1, 1)), np.arccos(x_g).reshape((1, -1))))
        y = np.dot(basis_values.transpose(), self.weights.reshape((-1, 1)))
        if hasattr(x, 'shape'):
            y = y.reshape(x.shape)
        return y

    def approx(self, values, grids=None, lr=1, update=True):
        if grids is None:
            grids = self.default_grids
        g_g = (grids - self.mean) / self.width
        basis_values = np.cos(np.kron(np.arccos(g_g).reshape((-1, 1)), np.arange(0, self.deg).reshape((1, -1))))
        values = np.reshape(values, (-1, 1))
        new_weights = np.dot(inv(np.dot(basis_values.transpose(), basis_values)), np.dot(basis_values.transpose(), values))
        new_weights = new_weights.ravel()
        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights
        return new_weights

    def gradient(self, x):
        x_g = (x - self.mean) / self.width
        def gradient_n(x):
            if not hasattr(gradient_n, 'n'):
                gradient_n.n = 0
            return gradient_n.n * np.sin(gradient_n.n * np.arccos(x)) / np.sin(np.arccos(x))

        gradients = []
        for n in range(self.deg):
            gradient_n.n = n
            gradients.append(np.piecewise(x_g, [x_g <= -1, np.logical_and(x_g > -1, x_g < 1), x_g >= 1], [-(-1) ** n * n * n, gradient_n, n * n]))
        return np.dot(self.weights.reshape(1, -1), np.array(gradients)).squeeze() / self.width


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

        prediction = np.dot(regressors, new_weights.reshape((-1, 1))).ravel()
        #r_square = r2_score(dependent.ravel(), prediction)
        z = dependent.ravel() - prediction
        u = np.sum(z * z) / len(z)
        #print(self.weights[0], np.mean(z))
        #print(np.linalg.norm(inv(np.dot(regressors.transpose(), regressors))))
        r_square = 1 - u / np.var(dependent)

        new_weights = lr * new_weights + (1 - lr) * self.weights
        if update:
            self.weights = new_weights

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
