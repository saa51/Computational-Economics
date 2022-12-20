import numpy as np


class MarkovProcess:
    def simulate(self, length, init):
        pass


class AR1Process(MarkovProcess):
    def __init__(self, rho, error_type='normal', err_params=None):
        self.rho = rho
        self.err = error_type
        self.err_params = err_params if err_params is not None else {'sigma': 1}

    def simulate(self, length, init):
        simulation = np.zeros(length + 1)
        simulation[0] = init
        for t in range(1, length + 1):
            simulation[t] = simulation[t - 1] * self.rho + np.random.normal(0, self.err_params['sigma'])
        return simulation


class FiniteMarkov(MarkovProcess):
    def __init__(self, grids, trans_mat):
        self.grids = grids
        self.trans_mat = trans_mat

    def simulate(self, length, init):
        simulation_idx = np.zeros(length + 1, dtype=int)
        simulation_idx[0] = np.argmin((self.grids - init) ** 2)
        rands = np.random.random(length + 1)
        trans_cdf = np.cumsum(self.trans_mat, axis=1)
        for t in range(1, length + 1):
            simulation_idx[t] = np.searchsorted(trans_cdf[simulation_idx[t - 1]], rands[t])
        return simulation_idx, self.grids[simulation_idx]


if __name__ == '__main__':
    mc = FiniteMarkov(np.array([-1, 0, 1]), np.array([[0.1, 0.5, 0.4], [0.4, 0.1, 0.5], [0.5, 0.4, 0.1]]))
    print(mc.simulate(20, 0))
