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
