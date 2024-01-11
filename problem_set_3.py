import numpy as np
import os
from Huggett1996Env import Huggett1996Env
import matplotlib.pyplot as plt
from itertools import product

data_path = './data/'
figure_path = './figures/problem_set_3/'
if __name__ == '__main__':
    # load data
    params = {}
    data_names = ['zvec', 'yvec', 'pimat', 'z1probvec']
    for data in data_names:
        params[data] = np.loadtxt(os.path.join(data_path, data + '.txt'), delimiter='\t')

    env = Huggett1996Env(**params)
    env.solve_stationary_equilibrium(200, 500, r_init=0.07, log=True)
    env.plot_policy(figure_path=figure_path)
    env.plot_Lorenz_curve(figure_path)
    env.plot_variance(figure_path)
    print('Gini index:', env.gini_index())
    percentiles = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8]
    for p in percentiles:
        print(f'percentile {p}, {1 - env.lorenz_curve.eval(1 - p)}')

    params['n'] = 0.001
    env2 = Huggett1996Env(**params)
    env2.solve_stationary_equilibrium(200, 500, r_init=0.07, log=True)
    env2.plot_Lorenz_curve()
    print('Gini index:', env2.gini_index())
    for p in percentiles:
        print(f'percentile {p}, {1 - env2.lorenz_curve.eval(1 - p)}')

    lamda = (env2.value_mat / env.value_mat) ** (1 / (1 - 1.5)) - 1
    zidx = [0, 9, 17]
    cidx = [0, 54 - 1, int(79 / 3)]

    for c in cidx:
        for z in zidx:
            plt.plot(env.avec, lamda[c, :, z], label='z={}'.format(z + 1, c + 1))
        plt.legend()
        plt.title('Welfare Difference: Age {}'.format(c + 1))
        plt.savefig(os.path.join(figure_path, 'lamda_{}.pdf'.format(c + 1)))
        plt.show()
