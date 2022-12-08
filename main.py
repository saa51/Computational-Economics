from OptimalGrowth import OptimalGrowthEnv, compute_moments
from FuncApproxEnv import FuncApproxEnv
from GrowthWithHabitIACEnv import GrowthWithHabitIACEnv
import matplotlib.pyplot as plt
from itertools import product
from RandomProcess import AR1Process
from MarkovApprox import markov_moments
from utilize import write_markdown_table
import numpy as np


def question_1():
    # Question 1
    growth = OptimalGrowthEnv({'tol': 1e-5})
    growth.print_steady_state()

    # Question 1(b)
    growth.grid_search(15, growth.k_ss * 0.5, 7, 0.01, n_threads=8)
    growth.plot_policy(True)
    growth.plot_value(True)
    growth.plot_consumption(True)

    # Question 1(c)
    growth.euler_method(15, growth.k_ss * 0.5, 7)
    growth.plot_policy(True)
    growth.plot_value(True)
    growth.plot_consumption(True)

    # Question 1(d)
    methods = ['T', 'TH', 'R']
    grid_nums = [5, 10, 25]
    a_series = None
    moments = []
    for method, grid_num in product(methods, grid_nums):
        print('Euler method:', method, grid_num)
        growth.euler_method(25, growth.k_ss * 0.95, grid_num, method)
        print('Simulating...')
        simu_data = growth.simulation(a_series=a_series)

        plot = False
        if plot:
            plt.plot(simu_data[0], label='a')
            plt.plot(simu_data[1] / growth.k_ss, label='k')
            plt.plot(simu_data[2] / growth.c_ss, label='c')
            plt.plot(simu_data[3] / growth.k_ss / growth.delta, label='i')
            plt.plot(simu_data[4] / growth.k_ss ** growth.alpha, label='y')
            plt.legend()
            plt.show()

        simu_mom = compute_moments(simu_data[0], simu_data[1], simu_data[2], simu_data[3], simu_data[4])
        a_series = simu_data[0]
        mar_mom = markov_moments(growth.a_grids, growth.trans_mat)
        data = mar_mom + simu_mom
        moments.append(data)

    moments = (np.array(moments) / np.array(moments[-1])).transpose().tolist()
    titles = [m + '(' + str(n) + ')' for m, n in product(methods, grid_nums)]
    indexs = ['rho', 'sigma_epsilon', 'sigma_a', 'sigma_k', 'sigma_ak', 'sigma_y', 'sigma_c', 'sigma_i', 'rho_y']
    print(write_markdown_table(moments, title=titles, index=indexs, align='c'))


def question_2():
    func_approx = FuncApproxEnv()
    func_approx.validate(5)
    func_approx.validate(10)


def question_3():
    growth = GrowthWithHabitIACEnv()
    growth.solve_steady_state()
    weights = growth.pea(1e-4, buried=500, order=1, max_iter=2000)
    # growth.pea(1e-4, buried=500, order=2, init_weight=weights)
    # growth.pea(1e-4, order=2)
    '''
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()
    '''
    growth = GrowthWithHabitIACEnv({'b':0.1})
    growth.solve_steady_state()
    weights = growth.pea(1e-4, buried=500, order=1, max_iter=2000)
    # growth.pea(1e-4, buried=500, order=2, init_weight=weights)
    # growth.pea(1e-4, order=2)
    '''
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()
    '''
    growth = GrowthWithHabitIACEnv({'phi':0.03})
    growth.solve_steady_state()
    weights = growth.pea(1e-4, buried=500, order=1, max_iter=2000)
    # growth.pea(1e-4, buried=500, order=2, init_weight=weights)
    # growth.pea(1e-4, order=2)
    '''
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()
    '''
    growth = GrowthWithHabitIACEnv({'phi':0.03, 'b':0.1})
    growth.solve_steady_state()
    weights = growth.pea(1e-4, buried=500, order=1, max_iter=2000)
    # growth.pea(1e-4, buried=500, order=2, init_weight=weights)
    # growth.pea(1e-4, order=2)
    '''
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()
    '''
    growth = GrowthWithHabitIACEnv()
    growth.solve_steady_state()
    weights = growth.pea(1e-4, buried=500, order=1, max_iter=2000, iri=True)
    # growth.pea(1e-4, buried=500, order=2, init_weight=weights)
    # growth.pea(1e-4, order=2)

if __name__ == '__main__':
    question_1()
    question_2()
    question_3()
