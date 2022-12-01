from OptimalGrowth import OptimalGrowthEnv
from FuncApproxEnv import FuncApproxEnv
from GrowthWithHabitIACEnv import GrowthWithHabitIACEnv
import matplotlib.pyplot as plt


def question_1():
    # Question 1
    growth = OptimalGrowthEnv({'tol': 1e-6, 'beta': 0.8})
    growth.print_steady_state()
    '''
    # Question 1(b)
    growth.grid_search(15, 10, 7, 0.1)
    growth.plot_policy()
    growth.plot_value()
    '''
    growth.euler_method(15, 0.5, 7)
    growth.plot_policy()
    growth.plot_value()


def question_2():
    func_approx = FuncApproxEnv()
    func_approx.validate(5)
    func_approx.validate(10)


def question_3():
    growth = GrowthWithHabitIACEnv()
    growth.solve_steady_state()
    #growth.pea(220, 280, 1e-5, buried=500, order=2)
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()

    growth = GrowthWithHabitIACEnv({'b':0.1})
    growth.solve_steady_state()
    #growth.pea(220, 280, 1e-5, buried=500, order=2)
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()

    growth = GrowthWithHabitIACEnv({'phi':0.03})
    growth.solve_steady_state()
    #growth.pea(220, 280, 1e-5, buried=500, order=2)
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()

    growth = GrowthWithHabitIACEnv({'phi':0.03, 'b':0.1})
    growth.solve_steady_state()
    #growth.pea(220, 280, 1e-5, buried=500, order=2)
    growth.value_func_approx(0.1*growth.k_ss, 0.4*growth.c_ss, 21, 21)
    growth.plot_policy()
    growth.plot_value()


if __name__ == '__main__':
    #question_1()
    #question_2()
    question_3()