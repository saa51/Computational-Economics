from SOGEnv import SOGENV
import time
import numpy as np
from utilize import write_markdown_table, write_latex_table
import os
import matplotlib.pyplot as plt
import argparse

figure_path = './figures/problem_set_2'
tex_path ='./reports/'
width = 0.2


# Problem g
def process_simulation(a, k, c, i, y, title='', show=False):
    plt.hist(k)
    plt.title('Capital Histogram:' + title)
    plt.savefig(os.path.join(figure_path, title + '_KHist.png'))
    if show:
        plt.show()
    else:
        plt.clf()

    plt.hist(i / k)
    plt.title('Investment Rate Histogram:' + title)
    plt.savefig(os.path.join(figure_path, title + '_IRHist.png'))
    if show:
        plt.show()
    else:
        plt.clf()

    moments = []
    moments.append(np.sqrt(np.var(y)))
    moments.append(np.sqrt(np.var(c)))
    moments.append(np.sqrt(np.var(i)))
    corr = np.corrcoef([y, c, i])
    moments.append(corr[0][1])
    moments.append(corr[0][2])
    moments.append(corr[1][2])
    moments.append(np.mean(k))
    moments.append(np.mean(c))
    moments.append(np.mean(y))
    moments.append(np.mean(i))
    moments.append(np.corrcoef([y[1:], y[:-1]])[0][1])
    moments.append(np.corrcoef([c[1:], c[:-1]])[0][1])
    moments.append(np.corrcoef([i[1:], i[:-1]])[0][1])
    moments.append(np.corrcoef([a[1:], a[:-1]])[0][1])
    return moments


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true', help='whether show figures')
    parser.add_argument('-not_all', action='store_true', help='whether run all problems')
    parser.add_argument('--hpi', action='store_true', help='whether run problem h')
    parser.add_argument('--mqp', action='store_true', help='whether run problem i')
    parser.add_argument('--vfi', action='store_true', help='whether run problem c to g')
    parser.add_argument('--che', action='store_true', help='whether run problem j to l')
    parser.add_argument('--pea', action='store_true', help='whether run problem m')
    parser.add_argument('--end', action='store_true', help='whether run problem n')
    args = parser.parse_args()

    np.random.seed(42)
    moments = []
    tfp = None

    # Problem a: TBD in writing

    # Problem b:
    sog = SOGENV(show=args.show)

    # Problem c:
    if not args.not_all or args.vfi:
        start_time = time.time()
        sog.value_func_iter(200, 5, sog.k_ss * width, 3)
        end_time = time.time()
        print('Problem c: nk = 200, time =', end_time - start_time)
        sog.plot_capital_diff(title='C_VFI', fname=os.path.join(figure_path, 'C_KDiff.png'))
        start_time = time.time()
        sog.value_func_iter(400, 5, sog.k_ss * width, 3)
        end_time = time.time()
        print('Problem c: nk = 400, time =', end_time - start_time)

        value_mat, policy_mat = sog.value_mat, sog.policy_mat
        
        # Problem d:
        sog.plot_value(title='D_VFI', fname=os.path.join(figure_path, 'D_ValueFunc.png'))
        sog.plot_value_derivative(title='D_VFI', fname=os.path.join(figure_path, 'D_ValueFuncD.png'))
        sog.plot_value_2derivative(title='D_VFI', fname=os.path.join(figure_path, 'D_ValueFunc2D.png'))

        # Problem e:
        sog.plot_policy(title='E_VFI', fname=os.path.join(figure_path, 'E_PolicyFunc.png'))
        sog.plot_policy_derivative(title='E_VFI', fname=os.path.join(figure_path, 'E_PolicyFuncD.png'))
        sog.plot_policy_2derivative(title='E_VFI', fname=os.path.join(figure_path, 'E_PolicyFunc2D.png'))
        sog.plot_capital_diff(title='E_VFI', fname=os.path.join(figure_path, 'E_KDiff.png'))

        # Problem f:
        errs = sog.plot_euler_err(title='F_VFI', fname=os.path.join(figure_path, 'F_EulerErr.png'))
        titles = ['mean', 'min', 'max']
        index = [str(round(a, 2)) for a in sog.a_grids]
        content = [[round(float(np.mean(err)), 3), round(np.min(err), 3), round(np.max(err), 3)] for err in errs]
        with open(os.path.join(tex_path, 'ps2_f.tex'), 'w+') as f:
            print(write_latex_table(content, title=titles, index=index, align='c'), file=f)
        print(write_markdown_table(content, title=titles, index=index, align='c'))

        # Problem g:
        a, k, c, i, y, tfp = sog.simulate(2500, tfp_series=None)
        a, k, c, i, y = a[-1000:], k[-1000:], c[-1000:], i[-1000:], y[-1000:]
        print('Are all k_t != k_max or k_min?',
            np.all(np.logical_and(np.min(sog.k_grids) + 1e-6 < k, k < np.max(sog.k_grids) - 1e-6)))
        moments.append(process_simulation(a, k, c, i, y, title='G_VFI', show=args.show))

    # Problem h:
    if args.hpi or not args.not_all:
        optimize_dict = {
            'methods': ['HPI'],
            'HPI_policy_iter': 5,
            'HPI_value_iter': 20
        }
        start_time = time.time()
        sog.value_func_iter(400, 5, sog.k_ss * width, 3, **optimize_dict)
        end_time = time.time()
        print('Problem h: Howard Policy Iteration, nh = 5, nv = 20, time =', end_time - start_time)

        optimize_dict = {
            'methods': ['HPI'],
            'HPI_policy_iter': 20,
            'HPI_value_iter': 10
        }
        start_time = time.time()
        sog.value_func_iter(400, 5, sog.k_ss * width, 3, **optimize_dict)
        end_time = time.time()
        print('Problem h: Howard Policy Iteration, nh = 20, nv = 10, time =', end_time - start_time)
        sog.plot_policy(title='H_HPI', fname=os.path.join(figure_path, 'H_PolicyFunc.png'))

    # Problem i:
    if args.mqp or not args.not_all:
        optimize_dict = {
            'methods': ['MQP']
        }
        start_time = time.time()
        sog.value_func_iter(400, 5, sog.k_ss * width, 3, **optimize_dict)
        end_time = time.time()
        print('Problem i: McQueen Perteus Error Bound, time =', end_time - start_time)

    # Problem j
    if args.che or not args.not_all:
        optimize_dict = {
            'methods': ['HPI', 'MQP', 'parallel'],
            'HPI_policy_iter': 20,
            'HPI_value_iter': 10,
            'n_jobs': 6
        }
        start_time = time.time()
        sog.value_approx(400, 5, sog.k_ss * width, 7, 3, **optimize_dict)
        end_time = time.time()
        print('Problem j: Chebyshev Approximation, time =', end_time - start_time)
        sog.plot_value(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_ValueFunc.png'))
        sog.plot_value_derivative(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_ValueFuncD.png'))
        sog.plot_value_2derivative(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_ValueFunc2D.png'))
        sog.plot_policy(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_PolicyFunc.png'))
        sog.plot_policy_derivative(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_PolicyFuncD.png'))
        sog.plot_policy_2derivative(title='J_Chebyshev', fname=os.path.join(figure_path, 'J_PolicyFunc2D.png'))

        # Problem k
        errs = sog.plot_euler_err(title='K_Chebyshev', fname=os.path.join(figure_path, 'K_EulerErr.png'))
        titles = ['mean', 'min', 'max']
        index = [str(round(a, 2)) for a in sog.a_grids]
        content = [[round(float(np.mean(err)), 3), round(np.min(err), 3), round(np.max(err), 3)] for err in errs]
        with open(os.path.join(tex_path, 'ps2_k.tex'), 'w+') as f:
            print(write_latex_table(content, title=titles, index=index, align='c'), file=f)
        print(write_markdown_table(content, title=titles, index=index, align='c'))

        # Problem l
        a, k, c, i, y, _ = sog.simulate(2500, tfp_series=tfp)
        a, k, c, i, y = a[-1000:], k[-1000:], c[-1000:], i[-1000:], y[-1000:]
        moments.append(process_simulation(a, k, c, i, y, title='L_CHE', show=args.show))

    # Problem m
    if args.pea or not args.not_all:
        sog.modified_pea(400, 5, sog.k_ss * width, 7, 3)
        a, k, c, i, y, _ = sog.simulate(2500, tfp_series=tfp)
        a, k, c, i, y = a[-1000:], k[-1000:], c[-1000:], i[-1000:], y[-1000:]
        moments.append(process_simulation(a, k, c, i, y, title='M_PEA', show=args.show))

    # Problem n
    if args.end or not args.not_all:
        #sog.endo_grid(400, 5, sog.k_ss * width, 7, 3, init=(value_mat, policy_mat))
        sog.endo_grid(400, 5, sog.k_ss * width, 7, 3)
        a, k, c, i, y, _ = sog.simulate(2500, tfp_series=tfp)
        a, k, c, i, y = a[-1000:], k[-1000:], c[-1000:], i[-1000:], y[-1000:]
        moments.append(process_simulation(a, k, c, i, y, title='N_EGM', show=args.show))
    
    if not args.not_all:
        titles = ['VFI', 'Chebyshev Approximation', 'modified PEA', 'EGM', 'Real']
        index = ['sigma y', 'sigma c', 'sigma i', 'corr(y, c)', 'corr(y, i)', 'corr(c, i)', 'mean(k)', 'mean(c)', 'mean(y)',
                'mean(i)', 'autocorr(y)', 'autocorr(c)', 'autocorr(i)', 'autocorr(a)']
        moments.append([1.8, 1.3, 5.1, 0.739, 0.714, -100, -100, -100, -100, -100, -100, -100, -100, -100])
        moments = np.array(moments).transpose()
        moments = np.around(moments, 4).tolist()
        with open(os.path.join(tex_path, 'ps2_n.tex'), 'w+') as f:
            print(write_latex_table(moments, title=titles, index=index, align='c'), file=f)
        print(write_markdown_table(moments, title=titles, index=index, align='c'))
