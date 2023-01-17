import logging, os
from KrusellSmithEnv import KrusellSmithEnv
from utilize import gen_time_series_moments, compute_gini
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

logfile = './KrusellSmith.log'
figure_path = './figures/final_project/'

if __name__ == '__main__':
    if os.path.exists(logfile):
        os.remove(logfile)

    logger = logging.getLogger('FinalProject')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    env = KrusellSmithEnv()

    # problem a
    env.no_agg_risk_equilibrium('good', r_init=0.0312)
    env.no_agg_risk_equilibrium('bad', r_init=0.0322)

    # problem b
    env.solve_approximated_equilibrium()

    # problem c
    env.plot_policy(figure_path=figure_path)

    simulation_length = 11000
    burned = 1000
    agg_k, agg_y, agg_c, agg_i, income_vec, kvec, idxz, z = env.full_simulation(length=simulation_length)

    # problem f
    rbc_mean, rbc_cov, rbc_corr, rbc_ar = gen_time_series_moments(np.array([agg_c, agg_i, agg_k, agg_y])[:, burned:])
    logger.info('Business cycle data mean: consumption {}, investment {}, capital stock {}, output {}.'
                .format(rbc_mean[0], rbc_mean[1], rbc_mean[2], rbc_mean[3]))
    logger.info('Business cycle data covariance matrix (c, i, k, y):\n {}.'.format(rbc_cov))
    logger.info('Business cycle data correlation matrix (c, i, k, y):\n {}.'.format(rbc_corr))
    logger.info('Business cycle data auto-correlation: consumption {}, investment {}, capital stock {}, output {}.'
                .format(rbc_ar[0], rbc_ar[1], rbc_ar[2], rbc_ar[3]))

    logger.info(
        'Normalized business cycle data mean (Ey = 1): consumption {}, investment {}, capital stock {}, output {}.'
        .format(rbc_mean[0] / rbc_mean[3], rbc_mean[1] / rbc_mean[3], rbc_mean[2] / rbc_mean[3], 1))
    logger.info(
        'Normalized business cycle data covariance matrix (c, i, k, y):\n {}.'.format(rbc_cov / rbc_mean[3] ** 2))

    # problem e
    periods = [1, 10, 50]
    valid_k = agg_k[burned:]
    k_num = len(valid_k)
    p_max = np.max(periods)
    idxz_series = np.zeros((p_max, simulation_length - p_max - burned))
    for idxk in range(k_num - p_max):
        idxz_series[:, idxk] = idxz[burned + idxk: burned + idxk + p_max]
    real_k = valid_k[p_max:]
    logger.info('Forecasting: real series mean {}, real series median {}.'.format(np.mean(real_k), np.median(real_k)))
    forcast_ks = []
    for p in periods:
        forcast_k = env.forecast_aggk(valid_k[p_max - p: -p], idxz_series[-p:], p)
        logger.info("Forecast over {} period(s): average forecast error {}, forecast mean {}, forecast median {}."
                    .format(p, np.mean(np.abs(forcast_k - real_k)), np.mean(forcast_k), np.median(forcast_k)))
        plt.hist(forcast_k)
        plt.title('Forecast k distribution over {} period(s)'.format(p))
        plt.savefig(os.path.join(figure_path, 'forecast_{}.png'.format(p)))
        plt.savefig(os.path.join(figure_path, 'forecast_{}.pdf'.format(p)))
        plt.show()
        forcast_ks.append(forcast_k)
    plt.hist(real_k)
    plt.title('Real k distribution')
    plt.savefig(os.path.join(figure_path, 'real.png'))
    plt.savefig(os.path.join(figure_path, 'real.pdf'))
    plt.show()

    for p, fk in zip(periods, forcast_ks):
        plt.plot(fk[-1000:], label=str(p))
    plt.plot(real_k[-1000:], label='real')
    plt.legend()
    plt.title('forcast series')
    plt.savefig(os.path.join(figure_path, 'forecast_series.png'))
    plt.savefig(os.path.join(figure_path, 'forecast_series.pdf'))
    plt.show()

    # problem g
    v_name = ['wealth', 'income']
    for vidx, vvec in enumerate([kvec, income_vec]):
        vvec = vvec[burned:]

        pooled_gini, pooled_lorenz = compute_gini(vvec[-10:].ravel())
        average_gini, average_lorenz = compute_gini(np.mean(vvec[-10:], axis=0))

        gini_list = []
        lorenz_list = []
        for t in range(10):
            gini, lorenz = compute_gini(vvec[-t])
            gini_list.append(gini)
            lorenz_list.append(lorenz)
        gini_average = np.mean(gini_list)
        lorenz_average = np.mean(lorenz_list, axis=0)

        logger.info(
            '{} gini index for the last ten periods: pooled gini {}, averaged gini {}, average of period gini {}'
            .format(v_name[vidx], pooled_gini, average_gini, gini_average))
        plt.plot(np.linspace(0, 1, len(pooled_lorenz), endpoint=True), pooled_lorenz, label='Gini pooled')
        plt.plot(np.linspace(0, 1, len(average_lorenz), endpoint=True), average_lorenz, label='Gini averaged')
        plt.plot(np.linspace(0, 1, len(lorenz_average), endpoint=True), lorenz_average, label='average Gini')
        plt.plot(np.linspace(0, 1, len(pooled_lorenz), endpoint=True),
                 np.linspace(0, 1, len(pooled_lorenz), endpoint=True), label='45 degree')
        plt.legend()
        plt.title('{} lorenz curve'.format(v_name[vidx]))
        plt.savefig(os.path.join(figure_path, 'gini_{}.png'.format(v_name[vidx])))
        plt.savefig(os.path.join(figure_path, 'gini_{}.pdf'.format(v_name[vidx])))
        plt.show()

        mean, var, skw, kur = np.mean(vvec, axis=1), np.var(vvec, axis=1), skew(vvec, axis=1), kurtosis(vvec, axis=1)
        for name, v in zip(['mean', 'variance', 'skewness', 'kurtosis'], [mean, var, skw, kur]):
            plt.plot(v[-1000:])
            plt.title('{} of {} distribution'.format(name, v_name[vidx]))
            plt.savefig(os.path.join(figure_path, '{}_{}.png'.format(name, v_name[vidx])))
            plt.savefig(os.path.join(figure_path, '{}_{}.pdf'.format(name, v_name[vidx])))
            plt.show()

