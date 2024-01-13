from menu_cost import MenuCostEnv
import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-6
def get_panel_data(p):
    n, T = p.shape
    p = np.concatenate((np.zeros((n, 1)), p), axis=1)
    freq_inc = p[:, 1:] - p[:, :-1] > epsilon
    freq_inc = np.average(freq_inc.reshape((n, -1, 12)), axis=-1)
    freq_dec = p[:, 1:] - p[:, :-1] < -epsilon
    freq_dec = np.average(freq_dec.reshape((n, -1, 12)), axis=-1)
    freq_change = np.abs(p[:, 1:] - p[:, :-1]) > epsilon
    freq_change = np.average(freq_change.reshape((n, -1, 12)), axis=-1)

    size_inc_ = np.maximum(p[:, 1:] - p[:, :-1], 0)
    size_inc = np.zeros(freq_inc.shape)
    size_inc[freq_inc > 0] = np.average(size_inc_.reshape((n, -1, 12)), axis=-1)[freq_inc > 0] / freq_inc[freq_inc > 0]
    size_dec_ = np.minimum(p[:, 1:] - p[:, :-1], 0)
    size_dec = np.zeros(freq_inc.shape)
    size_dec[freq_dec > 0] = np.average(size_dec_.reshape((n, -1, 12)), axis=-1)[freq_dec > 0] / freq_dec[freq_dec > 0]
    size_change_ = np.abs(p[:, 1:] - p[:, :-1])
    size_change = np.zeros(freq_inc.shape)
    size_change[freq_change > 0] = np.average(size_change_.reshape((n, -1, 12)), axis=-1)[freq_change > 0] / freq_change[freq_change > 0]

    return freq_inc, freq_dec, size_inc, size_dec, freq_change, size_change

if __name__ == '__main__':
    env = MenuCostEnv()
    res = env.vfi()
    env.plot_policy()
    p, a, cpi, eta, p_paths, _, eta_paths = env.simulate()
    data = list(get_panel_data(p))
    cpi = np.concatenate((np.zeros((cpi.shape[0], 1)), cpi), axis=1)
    cpi = np.sum((cpi[:, 1:] - cpi[:, :-1]).reshape((cpi.shape[0], -1, 12)), axis=-1)
    data.append(cpi)
    labels = ['Frequency of price increase', 'Frequency of price decrease', 
            'Size of price increase', 'Size of price decrease', 'Frequency of price change',
            'Size of price change', 'CPI', ]
    for item, label in zip(data, labels):
        seq = np.average(item, axis=0)
        if 'change' in label:
            continue
        plt.plot(seq, label=label)
    plt.xlabel('Time')
    plt.legend()
    plt.show()

    # Feel free to use Stata to run regression

    initial_prob = np.zeros(res.value.shape)
    initial_prob[int(res.value.shape[0] / 2), int(res.value.shape[1] / 2)] = 1
    hazard = env.hazard_function(initial_prob=initial_prob, length=100)
    plt.plot(hazard)
    plt.show()
