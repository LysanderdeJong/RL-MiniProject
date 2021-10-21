import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt



def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def running_gaussian(vals, sigma=3):
    return gaussian_filter(vals, sigma, mode='nearest')

def running_linear(vals, n=1):
    half_n = n // 2
    np.pad(vals, half_n, mode='edge')
    if n % 2 == 1:
        weights = np.array(list(range(half_n)) + list(reversed(range(half_n))), dtype='float')
    else:
        weights = np.array(list(range(half_n)) + [half_n + 1] + list(reversed(range(half_n))), dtype='float')
    weights /= weights.sum()
    return np.convolve(vals, weights, mode='same')



def plot_results(results, env_name, smooth=1):
    keys = list(results.keys())
    for i in results[keys[0]].keys():
        if i == "q_values":
            continue
        for j in keys:
            mean = running_mean(results[j][i][0],smooth)
            std = running_mean(results[j][i][1],smooth)
            plt.plot(mean, label=f"{j}")
            plt.fill_between(range(len(mean)), mean-std, mean+std, alpha=0.5)
        plt.legend()
        plt.ylabel(f"{i.replace('_', ' ').capitalize()}")
        plt.xlabel("Episodes")
        plt.title(f"{i.replace('_', ' ').capitalize()} on {env_name}")
        plt.savefig(f'figures/{env_name}_{i}.pdf', dpi=300)
        plt.show()
