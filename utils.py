import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from experiments import multi_trail_experiment



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
        if i == "q_values" or i == "outcomes":
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



def parameter_search(environment, config, parameter, values):
    all_results = []
    for v in values:
        config["q_" + parameter] = v
        config["dq_" + parameter] = v
        results = multi_trail_experiment("", environment, config)
        result_q = np.asarray(results["Q"]["outcomes"]).mean()
        result_dq = np.asarray(results["DQ"]["outcomes"]).mean()
        all_results.append((v, result_q, result_dq))
    for (v, q, dq) in all_results:
        print(f"{parameter} = {v}: \t{q:.4f}, \t{dq:.4f}")
