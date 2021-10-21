import numpy as np
from scipy.ndimage import gaussian_filter

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
