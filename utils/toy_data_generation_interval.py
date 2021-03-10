from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math
from decimal import *
import pandas as pd
getcontext().prec = 28
import bisect
import shutil
import os
rcParams.update({'figure.autolayout': True})
plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 6
plt.rcParams['axes.labelsize'] = 35
plt.rcParams['axes.titlesize'] = 35
plt.rcParams['font.size'] = 35
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 26
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"

np.random.seed(1)


class Weibull():  # A weibull distribution with scale set to 1

    def __init__(self, a, b):  # The shape of the weibull is a + b * X
        self.a = a
        self.b = b

    def shape_given_x(self, x):
        return self.a + self.b * x

    def survival_given_x(self, t, x):
        shape = self.shape_given_x(x)
        return np.exp(-t ** shape)

    def sample(self, n):
        x = np.random.uniform(0, 1, size=n)
        t = np.zeros(n)
        for i in range(n):
            t[i] = np.random.weibull(a=self.shape_given_x(x[i]))
        return x, t

class VaryingNormals:

    def __init__(self, mean, var_slope):
        self.mean = mean
        self.var_slope = var_slope

    def survival_given_x(self, t, x):
        return 1 - scipy.stats.norm.cdf(t, loc=self.mean, scale=1 + self.var_slope * x)

    def sample(self, n):
        x = np.random.uniform(size=n)
        t = np.zeros(shape=n)
        for i in range(n):
            t[i] = np.random.normal(loc=self.mean, scale=1 + self.var_slope * x[i])
        return x, t

class Checkerboard:

    def __init__(self, grid_width, grid_length, num_tiles_width,
                 num_tiles_length):  # num_tiles_length needs to be even.
        self.grid_width = grid_width
        self.grid_length = grid_length
        self.num_tiles_width = num_tiles_width
        self.num_tiles_length = num_tiles_length
        self.tile_width = self.grid_width / self.num_tiles_width
        self.tile_length = self.grid_length / self.num_tiles_length

    def find_class(self, x):
        return (math.floor(x / self.tile_width) % 2)

    def surv_given_x(self, t, x):
        c = math.floor(t / (self.tile_length * 2))
        res = t % (2 * self.tile_length)
        if self.find_class(x) == 0:
            c += min(1, res / self.tile_length)
        elif self.find_class(x) == 1:
            c += max(0, (res - self.tile_length) / self.tile_length)
        return (1 - max(0, min(c / (self.num_tiles_length / 2), 1)))

    def sample(self, n):
        x = np.random.uniform(0, self.grid_width, size=n)
        t = np.array([self.find_class(xi) for xi in x]) * self.tile_length
        t += np.random.choice(np.arange(0, self.num_tiles_length, step=2), size=n) * self.tile_length
        t += np.random.uniform(low=0, high=self.tile_length, size=n)
        return (x, t)


class Censoring_checkers(object):

    def __init__(self, mean, n_intervals, n):
        self.n_intervals = n_intervals
        self.n = n
        self.mean = mean
        self.observation_grid = np.zeros((n, n_intervals + 2))
        for k in range(n_intervals):
            c = np.random.exponential(mean, size=n)
            self.observation_grid[:, k + 1] = self.observation_grid[:, k] + c
        self.observation_grid[:, -1] = np.inf

    def get_intervals(self, t):
        intervals = np.zeros((self.n, 2))
        for i in range(self.n):
            right_idx = bisect.bisect_right(self.observation_grid[i], t[i])
            left_idx = right_idx - 1
            intervals[i, :] = [self.observation_grid[i, left_idx], self.observation_grid[i, right_idx]]
        return intervals

class Censoring_normals(object):


    def __init__(self, mean, n_intervals, n):
        self.n_intervals = n_intervals
        self.n = n
        self.mean = mean
        self.observation_grid = np.zeros((n, n_intervals + 2)) + 80
        for k in range(n_intervals):
            c = np.random.exponential(mean, size=n)
            self.observation_grid[:, k + 1] = self.observation_grid[:, k] + c
        self.observation_grid[:, -1] = np.inf

    def get_intervals(self, t):
        intervals = np.zeros((self.n, 2))
        for i in range(self.n):
            right_idx = bisect.bisect_right(self.observation_grid[i], t[i])
            left_idx = right_idx - 1
            intervals[i, :] = [self.observation_grid[i, left_idx], self.observation_grid[i, right_idx]]
        return intervals

class Censoring_weibulls(object):

    def __init__(self, mean, n_intervals, n):
        self.n_intervals = n_intervals
        self.n = n
        self.mean = mean
        self.observation_grid = np.zeros((n, n_intervals + 2))
        for k in range(n_intervals):
            c = np.random.exponential(mean, size=n)
            self.observation_grid[:, k + 1] = self.observation_grid[:, k] + c
        self.observation_grid[:, -1] = np.inf

    def get_intervals(self, t):
        intervals = np.zeros((self.n, 2))
        for i in range(self.n):
            right_idx = bisect.bisect_right(self.observation_grid[i], t[i])
            left_idx = right_idx - 1
            intervals[i, :] = [self.observation_grid[i, left_idx], self.observation_grid[i, right_idx]]
        return intervals


def generate_weibull(n):
    mean_censoring_interval = 0.1
    number_of_censoring_intervals = 14
    censoring = Censoring_weibulls(mean_censoring_interval, number_of_censoring_intervals, n)
    distribution = Weibull(a=2, b=6)
    X, T = distribution.sample(n)
    Z = censoring.get_intervals(T)
    df = pd.DataFrame(np.concatenate([X[:,np.newaxis],Z],axis=1),columns=['X','l','r'])
    if os.path.exists("../interval_weibull/interval_weibull.csv"):
        shutil.rmtree("../interval_weibull/")
        os.makedirs("../interval_weibull/")
    else:
        os.makedirs("../interval_weibull/")
    df.to_csv("../interval_weibull/interval_weibull.csv")
    
def generate_normal(n):
    mean_censoring_interval = 1
    number_of_censoring_intervals = 30
    censoring = Censoring_normals(mean_censoring_interval, number_of_censoring_intervals, n)
    distribution = VaryingNormals(mean=100, var_slope=6)
    X, T = distribution.sample(n)
    Z = censoring.get_intervals(T)
    df = pd.DataFrame(np.concatenate([X[:,np.newaxis],Z],axis=1),columns=['X','l','r'])
    if os.path.exists("../interval_normal/interval_normal.csv"):
        shutil.rmtree("../interval_normal/")
        os.makedirs("../interval_normal/")
    else:
        os.makedirs("../interval_normal/")
    df.to_csv("../interval_normal/interval_normal.csv")

def generate_checkboard(n):
    mean_censoring_interval = 0.02
    number_of_censoring_intervals = 50
    censoring = Censoring_checkers(mean_censoring_interval, number_of_censoring_intervals, n)
    distribution = Checkerboard(grid_width=1, grid_length=1, num_tiles_width=4, num_tiles_length=6)
    X, T = distribution.sample(n)
    Z = censoring.get_intervals(T)
    df = pd.DataFrame(np.concatenate([X[:,np.newaxis],Z],axis=1),columns=['X','l','r'])
    if os.path.exists("../interval_checkboard/interval_checkboard.csv"):
        shutil.rmtree("../interval_checkboard/")
        os.makedirs("../interval_checkboard/")
    else:
        os.makedirs("../interval_checkboard/")
    df.to_csv("../interval_checkboard/interval_checkboard.csv")
if __name__ == '__main__':
    n=25000
    generate_weibull(n)
    generate_normal(n)
    generate_checkboard(n)
