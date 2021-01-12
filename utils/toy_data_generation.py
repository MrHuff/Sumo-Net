import numpy as np
import math
import scipy
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import scipy.stats
from statsmodels.distributions import ECDF
#Empirical CDF
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

class toy_data_class():
    def __init__(self,variant):
        self.load_path = f'./{variant}/'
        self.log_cols = ['x1']
        self.col_event = 'event'
        self.col_duration = 'duration'

    def read_df(self):
        df = pd.read_csv(self.load_path+'data.csv')
        print('covariate (X)', df[self.log_cols].max())
        print('target (y)', df[self.col_duration].max())
        return df

class weibull():  # a weibull distribution with scale set to 1
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def shape_given_x(self, x):
        return (self.a + self.b * x)

    def surv_given_x(self, t, x):
        shape = self.shape_given_x(x)
        return (np.exp(-t ** shape))

    def sample(self, n):
        x = np.random.uniform(0, 1, size=n)
        t = np.zeros(n)
        for i in range(n):
            t[i] = np.random.weibull(a=self.shape_given_x(x[i]))
        return (x, t)

    def get_censoring(self,n):
        return np.random.exponential(1.5, size = n)

class checkerboard_grid():
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

    def get_censoring(self,n):
        return np.random.exponential(1.5, size = n)

class varying_normals():

    def __init__(self, mean, var_slope,intercept):
        self.mean = mean
        self.var_slope = var_slope
        self.intercept = intercept

    def surv_given_x(self, t, x):
        return (1 - scipy.stats.norm.cdf(t, loc=self.mean, scale=self.intercept + self.var_slope * x))

    def sample(self, n):
        x = np.random.uniform(size=n)
        t = np.zeros(shape=n)
        for i in range(n):
            t[i] = np.random.normal(loc=self.mean, scale=self.intercept + self.var_slope * x[i])
        return (x, t)

    def get_censoring(self,n):
        return np.random.normal(loc=self.mean, scale=np.abs(self.var_slope), size=n)
        #return np.random.exponential(self.mean, size=n)

def get_delta_and_z(t,c):
    d = np.int32(c > t)
    z = np.minimum(t, c)
    return d,z

def unit_scaling(z): #Trick is probably to match x's and y's... i.e. it gets confused if function has to map "sparsely"
    mean = z.mean()
    std = z.std()
    z = (z-mean)/std
    return z

def max_min_scaling(x):
    max = x.max()
    min = x.min()
    return (x-min)/(max-min), min ,max

def cum_f_plot(variant,x_array,t_array,d,comment):
    legend = []
    for i,x in enumerate(x_array):
        S = lambda t: d.surv_given_x(t, x)
        S_array = [S(t) for t in t_array]
        p = plt.plot(t_array, S_array,label=f'x={x}')
        legend.append(p[0])
    plt.legend()
    plt.savefig(f"{variant}_{comment}.png",tight_layout=True)
    plt.clf()


def scatter_plot_1(variant,x,t,comment):
    plt.scatter(x, t)
    plt.savefig(f'{variant}_scatter_{comment}.png')
    plt.clf()

def scatter_plot_2(variant,x,z,observed,censored):
    fig_1 = plt.axes()
    fig_1.scatter(x[observed], z[observed], c='navy', label='observed')
    fig_1.figure.savefig(f'{variant}_observed.png')
    fig_2 = plt.axes()
    fig_2.scatter(x[censored], z[censored], facecolors='none', edgecolors='r', label='censored')
    fig_2.figure.savefig(f'{variant}_censored.png')
    fig_2.figure.clf()
    fig_1.figure.clf()

def empirical_cdf(variant,t,comment):
    ecdf = ECDF(t)
    plt.scatter(ecdf.x,1-ecdf.y)
    plt.savefig(f'{variant}_{comment}_ecdf.png')
    plt.clf()

def generate_toy_data(variant,n,**kwargs):
    if variant=='weibull':
        t_array = np.linspace(0, 2, num=100)
        x_array = [0.0, 0.3, 1.0]
        dist = weibull(kwargs['a'],kwargs['b'])
    elif variant=='checkboard':
        t_array = np.linspace(0, 1, num=100)
        x_array = [0.1, 0.4]
        dist = checkerboard_grid(kwargs['grid_width'],kwargs['grid_length'],kwargs['num_tiles_width'],kwargs['num_tiles_length'])
    elif variant =='normal':
        t_array = np.linspace(80, 120, num=100)
        x_array = [ 0.2, 0.4, 0.6, 0.8, 1.0]
        dist = varying_normals(kwargs['mean'],kwargs['var_slope'],kwargs['intercept'])

    x,t = dist.sample(n)
    empirical_cdf(variant,t,'pre')
    cum_f_plot(variant=variant,x_array=x_array,t_array=t_array,d=dist,comment='pre')
    # scatter_plot_1(variant,x,t,'pre')
    # c = dist.get_censoring(n)
    # d,z =get_delta_and_z(t,c)
    # empirical_cdf(variant,t,'after')
    #
    # # scatter_plot_1(variant,x,t,'after')
    # censored, observed = np.where(d == 0), np.where(d == 1)
    # scatter_plot_2(variant,x,z,observed,censored)
    # # cum_f_plot(variant,(x_array-x_min)/(x_max-x_min),(t_array-t_min)/(t_max-t_min),dist,'after')
    #
    # cols = ['x1','event','duration']
    # df = pd.DataFrame(np.stack([x,d,z],axis=1),columns=cols)
    # data_path =f'../{variant}/'
    # if not os.path.exists(data_path):
    #     os.makedirs(data_path)
    # df.to_csv(data_path+'data.csv',index=False)

if __name__ == '__main__':
    n=25000
    kwargs = {'a': 2.,'b':6.,'grid_width': 1., 'grid_length': 1., 'num_tiles_width': 4., 'num_tiles_length': 6.,'mean':100.,'var_slope':6.,'intercept':0}
    for variant in ['weibull','checkboard','normal']:
        generate_toy_data(variant,n,**kwargs)










