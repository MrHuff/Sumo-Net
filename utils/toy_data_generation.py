import numpy as np
import math
import scipy
import pandas as pd
import os
class toy_data_class():
    def __init__(self,variant):
        self.load_path = f'./{variant}/'
        self.log_cols = ['x_1']
        self.col_event = 'delta'
        self.col_duration = 'y'

    def read_df(self):
        df = pd.read_csv(self.load_path+'data.csv')
        print(df['x_1'].max())
        print(df['y'].max())
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

    def __init__(self, mean, var_slope):
        self.mean = mean
        self.var_slope = var_slope

    def surv_given_x(self, t, x):
        return (1 - scipy.stats.norm.cdf(t, loc=self.mean, scale=1 + self.var_slope * x))

    def sample(self, n):
        x = np.random.uniform(size=n)
        t = np.zeros(shape=n)
        for i in range(n):
            t[i] = np.random.normal(loc=self.mean, scale=1. + self.var_slope * x[i])
        return (x, t)

    def get_censoring(self,n):
        return np.random.normal(loc=self.mean, scale=self.var_slope, size=n)

def get_delta_and_z(t,c):
    d = np.int32(c > t)
    z = np.minimum(t, c)
    #z = (z-z.mean())/z.std()
    return d,z


def generate_toy_data(variant,n,**kwargs):
    if variant=='weibull':
        d = weibull(kwargs['a'],kwargs['b'])
    elif variant=='checkboard':
        d = checkerboard_grid(kwargs['grid_width'],kwargs['grid_length'],kwargs['num_tiles_width'],kwargs['num_tiles_length'])
    elif variant =='normal':
        d = varying_normals(kwargs['mean'],kwargs['var_slope'])

    x,t = d.sample(n)
    c = d.get_censoring(n)
    d,z =get_delta_and_z(t,c)
    cols = ['x_1','delta','y']
    df = pd.DataFrame(np.stack([x,d,z],axis=1),columns=cols)
    data_path =f'../{variant}/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    df.to_csv(data_path+'data.csv',index=False)

if __name__ == '__main__':
    n=25000
    kwargs = {'a': 2.,'b':6.,'grid_width': 1., 'grid_length': 1., 'num_tiles_width': 4., 'num_tiles_length': 6.,'mean':100.,'var_slope':6.}
    for variant in ['weibull','checkboard','normal']:
        generate_toy_data(variant,n,**kwargs)










