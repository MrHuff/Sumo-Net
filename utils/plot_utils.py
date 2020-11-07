import numpy as np
import pickle
from nets.nets import *
import matplotlib.pyplot as plt
from matplotlib import rcParams


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


def load_best_model(dataset_string,seed):
    PATH =f'./{dataset_string}_{seed}/'
    trials = pickle.load(open(PATH+'hyperopt_database.p', "rb"))
    best = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=False)[0]
    best_tid = best['misc']['tid']
    best_params = best['result']['net_init_params']
    model = survival_net(**best_params)
    model.load_state_dict(torch.load(PATH+f'best_model_{best_tid}.pt'))
    return model

def plot_survival(fixed_X,model,max_time,plt_name,dl,points=100,begin=0,):
    grid = torch.from_numpy( dl.dataset.transform_duration(np.linspace(begin, max_time,points).reshape(-1, 1)) ).float()
    x_in = torch.from_numpy(dl.dataset.transform_x(fixed_X)).float()
    for i in range(fixed_X.shape[0]):
        with torch.no_grad():
            S=model.forward_S_eval(x_in[i,:].unsqueeze(-1).repeat(points,1),grid)
            S = S.numpy()
        plt.plot( dl.dataset.invert_duration(grid.numpy()), S)
    plt.savefig(plt_name)
    plt.clf()



