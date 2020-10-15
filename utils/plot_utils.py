import matplotlib.pyplot as plt
import torch
import numpy as np
import pickle
from nets.nets import *

def load_best_model(dataset_string,seed):
    PATH =f'./{dataset_string}_{seed}/'
    trials = pickle.load(open(PATH+'hyperopt_database.p', "rb"))
    best = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=False)[0]
    best_tid = best['misc']['tid']
    best_params = best['result']['net_init_params']
    model = survival_net(**best_params)
    model.load_state_dict(torch.load(PATH+f'best_model_{best_tid}.pt'))
    return model

def plot_survival(fixed_X,model,max_time,plt_name,points=100,begin=0):
    grid = torch.from_numpy(np.linspace(begin,max_time,points)).float().unsqueeze(-1)

    for i in range(fixed_X.shape[0]):
        with torch.no_grad():
            S=model.forward_S(fixed_X[i,:].unsqueeze(-1).repeat(points,1),grid)
            S = S.numpy()
        plt.scatter(grid.numpy(),S,s=4)
    plt.savefig(plt_name)
    plt.clf()




