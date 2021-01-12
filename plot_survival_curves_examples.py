import numpy as np
from serious_run import datasets
from utils.plot_utils import *
from utils.dataloaders import *
import os
import shutil
if __name__ == '__main__':
    save_folder = 'sumo_example_plots'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    for idx in [5,6,7]:
        d_str = datasets[idx]
        if d_str=='weibull':
            t_array = np.linspace(0, 2, num=100)
            x_array = [0.0, 0.3, 1.0]
        elif d_str=='checkboard':
            t_array = np.linspace(0, 1, num=100)
            x_array = [0.1, 0.4]
        elif d_str =='normal':
            t_array = np.linspace(80, 120, num=100)
            x_array = [ 0.2, 0.4, 0.6, 0.8, 1.0]

        net_types = ['survival_net_basic','benchmark']
        net_type = net_types[0]
        o = 'S_mean'
        bs = 100
        seed = 1337
        fold_idx = 3
        folder = 'ibs_eval_new_example'
        PATH = f'./{folder}/{d_str}_seed={seed}_fold_idx={fold_idx}_objective={o}_{net_type}/'
        model = load_best_model(PATH=PATH)
        model = model.eval()
        dl = get_dataloader(d_str,bs,seed,fold_idx)
        dat = pd.DataFrame(np.array(x_array).reshape(-1,1),columns=['x1'])
        plot_survival(dat,t_array.reshape(-1,1),dl,model,f'{save_folder}/{d_str}_{fold_idx}_survival_plot.png')






