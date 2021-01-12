import numpy as np
from serious_run import datasets
from utils.dataloaders import *
import os
import shutil
from tqdm import tqdm
from kmeans_pytorch import kmeans
from utils.plot_utils import *


if __name__ == '__main__':
    save_folder = 'sumo_example_plots'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:
        shutil.rmtree(save_folder)
        os.makedirs(save_folder)
    idx = 4
    d_str = datasets[idx]
    net_types = ['survival_net_basic','benchmark']
    net_type = net_types[0]
    o = 'S_mean'
    bs = 2000
    seed = 1337
    fold_idx = 3
    folder = 'ibs_eval'
    PATH = f'./{folder}/{d_str}_seed={seed}_fold_idx={fold_idx}_objective={o}_{net_type}/'
    model = load_best_model(PATH=PATH)
    model = model.eval()
    dataloader = get_dataloader(d_str,bs,seed,fold_idx)
    dataloader.dataset.set('test')
    chunks=10
    grid_size=100
    S_series_container = []
    S_log = []
    f_log = []
    durations = []
    events = []
    device = 'cuda:0'
    model = model.to(device)
    num_clusters = 10
    with torch.no_grad():
        t_grid_np = np.linspace(dataloader.dataset.min_duration, dataloader.dataset.max_duration,
                                grid_size)
        time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
        for i, (X, x_cat, y, delta) in enumerate(tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            delta = delta.to(device)
            mask = delta == 1
            X_f = X[mask, :]
            y_f = y[mask, :]
            if not isinstance(x_cat, list):
                x_cat = x_cat.to(device)
                x_cat_f = x_cat[mask, :]
            else:
                x_cat_f = []
            S = model.forward_cum(X, y, mask, x_cat)
            f = model(X_f, y_f, x_cat_f)
            x_cat_repeat = []
            for chk, chk_cat in zip(torch.chunk(X, chunks), torch.chunk(x_cat, chunks)):
                input_time = time_grid.repeat((chk.shape[0], 1)).to(device)
                X_repeat = chk.repeat_interleave(grid_size, 0)
                x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                S_serie = model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                S_series_container.append(S_serie.view(-1, grid_size))
        t_grid_np = dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
        S_series_container = torch.cat(S_series_container,dim=0)
        cluster_ids_x, cluster_centers = kmeans(
            X=S_series_container, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
        )
        n = cluster_ids_x.shape[0]
        u,counts = cluster_ids_x.unique(return_counts=True)
        counts_prop = counts.float().cpu().numpy()/n *100

    rcParams['figure.figsize'] = 40, 20
    for i in range(cluster_centers.shape[0]):
        plt.plot( t_grid_np,  S_series_container[i,:].cpu().numpy(),label=f'Cluster {i}: {round(counts_prop[i].item())}'+r'\%') #fix
    plt.legend(borderpad=1)
    plt.xlabel('Time')
    plt.ylabel(r'S(t)')
    plt.savefig('test_kkbox_kmeans.png')



