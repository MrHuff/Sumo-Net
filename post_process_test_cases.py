from utils.plot_utils import *
from utils.dataloaders import *
from debug_run import datasets

ds = [5,6,7]
X_list = [torch.Tensor([[0],[0.3],[1.0]]),torch.Tensor([[0.1],[0.4]]),torch.Tensor([[0.0],[0.2],[0.4],[0.6],[0.8],[1.0]])]
T_list = [2.0,1.0,120]
for i in [0,1,2]:
    X = X_list[i]
    max_time = T_list[i]
    dstr = datasets[ds[i]]
    seed = 123
    model = load_best_model(dstr, seed)
    dl = get_dataloader(dstr, 100, seed)
    dl.dataset.set('test')
    if ds[i]==7:
        begin=0
    else:
        begin = 0
    plot_survival(X, model, max_time, f'dataset_{ds[i]}_sanity.png', 1000, begin)



