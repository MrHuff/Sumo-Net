from utils.plot_utils import *
from utils.dataloaders import *
from debug_run import datasets


for i,X,max_time in zip([5,6,7],[torch.Tensor([[0],[0.3],[1.0]]),torch.Tensor([[0.1],[0.4]]),torch.Tensor([[0.0],[0.2],[0.4],[0.6],[0.8],[1.0]])],[2.0,1.0,120]):
    dstr = datasets[i]
    seed = 123
    model = load_best_model(dstr,seed)
    dl = get_dataloader(dstr,100,seed)
    dl.dataset.set('test')
    if i==7:
        begin=75
    else:
        begin = 0
    plot_survival(X,model,max_time,f'dataset_{i}_sanity.png',1000,begin)


