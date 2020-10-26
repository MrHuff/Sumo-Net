from utils.plot_utils import *
from utils.dataloaders import *
from debug_run import datasets

ds = [5,6,7]
X_list = [torch.Tensor([[-1], [-0.5], [0], [0.5], [1.0]]),
          torch.Tensor([[-1], [-0.5], [0], [0.5], [1.0]]),
          torch.Tensor([[-1], [-0.5], [0], [0.5], [1.0]])]
max_times = [2, 2, 2]
min_times = [-2, -2, -2]
for i in [0, 1, 2]:
    X = X_list[i]
    max_time = max_times[i]
    min_time = min_times[i]
    dstr = datasets[ds[i]]
    seed = 123
    model, params = load_best_model(dstr, seed)
    dl = get_dataloader(dstr, 100, seed)
    dl.dataset.set('test')
    if ds[i] == 7:
        begin = 0
    else:
        begin = 0
    plot_survival(X, model, params, min_time, max_time, f'dataset_{ds[i]}_sanity.png', 1000)



