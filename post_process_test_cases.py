from utils.plot_utils import *
from utils.dataloaders import *
from debug_run import datasets
import pandas as pd
ds = [5,6,7]
X_list = [np.array([[0],[0.3],[1.0]]),np.array([[0.1],[0.2],[0.4],[0.6],[0.8],[1.0]]),np.array([[0.0],[0.2],[0.4],[0.6],[0.8],[1.0]])]
T_list = [2.0,1.0,150.0]
for i in [0,1,2]:
    X = X_list[i]
    max_time = T_list[i]
    dstr = datasets[ds[i]]
    seed = 123
    model = load_best_model(dstr, seed)
    model = model.eval()
    dl = get_dataloader(dstr, 100, seed)
    dl.dataset.set('test')
    if ds[i]==7:
        begin=40
    else:
        begin = 0
    X_in = pd.DataFrame(X,columns=['x1'])
    plot_survival(X_in, model, max_time, f'dataset_{ds[i]}_sanity.png',dl, 1000, begin)



