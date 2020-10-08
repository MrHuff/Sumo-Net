from utils.plot_utils import *
from utils.dataloaders import *
from debug_run import datasets


dstr = datasets[1]
seed = 123
model = load_best_model(dstr,seed)
dl = get_dataloader(dstr,1,seed+5)
X,y,delta=next(iter(dl))
print(y)
plot_survival(X,model,10000,'test.png',1000)


