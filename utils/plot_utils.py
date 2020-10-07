import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_survival(fixed_X,model,max_time,plt_name,points=100):
    grid = torch.from_numpy(np.linspace(0,max_time,points))
    with torch.no_grad():
        f,S=model(fixed_X,grid)
        S = S.numpy()

    plt.scatter(grid.numpy(),S)
    plt.savefig(plt_name)
    plt.clf()




