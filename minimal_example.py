import torch
from nets.nets import *
from utils.dataloaders import *
import tqdm


if __name__ == '__main__':

    dataset_string = 'support'
    bs=100
    seed=42
    fold_idx = 1
    dataloader = get_dataloader(dataset_string,bs,seed,fold_idx,sumo_net=True)  #Load data into dataloader

    #Specify neural network parameters
    x_c = dataloader.dataset.X.shape[1]
    net_init_params = {
        'd_in_x': x_c,
        'cat_size_list': dataloader.dataset.unique_cat_cols,
        'd_in_y': 1,
        'd_out': 1,
        'bounding_op': torch.relu,
        'transformation': torch.tanh,
        'layers_x': [32,32],
        'layers_t': [1],
        'layers': [32,32],
        'direct_dif': 'autograd',
        'objective': 'S_mean',
        'dropout': 0.1,
        'eps': None
    }

    #initialize model
    model = survival_net_basic(**net_init_params)

    #Optimization configuration
    wr=0.0
    lr=1e-2
    device = 'cuda:0'
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                      weight_decay=wr)
    #Train! For each epoch ("a runthrough of the entire dataset")
    for e in range(epochs):
        #Iterate through the data in batches!
        for i, (X, x_cat, y, delta) in enumerate(tqdm.tqdm(dataloader)):
            X = X.to(device)
            y = y.to(device)
            delta = delta.to(device)
            mask = delta == 1
            X_f = X[mask, :]
            y_f = y[mask, :]
            if not isinstance(x_cat, list):  # F
                x_cat = x_cat.to(device)
                x_cat_f = x_cat[mask, :]
            else:
                x_cat_f = []
            S = model.forward_cum(X, y, mask, x_cat)
            if S.numel() == 0:
                S = S.detach()
            f = model(X_f, y_f, x_cat_f)
            if f.numel() == 0:
                f = f.detach()
            total_loss = log_objective_mean(S, f)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
