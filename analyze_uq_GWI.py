from full_eval import *
import GPUtil
from nets.GWI import *
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
def predict_one_point(X, x_cat, y, delta,dataloader,model):

    with torch.no_grad():
        t_grid_np = np.linspace(dataloader.dataset.min_duration, dataloader.dataset.max_duration,
                                grid_size)
        time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
        X_repeat = X.repeat_interleave(grid_size, 0)
        if not isinstance(x_cat, list):
            x_cat_repeat = x_cat.repeat_interleave(grid_size, 0)
        else:
            x_cat_repeat = []
        h_mean = model.predict_mean_h(X_repeat, time_grid, x_cat_repeat).squeeze()  # Fix
        h_std = model.predict_variance_h(X_repeat, time_grid)**0.5  # Fix
        S_up = 1- torch.sigmoid(h_mean + h_std).numpy()
        S_down = 1- torch.sigmoid(h_mean - h_std).numpy()
        S_mean = 1-torch.sigmoid(h_mean).numpy()
        t_grid_np = dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
        return S_up,S_down,S_mean,t_grid_np

if __name__ == '__main__':
    gpu = False
    if gpu:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        device = devices[0]
    else:
        device="cpu"
    nr_of_seeds = 5
    dataset_id = 0
    bs = 1
    dataset_string = datasets[dataset_id]
    grid_size=500 #hmmm
    # for s in range(nr_of_seeds):
    s=1
    load_path = f'test/support_seed=1_fold_idx=1_objective=S_mean_survival_GWI/'
    init_params,tid = get_best_model(load_path,'train')
    train_objective = get_objective(init_params['objective'])
    model= torch.load(load_path + f'best_model_full_{tid}.pt')
    model.to(device)
    dl = get_dataloader(dataset_string,bs,s,fold_idx=1)
    dl.dataset.set(mode='test')
    i, (X, x_cat, y, delta) = next(enumerate(dl))
    S_up,S_down,S_mean,time_grid = predict_one_point(X, x_cat, y, delta,dl,model)
    fig, ax = plt.subplots()
    ax.plot(time_grid,S_mean)
    ax.fill_between(time_grid, (S_down), (S_up), color='b', alpha=.1)
    plt.savefig('surv_gwi.png')

    # data = [test_likelihood, test_conc, test_ibs,test_inll]
    # df = pd.DataFrame([data], columns=['test_loglikelihood', 'test_conc', 'test_ibs', 'test_inll'])
    # df.to_csv(load_path + 'best_results.csv', index_label=False)