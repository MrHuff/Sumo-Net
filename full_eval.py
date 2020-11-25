# read .p file
# Select best model based on smaller runs
# load model with correct parameters
# Do full eval!

from serious_run import *

def calc_eval_objective( S, f, S_extended, durations, events, time_grid,train_objective):
    val_likelihood = train_objective(S, f)
    eval_obj = EvalSurv(surv=S_extended, durations=durations, events=events,
                        censor_surv='km')  # Add index and pass as DF
    conc = eval_obj.concordance_td()
    ibs = eval_obj.integrated_brier_score(time_grid)
    inll = eval_obj.integrated_nbll(time_grid)
    return val_likelihood, conc, ibs, inll

def eval_loop(grid_size,model,dataloader,train_objective,device):
    S_series_container = []
    S_log = []
    f_log = []
    durations = []
    events = []
    model = model.eval()
    # durations  = self.dataloader.dataset.invert_duration(self.dataloader.dataset.y.numpy()).squeeze()
    # events  = self.dataloader.dataset.delta.numpy()
    chunks = dataloader.batch_size // 50 + 1
    with torch.no_grad():
        t_grid_np = np.linspace(dataloader.dataset.min_duration, dataloader.dataset.max_duration,
                                grid_size)
        time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
        for i, (X, x_cat, y, delta) in enumerate(dataloader):
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
            if not isinstance(x_cat, list):
                for chk, chk_cat in zip(torch.chunk(X, chunks), torch.chunk(x_cat, chunks)):
                    input_time = time_grid.repeat((chk.shape[0], 1)).to(device)
                    X_repeat = chk.repeat_interleave(grid_size, 0)
                    x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                    S_serie = model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                    S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
            else:
                x_cat_repeat = []
                for chk in torch.chunk(X, chunks):
                    input_time = time_grid.repeat((chk.shape[0], 1)).to(device)
                    X_repeat = chk.repeat_interleave(grid_size, 0)
                    S_serie = model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                    S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
            S_log.append(S)
            f_log.append(f)
            durations.append(y.cpu().numpy())
            events.append(delta.cpu().numpy())
        durations = dataloader.dataset.invert_duration(np.concatenate(durations)).squeeze()
        # durations = np.concatenate(durations).squeeze()
        events = np.concatenate(events).squeeze()
        S_log = torch.cat(S_log)
        f_log = torch.cat(f_log)
        # reshape(-1, 1)).squeeze()
        S_series_container = pd.DataFrame(torch.cat(S_series_container, 1).numpy())
        t_grid_np = dataloader.dataset.invert_duration(t_grid_np.reshape(-1, 1)).squeeze()
        S_series_container = S_series_container.set_index(t_grid_np)
        # S_series_container=S_series_container.set_index(t_grid_np)
        val_likelihood, conc, ibs, inll = calc_eval_objective(S_log, f_log, S_series_container,
                                                                   durations=durations, events=events,
                                                                   time_grid=t_grid_np,train_objective=train_objective)
    return val_likelihood.item(), conc, ibs, inll

def get_best_model(load_path,selection_criteria):
    trials = pickle.load(open(load_path + 'hyperopt_database.p',
                              "rb"))

    if selection_criteria == 'train':
        reverse = False
    elif selection_criteria == 'concordance':
        reverse = True
    elif selection_criteria == 'ibs':
        reverse = False
    elif selection_criteria == 'inll':
        reverse = False
    best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0]  # low to high
    best_tid = sorted(trials.trials, key=lambda x: x['result']['test_loss'], reverse=reverse)[0]['misc']['tid']
    net_init_params = best_trial['net_init_params']
    return net_init_params,best_tid

if __name__ == '__main__':
    gpu = True
    if gpu:
        devices = GPUtil.getAvailable(order='memory', limit=8)
        device = devices[0]
    else:
        device="cpu"
    nr_of_seeds = 5
    dataset_id = 5
    bs = 500
    dataset_string = datasets[dataset_id]
    grid_size=500 #hmmm
    # for s in range(nr_of_seeds):
    s=123
    load_path = f'./{dataset_string}_{s}/'
    init_params,tid = get_best_model(load_path,'train')
    train_objective = get_objective(init_params['objective'])
    model = survival_net(**init_params)
    model.load_state_dict(torch.load(load_path + f'best_model_{tid}.pt'))
    model.to(device)
    dl = get_dataloader(dataset_string,bs,s)
    dl.dataset.set(mode='test')
    test_likelihood,test_conc,test_ibs,test_inll=eval_loop(grid_size=grid_size,model=model,dataloader=dl,train_objective=train_objective,device=device)
    data = [test_likelihood, test_conc, test_ibs,test_inll]
    df = pd.DataFrame([data], columns=['test_loglikelihood', 'test_conc', 'test_ibs', 'test_inll'])
    df.to_csv(load_path + 'best_results.csv', index_label=False)