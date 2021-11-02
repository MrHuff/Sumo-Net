import numpy as np

from serious_run import *
import matplotlib.pyplot as plt
def sumo_loop(model,dataloader,device='cuda:0'):
    model.eval()
    S_prob_list = []
    durations = []

    for i, (X, x_cat, y, delta, s_kmf) in enumerate(tqdm(dataloader)):
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
        with torch.no_grad():
            S_prob = model.forward_S_eval(X_f,y_f,x_cat_f)
        S_prob_list.append(S_prob.cpu().numpy())
        durations.append(y_f.cpu().numpy())
    S_prob_list = np.concatenate(S_prob_list)
    og_durations = dataloader.dataset.invert_duration(np.concatenate(durations))
    return og_durations,S_prob_list

def plot_event_times(dataset_string,path,seed,fold_idx,device):
    reverse = False
    trials = pickle.load(open(path + 'hyperopt_database.p', "rb"))
    best_trial = sorted(trials.trials, key=lambda x: x['result'][selection_criteria], reverse=reverse)[0]
    net_params = best_trial['result']['net_init_params']
    best_tid = best_trial['tid']
    sumo_net = net_type in ['survival_net_basic', 'deepsurv_benchmark', 'cox_CC_benchmark', 'cox_linear_benchmark',
                        'deephit_benchmark']
    dataloader = get_dataloader(dataset_string, 5000, seed, fold_idx, shuffle=False, sumo_net=sumo_net)
    model = survival_net_basic(**net_params).to(device)
    model.load_state_dict(torch.load(path + f'best_model_{best_tid}.pt', map_location=device))
    x,y = sumo_loop(model,dataloader,device)
    plt.scatter(x,y)
    plt.title(d_str)
    plt.xlabel('Event times')
    plt.ylabel(r'$S(t\mid x)$')
    plt.savefig(f'{dataset_string}_qualtitative.png')
    plt.clf()


if __name__ == '__main__':
    device='cuda:0'
    folder = 'kkbox_sumo_20_3_results'
    objective = ['S_mean']
    model = ['survival_net_basic']
    selection_criteria_list = ['test_loss']
    result_name = f'{folder}_results'
    # dataset_indices = [0, 1, 2, 3]
    dataset_indices = [4]
    plot_data = {}
    for net_type, selection_criteria in zip(model, selection_criteria_list):
        for o in objective:
            for d in dataset_indices:
                d_str = datasets[d]
                for s in [1337]:
                    for f_idx in [0]:
                        pat = f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/'
                        plot_event_times(d_str,pat,s,f_idx,device)




