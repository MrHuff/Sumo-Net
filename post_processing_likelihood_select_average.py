from serious_run import *
from astropy.table import Table
from utils.hazard_model_likelihood import *
DURS = [50, 100, 200, 400]

def sumo_loop(model,dataloader,device='cuda:0',grid_size=100):
    model.eval()
    S_series_container = []
    S_log = []
    f_log = []
    durations = []
    events = []
    chunks = dataloader.batch_size // 50 + 1
    t_grid_np = np.linspace(dataloader.dataset.min_duration, dataloader.dataset.max_duration,
                            grid_size)
    time_grid = torch.from_numpy(t_grid_np).float().unsqueeze(-1)
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
        S = model.forward_cum(X, y, mask, x_cat)
        S = S.detach()
        f = model(X_f, y_f, x_cat_f)
        f = f.detach()
        if not isinstance(x_cat, list):
            for chk, chk_cat in zip(torch.chunk(X, chunks), torch.chunk(x_cat, chunks)):
                input_time = time_grid.repeat((chk.shape[0], 1)).to(device)
                X_repeat = chk.repeat_interleave(grid_size, 0)
                x_cat_repeat = chk_cat.repeat_interleave(grid_size, 0)
                S_serie = model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                S_serie = S_serie.detach()
                S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
        else:
            x_cat_repeat = []
            for chk in torch.chunk(X, chunks):
                input_time = time_grid.repeat((chk.shape[0], 1)).to(device)
                X_repeat = chk.repeat_interleave(grid_size, 0)
                S_serie = model.forward_S_eval(X_repeat, input_time, x_cat_repeat)  # Fix
                S_serie = S_serie.detach()
                S_series_container.append(S_serie.view(-1, grid_size).t().cpu())
        S_log.append(S)
        f_log.append(f)
        durations.append(y.cpu().numpy())
        events.append(delta.cpu().numpy())
    non_normalized_durations = np.concatenate(durations)
    events = np.concatenate(events).squeeze()
    S_series_container = pd.DataFrame(torch.cat(S_series_container, 1).numpy())
    S_series_container_2 = S_series_container.set_index(t_grid_np)


    return torch.from_numpy(non_normalized_durations).float(),torch.from_numpy(events),S_series_container_2



def get_likelihoods(PATH,best_tid,net_init_params,dataset_string,seed,fold_idx,half_width,device='cuda:0',num_dur=0):
    sumo_net = net_type in ['survival_net_basic', 'deepsurv_benchmark', 'cox_CC_benchmark', 'cox_linear_benchmark',
                         'deephit_benchmark']
    dataloader = get_dataloader(dataset_string,5000, seed, fold_idx,shuffle=False,sumo_net=sumo_net)
    cat_cols_nr = len(dataloader.dataset.unique_cat_cols)

    if net_type in ['survival_net_basic']:
        model = survival_net_basic(**net_init_params).to(device)
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))
    elif net_type == 'weibull_net':
        model = weibull_net(**net_init_params).to(device)
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))
    elif net_type == 'lognormal_net':
        model = lognormal_net(**net_init_params).to(device)
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))

    elif net_type == 'cox_time_benchmark':
        if cat_cols_nr == 0:
            model = MLPVanillaCoxTime(in_features=net_init_params['d_in_x'],
                                           num_nodes=net_init_params['layers'],
                                           batch_norm=False,
                                           dropout=net_init_params['dropout'],
                                           )  # Actual net to be used

        else:
            model = MixedInputMLPCoxTime(in_features=net_init_params['d_in_x'],
                                              num_nodes=net_init_params['layers'],
                                              batch_norm=False,
                                              dropout=net_init_params['dropout'],
                                              num_embeddings=dataloader.dataset.unique_cat_cols,
                                              embedding_dims=[el // 2 + 1 for el in
                                                              dataloader.dataset.unique_cat_cols]
                                              )  # Actual net to be used
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))
        wrapper = CoxTime(model, tt.optim.Adam)

    elif net_type == 'deepsurv_benchmark':
        if cat_cols_nr == 0:
            model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                                 num_nodes=net_init_params['layers'],
                                                 batch_norm=False,
                                                 dropout=net_init_params['dropout'],
                                                 out_features=1)  # Actual net to be used
        else:
            model = tt.practical.MixedInputMLP(
                in_features=net_init_params['d_in_x'],
                num_nodes=net_init_params['layers'],
                batch_norm=False,
                dropout=net_init_params['dropout'],
                out_features=1,
                num_embeddings=dataloader.dataset.unique_cat_cols,
                embedding_dims=[el // 2 + 1 for el in
                                dataloader.dataset.unique_cat_cols]
            )
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))
        wrapper = CoxPH(model, tt.optim.Adam)

    elif net_type == 'deephit_benchmark':
        labtrans = LabTransDiscreteTime(num_dur)
        # y_input = dataloader.dataset.duration_mapper.inverse_transform(dataloader.dataset.train_y.squeeze().numpy().reshape(-1,1))
        y_train = (dataloader.dataset.train_y.squeeze().numpy(), dataloader.dataset.train_delta.squeeze().numpy())
        labtrans.fit(y_train[0], y_train[1])
        if cat_cols_nr == 0:
            model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                                 num_nodes=net_init_params['layers'],
                                                 batch_norm=False,
                                                 dropout=net_init_params['dropout'],
                                                 out_features=labtrans.out_features)  # Actual net to be used
        else:
            model = tt.practical.MixedInputMLP(in_features=net_init_params['d_in_x'],
                                                    num_nodes=net_init_params['layers'],
                                                    batch_norm=False,
                                                    dropout=net_init_params['dropout'],
                                                    out_features=labtrans.out_features,
                                                    num_embeddings=dataloader.dataset.unique_cat_cols,
                                                    embedding_dims=[el // 2 + 1 for el in
                                                                    dataloader.dataset.unique_cat_cols]
                                                )  # Actual net to be used
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt',map_location=device))
        wrapper = DeepHitSingle(model, tt.optim.Adam, alpha=0.5,
                                     sigma=0.5, duration_index=labtrans.cuts)
    elif net_type == 'cox_CC_benchmark':
        if cat_cols_nr == 0:
            model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                                 num_nodes=net_init_params['layers'],
                                                 batch_norm=False,
                                                 dropout=net_init_params['dropout'],
                                                 out_features=1)  # Actual net to be used
        else:
            model = tt.practical.MixedInputMLP(
                in_features=net_init_params['d_in_x'],
                num_nodes=net_init_params['layers'],
                batch_norm=False,
                dropout=net_init_params['dropout'],
                out_features=1,
                num_embeddings=dataloader.dataset.unique_cat_cols,
                embedding_dims=[el // 2 + 1 for el in
                                dataloader.dataset.unique_cat_cols]
            )
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt', map_location=device))
        wrapper = CoxCC(model, tt.optim.Adam)

    elif net_type == 'cox_linear_benchmark':
        if cat_cols_nr == 0:
            model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                                 num_nodes=[],
                                                 batch_norm=False,
                                                 dropout=net_init_params['dropout'],
                                                 out_features=1)  # Actual net to be used
        else:
            model = tt.practical.MixedInputMLP(
                in_features=net_init_params['d_in_x'],
                num_nodes=[],
                batch_norm=False,
                dropout=net_init_params['dropout'],
                out_features=1,
                num_embeddings=dataloader.dataset.unique_cat_cols,
                embedding_dims=[el // 2 + 1 for el in
                                dataloader.dataset.unique_cat_cols]
            )
        model.load_state_dict(torch.load(PATH + f'best_model_{best_tid}.pt', map_location=device))
        wrapper = CoxPH(model, tt.optim.Adam)

    if net_type in ['cox_time_benchmark', 'deepsurv_benchmark', 'cox_CC_benchmark', 'cox_linear_benchmark',
                         'deephit_benchmark']:
        y_train = (dataloader.dataset.train_y.squeeze().numpy(), dataloader.dataset.train_delta.squeeze().numpy())
        if cat_cols_nr==0:
            X_train = dataloader.dataset.train_X.numpy()
            X, y, delta = dataloader.dataset.test_X.numpy(), dataloader.dataset.test_y.squeeze().numpy(), dataloader.dataset.test_delta.squeeze().numpy()

        else:
            X_train = tt.tuplefy((dataloader.dataset.train_X.numpy(),dataloader.dataset.train_cat_X.numpy()))
            X, y, delta = tt.tuplefy((dataloader.dataset.test_X.numpy(),dataloader.dataset.test_cat_X.numpy())), dataloader.dataset.test_y.squeeze().numpy(), dataloader.dataset.test_delta.squeeze().numpy()

        l_obj=ApproximateLikelihood(wrapper,X,y,delta,1000,half_width=half_width,dataset=dataloader.dataset)
        ll = l_obj.get_approximated_likelihood(input_dat=X_train,target_dat=y_train)
    else:
        y_train = (dataloader.dataset.train_y.squeeze().numpy(), dataloader.dataset.train_delta.squeeze().numpy())
        X_train = dataloader.dataset.train_X.numpy()
        y,delta,surv_df=sumo_loop(model,dataloader,device,100)
        l_obj=ApproximateLikelihood(model, dataloader.dataset.test_X.numpy(), y.numpy(), delta.numpy(), 1000, half_width=half_width,dataset=dataloader.dataset)
        ll=l_obj.get_approximated_likelihood(input_dat=X_train,target_dat=y_train,surv_df_raw=surv_df.transpose())
    return ll

def get_best_params(path,selection_criteria,model,dataset,fold,seed,half_width,device,CRIT):
    if selection_criteria in ['test_loglikelihood','test_loglikelihood_1','test_loglikelihood_2','test_loss']:
        reverse = False
    elif selection_criteria == 'test_conc':
        reverse = True
    elif selection_criteria == 'test_ibs':
        reverse = False
    elif selection_criteria == 'test_inll':
        reverse = False
    trials = pickle.load(open(path+'hyperopt_database.p', "rb"))
    best_trial = sorted(trials.trials, key=lambda x: x['result'][selection_criteria], reverse=reverse)[0]
    net_params = best_trial['result']['net_init_params']
    best_tid = best_trial['tid']
    dur=0
    if model=='deephit_benchmark':
        dur=DURS[best_trial['misc']['vals']['num_dur'][0]]
    print('best_param_for ', selection_criteria)
    output = [-best_trial['result'][x] if x in ['test_loss','test_inll'] else best_trial['result'][x] for x in CRIT]

    if 'test_loss' in CRIT:
        if not (model in ['survival_net_basic','weibull_net','lognormal_net']):
            ll = get_likelihoods(
                PATH=path,
                best_tid=best_tid,
            net_init_params=net_params,
            dataset_string=dataset,
            fold_idx=fold,
            seed=seed,
            device=device,
            num_dur=dur,
            half_width=half_width
            )
            output[0]=ll
    return output


if __name__ == '__main__':
    folder = 'sumo_net_flchain_1_results'
    # folder = 'weibull_rerun_4_results'
    objective = ['S_mean']
    criteria =['test_loss','test_conc','test_ibs','test_inll']
    # model = ['deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark','cox_time_benchmark','survival_net_basic']
    # selection_criteria_list = ['test_loss','test_loss','test_loss','test_conc','test_loss','test_loss']
    # model = ['lognormal_net','weibull_net']
    # selection_criteria_list = ['test_loss','test_loss']
    model = ['survival_net_basic']
    selection_criteria_list = ['test_loss']
    result_name = f'{folder}_results'
    # for index_indicator,(criteria,criteria_labels) in enumerate(zip([['test_loss','test_inll'],['test_conc','test_ibs']],[['likelihood', 'IBLL'],[r'$C^\text{td}$', 'IBS',]])):
    for index_indicator,(criteria,criteria_labels) in enumerate(zip([['test_loss'],['test_inll'],['test_conc'],['test_ibs']],[['likelihood'], ['IBLL'],[r'$C^\text{td}$'], ['IBS',]])):
        cols = ['objective', 'model', 'dataset']
        for criteria_name in criteria:
            cols.append(criteria_name+'_mean')
            cols.append(criteria_name+'_std')
        # dataset_indices = [0,1,2,3]
        dataset_indices = [3]
        # dataset_indices = [4]
        if criteria_labels[0]=='likelihood':
            half_width_list = [1, 2, 4, 8]
        else:
            half_width_list = [0]
        best_device = f'cuda:{GPUtil.getFirstAvailable()[0]}'
        for half_width in half_width_list:
            df = []
            for net_type,selection_criteria in zip(model,selection_criteria_list):
                for o in objective:
                    for d in dataset_indices:
                        d_str = datasets[d]
                        row = [o,net_type,d_str]
                        desc_df = []
                        for s in [4]:
                            for f_idx in [0,1,2,3,4]:
                                try:
                                    pat =f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/'
                                    vals = get_best_params(pat,selection_criteria,model=net_type,dataset=d_str,fold=f_idx,seed=s,device=best_device,half_width=half_width,CRIT=criteria)
                                    desc_df.append(vals)
                                except Exception as e:
                                    print(e)
                        tmp = pd.DataFrame(desc_df,columns =criteria)
                        tmp = tmp.describe()
                        means = tmp.iloc[1,:].values.tolist()
                        stds = tmp.iloc[2,:].values.tolist()
                        desc_df = []
                        for i in range(len(criteria)):
                            desc_df.append(round(means[i],3))
                            desc_df.append(round(stds[i],3))
                        row = row + desc_df
                        df.append(row)
            all_jobs = pd.DataFrame(df,columns=cols)
            piv_df  = pd.DataFrame()
            piv_df['Method'] = all_jobs['objective'].apply(lambda x: x.replace('_','-')) +': '+ all_jobs['model'].apply(lambda x: x.replace('_','-'))
            piv_df['dataset'] = all_jobs['dataset'].apply(lambda x: x.upper())
            for crit,new_crit in zip(criteria,criteria_labels):
                mean_col = crit+'_mean'
                std_col = crit+'_std'
                piv_df[new_crit] = '$'+ all_jobs[mean_col].astype(str)+'\pm'+ all_jobs[std_col].astype(str)+'$'

            final_ = pd.pivot(piv_df,index='Method',columns='dataset')
            print(final_)
            print(final_.to_latex(buf=f"{result_name}_half_width={half_width}_index={index_indicator}.tex",escape=False))
            final_.to_csv(f"{result_name}_half_width={half_width}_index={index_indicator}.csv")





