import torch

from serious_run import *
from astropy.table import Table
from utils.hazard_model_likelihood import *
DURS = [50, 100, 200, 400]

def calculate_t_p(model,X,x_cat,y,p):
    y_var = torch.nn.Parameter(torch.rand_like(y),requires_grad=True)
    ref_output = torch.ones_like(y)*p
    loss_func = torch.nn.L1Loss()
    opt = torch.optim.AdamW([y_var],lr=1e-1)
    lrs = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=0.5,patience=10)
    for i in range(250):
        F =1- model.forward_S_eval(X, y_var.exp(), x_cat)  # Fix
        loss = loss_func(F.squeeze(),ref_output.squeeze())
        # print(loss)
        if loss.item()<1e-4:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
        lrs.step(loss)
    return y_var.exp().detach().cpu()

def loop_over_p(model,dataloader,p_list, device='cuda:0'):
    model.eval()
    dataloader.dataset.set('test')
    T_list = []
    T_p_list = []
    for i, (X, x_cat, y, delta, s_kmf) in enumerate(tqdm(dataloader)):
        mask = delta == 1
        X_in = X[~mask, :].to(device)
        y_in = y[~mask, :].to(device)
        if not isinstance(x_cat, list):
            x_cat_in = x_cat[~mask, :].to(device)
        else:
            x_cat_in = x_cat
        tmp_T_p_list = []
        for p in p_list:
            F = calculate_t_p(model,X_in,x_cat_in,y_in,p)
            tmp_T_p_list.append(F)
        T_p_list.append(torch.cat(tmp_T_p_list,dim=1))
        T_list.append(y_in.cpu())
    T_p = torch.cat(T_p_list,dim=0)
    T=torch.cat(T_list,dim=0)

    return T,T_p

def raw_df_to_calib(S_df,p_list = [0.05*i for i in range(1,20)]):
    pass


def get_calibration(PATH, best_tid, net_init_params, dataset_string, seed, fold_idx, half_width, device='cuda:0', num_dur=0):
    sumo_net = net_type in ['survival_net_basic', 'deepsurv_benchmark', 'cox_CC_benchmark', 'cox_linear_benchmark',
                         'deephit_benchmark']
    dataloader = get_dataloader(dataset_string,5000, seed, fold_idx,shuffle=False,sumo_net=sumo_net)
    cat_cols_nr = len(dataloader.dataset.unique_cat_cols)
    p_list = [0.1*i for i in range(1,11)]
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
            delta = dataloader.dataset.test_delta.squeeze().numpy()==1
            X, y = dataloader.dataset.test_X.numpy()[delta,:], dataloader.dataset.test_y.squeeze().numpy()[delta],

        else:
            X_train = tt.tuplefy((dataloader.dataset.train_X.numpy(),dataloader.dataset.train_cat_X.numpy()))
            delta = dataloader.dataset.test_delta.squeeze().numpy()==1
            X, y = tt.tuplefy((dataloader.dataset.test_X.numpy()[delta,:],dataloader.dataset.test_cat_X.numpy()[delta,:])), dataloader.dataset.test_y.squeeze().numpy()[delta]
        wrapper.compute_baseline_hazards(sample=1000,input=X_train,target=y_train)
        F_df = 1- wrapper.predict_surv_df(X)
        pass

        #figure out how to get S from pycox models without big method... Start by writing loop for sumo net
    else:
        T,T_p=loop_over_p(model,dataloader,p_list,device)
        p_calib = (T <= T_p).sum(0) / T.shape[0]
        calib_score = torch.sum((torch.tensor(p_list) - p_calib)**2)
        return calib_score,{'x':p_list,'y':p_calib.numpy()}


def get_load_best_possible_calibration(path,selection_criteria,model,dataset,fold,seed,half_width,device,CRIT):
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

    p_calib_score,plot_data = get_calibration(
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


    return p_calib_score,plot_data

if __name__ == '__main__':
    folder = 'sumo_net_flchain_1_results'
    # folder = 'kkbox_sumo_20_results'
    # folder = 'parametric_basic_results'
    # folder = 'weibull_rerun_4_results'
    objective = ['S_mean']
    criteria =['test_loss','test_conc','test_ibs','test_inll']
    # model = ['deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark','cox_time_benchmark']
    # selection_criteria_list = ['test_loss','test_loss','test_loss','test_conc','test_loss','test_loss']
    # model = ['lognormal_net','weibull_net']
    # selection_criteria_list = ['test_loss','test_loss']
    # model = ['lognormal_net']
    # selection_criteria_list = ['test_loss']
    model = ['survival_net_basic']
    selection_criteria_list = ['test_loss']
    result_name = f'{folder}_results'
    # for index_indicator,(criteria,criteria_labels) in enumerate(zip([['test_loss','test_inll'],['test_conc','test_ibs']],[['likelihood', 'IBLL'],[r'$C^\text{td}$', 'IBS',]])):
    for index_indicator,(criteria,criteria_labels) in enumerate(zip([['test_loss']],[['Calibration score']])):
        cols = ['objective', 'model', 'dataset']
        for criteria_name in criteria:
            cols.append(criteria_name+'_mean')
            cols.append(criteria_name+'_std')
        # dataset_indices = [0,1,2,3]
        dataset_indices = [3]
        best_device = f'cuda:{GPUtil.getFirstAvailable()[0]}'
        df = []
        for net_type,selection_criteria in zip(model,selection_criteria_list):
            for o in objective:
                for d in dataset_indices:
                    d_str = datasets[d]
                    row = [o,net_type,d_str]
                    desc_df = []
                    for s in [3]:
                        for f_idx in [0,1,2,3,4]:
                            try:
                                pat =f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/'
                                vals,plot_data = get_load_best_possible_calibration(pat,selection_criteria,model=net_type,dataset=d_str,fold=f_idx,seed=s,device=best_device,half_width=1,CRIT=criteria)
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
        print(final_.to_latex(buf=f"{result_name}_calib_index={index_indicator}.tex",escape=False))
        final_.to_csv(f"{result_name}_calib_index={index_indicator}.csv")





