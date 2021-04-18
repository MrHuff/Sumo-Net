from serious_run import *
from astropy.table import Table


def get_likelihoods(net_init_params,dataset_string,seed,fold_idx,device='cuda:0',num_dur=0):
    dataloader = get_dataloader(dataset_string,5000, seed, fold_idx)
    if net_type == 'survival_net':
        model = survival_net(**net_init_params).to(device)
    elif net_type == 'survival_net_basic':
        model = survival_net_basic(**net_init_params).to(device)
    elif net_type == 'cox_time_benchmark':
        model = MLPVanillaCoxTime(in_features=net_init_params['d_in_x'],
                                       num_nodes=net_init_params['layers'],
                                       batch_norm=False,
                                       dropout=net_init_params['dropout'],
                                       )  # Actual net to be used
        wrapper = CoxTime(model, tt.optim.Adam)

    elif net_type == 'deepsurv_benchmark':
        model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                             num_nodes=net_init_params['layers'],
                                             batch_norm=False,
                                             dropout=net_init_params['dropout'],
                                             out_features=1)  # Actual net to be used
        wrapper = CoxPH(model, tt.optim.Adam)
    elif net_type == 'deephit_benchmark':
        print('num_dur', num_dur)
        labtrans = LabTransDiscreteTime(num_dur)
        model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                        num_nodes=net_init_params['layers'],
                                        batch_norm=False,
                                        dropout=net_init_params['dropout'],
                                        out_features=labtrans.out_features)  # Actual net to be used
        wrapper = DeepHitSingle(model, tt.optim.Adam, alpha=0.5,
                                sigma=0.5, duration_index=labtrans.cuts)

    elif net_type == 'cox_CC_benchmark':
        model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                             num_nodes=net_init_params['layers'],
                                             batch_norm=False,
                                             dropout=net_init_params['dropout'],
                                             out_features=1)  # Actual net to be used
        wrapper = CoxCC(model, tt.optim.Adam)
    elif net_type == 'cox_linear_benchmark':
        model = tt.practical.MLPVanilla(in_features=net_init_params['d_in_x'],
                                             num_nodes=[],
                                             batch_norm=False,
                                             dropout=net_init_params['dropout'],
                                             out_features=1)  # Actual net to be used
        wrapper = CoxPH(model, tt.optim.Adam)

    if net_type in ['cox_time_benchmark', 'deepsurv_benchmark', 'cox_CC_benchmark', 'cox_linear_benchmark',
                         'deephit_benchmark']:
        y_test = (
        dataloader.dataset.test_y.squeeze().numpy(), dataloader.dataset.test_delta.squeeze().numpy())



        test_data = tt.tuplefy(dataloader.dataset.test_X.numpy(), y_test)
        test_durations = dataloader.dataset.invert_duration(dataloader.dataset.test_y.numpy()).squeeze()
        with torch.no_grad():
            val_likelihood_list = [1e99, 1e99]
            test_likelihood_list = [1e99, 1e99]
            class_list = []
            general_class = general_likelihood(wrapper)
            class_list.append(general_class)
            if net_type != 'deephit_benchmark':
                hazard_class = HazardLikelihoodCoxTime(wrapper)
                class_list.append(hazard_class)
            for i, coxL in enumerate(class_list):

                test_likelihood = coxL.estimate_likelihood(torch.from_numpy(test_data[0]),
                                                           torch.from_numpy(test_data[1][0]),
                                                           torch.from_numpy(test_data[1][1]))
                test_likelihood_list[i] = test_likelihood.item()


def get_best_params(path,selection_criteria,model,dataset,fold,seed):
    if selection_criteria in ['test_loglikelihood_1','test_loglikelihood_2','test_loss']:
        reverse = False
    elif selection_criteria == 'test_conc':
        reverse = True
    elif selection_criteria == 'test_ibs':
        reverse = False
    elif selection_criteria == 'test_inll':
        reverse = False
    trials = pickle.load(open(path, "rb"))
    best_trial = sorted(trials.trials, key=lambda x: x['result'][selection_criteria], reverse=reverse)[0]
    print('best_param_for ', selection_criteria)
    output = [best_trial['result'][x] for x in ['test_conc','test_ibs','test_inll']]
    return output


if __name__ == '__main__':
    folder = 'likelihood_jobs_3_results'
    objective = ['S_mean']
    criteria =['test_loss','test_conc','test_ibs','test_inll']
    model = ['survival_net_basic','cox_time_benchmark','deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark']
    c_list = [0,0,0,0,0,1]
    result_name = f'{folder}_results'
    cols = ['objective','model','dataset']
    for criteria_name in criteria:
        cols.append(criteria_name+'_mean')
        cols.append(criteria_name+'_std')
    df = []
    dataset_indices = [0,1,2,3]
    for o in objective:
        for net_type,c in zip(model,c_list):
            for d in dataset_indices:
                d_str = datasets[d]
                row = [o,net_type,d_str]
                desc_df = []
                for s in [1337]:
                    for f_idx in [0,1,2,3,4]:
                        try:
                            pat =f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/hyperopt_database.p'
                            vals = get_best_params(pat,criteria[c])
                            desc_df.append(vals)
                        except Exception as e:
                            print(e)
                tmp = pd.DataFrame(desc_df,columns =criteria)
                tmp = tmp.describe()
                means = tmp.iloc[1,:].values.tolist()
                stds = tmp.iloc[2,:].values.tolist()
                desc_df = []
                for i in range(len(criteria)):
                    if i==3:
                        desc_df.append(-round(means[i],3))
                    else:
                        desc_df.append(round(means[i],3))
                    desc_df.append(round(stds[i],3))
                row = row + desc_df
                df.append(row)
    all_jobs = pd.DataFrame(df,columns=cols)
    piv_df  = pd.DataFrame()
    piv_df['Method'] = all_jobs['objective'].apply(lambda x: x.replace('_','-')) +': '+ all_jobs['model'].apply(lambda x: x.replace('_','-'))
    piv_df['dataset'] = all_jobs['dataset'].apply(lambda x: x.upper())
    for crit,new_crit in zip(criteria,['likelihood',r'$C^\text{td}$','IBS','IBLL']):
        mean_col = crit+'_mean'
        # std_col = crit+'_std'
        piv_df[new_crit] = '$'+ all_jobs[mean_col].astype(str)+'$'#'\pm '+ all_jobs[std_col].astype(str)+'$'

    final_ = pd.pivot(piv_df,index='Method',columns='dataset')
    print(final_)
    print(final_.to_latex(buf=f"{result_name}.tex",escape=False))
    final_.to_csv(f"{result_name}.csv")





