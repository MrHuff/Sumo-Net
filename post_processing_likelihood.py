from serious_run import *
from astropy.table import Table

def get_best_params(path,selection_criteria):
    if selection_criteria in ['test_loglikelihood_1','test_loglikelihood_2','test_loss']:
        reverse = False
    elif selection_criteria == 'test_conc':
        reverse = True
    elif selection_criteria == 'test_ibs':
        reverse = False
    elif selection_criteria == 'test_inll':
        reverse = False
    trials = pickle.load(open(path, "rb"))
    best_trial = sorted(trials.results, key=lambda x: x[selection_criteria], reverse=reverse)[0]
    print('best_param_for ', selection_criteria)
    return [best_trial[x] for x in ['test_loglikelihood_1','test_conc','test_ibs','test_inll']]


if __name__ == '__main__':
    folder = 'weibull_rerun_4_results'
    objective = ['S_mean']
    criteria =['test_loss','test_conc','test_ibs','test_inll']
    # model = ['survival_net_basic','cox_time_benchmark','deepsurv_benchmark','cox_CC_benchmark','cox_linear_benchmark','deephit_benchmark']
    # c_list = [0,0,0,0,0,1]
    model = ['weibull_net']
    c_list = [0]

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
                            vals = get_best_params(f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/hyperopt_database.p',criteria[c])
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





