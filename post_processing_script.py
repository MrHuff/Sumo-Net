from serious_run import *
from astropy.table import Table
tab1 = Table.read('kvamme_conc.tex').to_pandas()
tab2 = Table.read('kvamme_ibs.tex').to_pandas()
tab3 = Table.read('kvamme_inll.tex').to_pandas()
tab4 = Table.read('kvamme_kkbox.tex').to_pandas()
tab4['dataset'] = 'KKBOX'
tab1 = tab1.melt(id_vars=['Method'])
tab2 = tab2.melt(id_vars=['Method'])
tab3 = tab3.melt(id_vars=['Method'])
tab = pd.DataFrame()
tab['Method'] = tab1['Method']
tab['dataset'] = tab1['variable']
tab[r'$C^\text{td}$'] = tab1['value'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
tab['IBS'] = tab2['value'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
tab['IBLL'] = tab3['value'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
tab4[r'$C^\text{td}$'] =tab4[r'$C^\text{td}$'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
tab4['IBS'] = tab4['IBS'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
tab4['IBLL'] = tab4['IBLL'].apply(lambda x: x if x=='NaN' else '$'+str(x)+'$')
# tab = tab.append(tab4)
print(tab)


# def post_processing_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
#     parser.add_argument('--seeds', type=int, nargs='?', help='selects the seed to split the data on')
#     return parser
# Finish analysis tmrw!
def get_best_params(path,selection_criteria):
    if selection_criteria == 'test_loglikelihood':
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
    return [best_trial[x] for x in ['test_loglikelihood','test_conc','test_ibs','test_inll']]


if __name__ == '__main__':
    folder = 'conc_test'
    objective = ['S_mean']
    criteria =['test_loglikelihood','test_conc','test_ibs','test_inll']
    model = ['survival_net_basic']
    result_name = f'{folder}_results'
    c = criteria[2]
    cols = ['objective','model','dataset']
    for criteria_name in criteria:
        cols.append(criteria_name+'_mean')
        cols.append(criteria_name+'_std')
    df = []
    dataset_indices = [1,]
    for o in objective:
        for net_type in model:
            for d in dataset_indices:
                d_str = datasets[d]
                row = [o,net_type,d_str]
                desc_df = []
                for s in [1337]:
                    for f_idx in [0,1,2,3,4]:
                        try:
                    # get_best_params(f'./{d_str}_{s}/hyperopt_database.p',c)
                            vals = get_best_params(f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/hyperopt_database.p',c)
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
    for crit,new_crit in zip(criteria[1:],[r'$C^\text{td}$','IBS','IBLL']):
        mean_col = crit+'_mean'
        # std_col = crit+'_std'
        piv_df[new_crit] = '$'+ all_jobs[mean_col].astype(str)+'$'#'\pm '+ all_jobs[std_col].astype(str)+'$'

    if not (dataset_indices==[5,6,7]) and not (folder in ['ablation_results']):
        if dataset_indices==[4]:
            piv_df = piv_df.append(tab4)
        else:
            piv_df = piv_df.append(tab)
    final_ = pd.pivot(piv_df,index='Method',columns='dataset')
    print(final_)
    print(final_.to_latex(buf=f"{result_name}.tex",escape=False))
    final_.to_csv(f"{result_name}.csv")





