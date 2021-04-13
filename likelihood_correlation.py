import pickle
import pandas as pd

if __name__ == '__main__':
    dirnames = ['cox_time_conc_results','cox_time_ibl_results','cox_time_ibs_results']
    for d in dirnames:
        path = f'{d}/support_seed=1337_fold_idx=0_objective=S_mean_cox_time_benchmark/hyperopt_database.p'
        reverse = False
        selection_criteria = 'test_loss'
        trials = pickle.load(open(path, "rb"))
        df = []
        for res in trials.results:
            df.append([res['test_loglikelihood_1'],res['test_conc'],res['test_ibs'],-res['test_inll']])
        df_res = pd.DataFrame(df,columns = ['likelihood','concordance','ibs','ibl'])
        # print(df_res)
        # print(df_res.corr(method='spearman'))
        cor_mat = df_res.corr(method='spearman')
        cor_mat.to_latex(f'{d}.tex')


# best_trial = sorted(trials.results, key=lambda x: x[selection_criteria], reverse=reverse)[0]
