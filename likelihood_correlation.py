import pickle
import pandas as pd
datasets = [
            'support',
            'metabric',
            'gbsg',
            'flchain'
            ]
if __name__ == '__main__':
    d = 'base_run_results'
    seed = 1337
    df = []
    for ds in datasets:
        for fold_idx in [0,1,2,3,4]:
            path = f'{d}/{ds}_seed={seed}_fold_idx={fold_idx}_objective=S_mean_survival_net_basic/hyperopt_database.p'
            trials = pickle.load(open(path, "rb"))
            for res in trials.results:
                df.append([-res['test_loglikelihood_2'],res['test_conc'],-res['test_ibs'],-res['test_inll']])
    df_res = pd.DataFrame(df, columns=['likelihood', 'concordance', 'ibs', 'ibl'])
    cor_mat = df_res.corr(method='pearson')
    cor_mat.to_latex(f'{d}.tex')


# best_trial = sorted(trials.results, key=lambda x: x[selection_criteria], reverse=reverse)[0]
