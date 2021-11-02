import pickle
import pandas as pd
from functools import reduce

datasets = [
            'support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox'
            ]
if __name__ == '__main__':
    d = 'sumo_net_30_results'
    seed = 1337
    df = []
    corr_mats = []
    for ds in datasets:
        for fold_idx in [0,1,2,3,4]:
            path = f'{d}/{ds}_seed={seed}_fold_idx={fold_idx}_objective=S_mean_survival_net_basic/hyperopt_database.p'
            trials = pickle.load(open(path, "rb"))
            for res in trials.results:
                df.append([-res['test_loglikelihood_2'],res['test_conc'],-res['test_ibs'],-res['test_inll']])
        df_res = pd.DataFrame(df, columns=['likelihood', 'concordance', 'ibs', 'ibl'])
        cor_mat = df_res.corr(method='spearman')
        corr_mats.append(cor_mat)
    len_avg = len(corr_mats)
    d = reduce(lambda x, y: x.add(y, fill_value=0), corr_mats)/len_avg

    d.to_latex(f'correlation.tex')


# best_trial = sorted(trials.results, key=lambda x: x[selection_criteria], reverse=reverse)[0]
