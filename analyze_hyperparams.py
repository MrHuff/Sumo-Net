import pandas as pd
from serious_run import datasets
import pickle

def load_database_into_df(path,selection_criteria):
    if selection_criteria == 'test_loglikelihood':
        reverse = False
    elif selection_criteria == 'test_conc':
        reverse = True
    elif selection_criteria == 'test_ibs':
        reverse = False
    elif selection_criteria == 'test_inll':
        reverse = False
    trials = pickle.load(open(path, "rb"))
    results = trials.results
    print(results[0]['loss'])

folder = 'ibs_eval'
objective = ['S_mean', 'hazard_mean']
criteria = ['test_loglikelihood', 'test_conc', 'test_ibs', 'test_inll']
model = ['survival_net_basic']
result_name = 'survival_net_ibs_eval'
c = criteria[2]
cols = ['objective', 'model', 'dataset']



for o in objective:
    for net_type in model:
        for d in [0, 1, 2, 3]:
            d_str = datasets[d]
            row = [o, net_type, d_str]
            desc_df = []
            for s in [1337]:
                for f_idx in [0, 1, 2, 3, 4]:
                    # get_best_params(f'./{d_str}_{s}/hyperopt_database.p',c)
                    vals = load_database_into_df(
                        f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/hyperopt_database.p', c)


