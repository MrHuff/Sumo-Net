from serious_run import *

# def post_processing_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
#     parser.add_argument('--seeds', type=int, nargs='?', help='selects the seed to split the data on')
#     return parser

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
    objective = ['S_mean','hazard_mean']
    criteria =['test_loglikelihood','test_conc','test_ibs','test_inll']
    model = ['ocean_net','survival_net']
    c = criteria[0]

    cols = ['objective','model','dataset']
    for criteria_name in criteria:
        cols.append(criteria_name+'_mean')
        cols.append(criteria_name+'_std')
    df = []
    for o in objective:
        for net_type in model:
            for d in [0,1,2,3]:
                d_str = datasets[d]
                row = [o,net_type,d_str]
                desc_df = []
                for s in [1,2,3,4,5]:

                    # get_best_params(f'./{d_str}_{s}/hyperopt_database.p',c)
                    vals = get_best_params(f'./{d_str}_seed={s}_objective={o}_{net_type}/hyperopt_database.p',c)
                    desc_df.append(vals)
                tmp = pd.DataFrame(desc_df,columns =criteria)
                tmp = tmp.describe()
                means = tmp.iloc[1,:].values.tolist()
                stds = tmp.iloc[2,:].values.tolist()
                desc_df = []
                for i in range(len(criteria)):
                    desc_df.append(means[i])
                    desc_df.append(stds[i])
                row = row + desc_df
                df.append(row)
    all_jobs = pd.DataFrame(df,columns=cols)
    print(all_jobs)
    all_jobs.to_csv('all_jobs.csv')
    print(all_jobs.to_latex())
