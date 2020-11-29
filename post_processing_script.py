from serious_run import *

def post_processing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--seeds', type=int, nargs='?', help='selects the seed to split the data on')
    return parser

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
    print(best_trial)

if __name__ == '__main__':
    objective = ['S_mean','hazard_mean']
    criteria =['test_loglikelihood','test_conc','test_ibs','test_inll']
    c = criteria[1]
    o = objective[0]
    args = vars(post_processing_parser().parse_args())
    d_str = datasets[args['dataset']]
    seeds = [i+1 for i in range(args['seeds'])]
    df = []
    for s in seeds:
        load_str = f'./{d_str}_seed={s}_objective={o}/best_results.csv'
        # load_str = f'./{d_str}_{s}/best_results.csv'
        mini_df = pd.read_csv(load_str)
        df.append(mini_df)
        get_best_params(f'./{d_str}_{s}/hyperopt_database.p',c)
    df = pd.concat(df)
    print(df.describe())

