import pickle
from serious_run import *
def get_best_model_param(load_path,selection_criteria):
    trials = pickle.load(open(load_path + 'hyperopt_database.p',
                              "rb"))

    if selection_criteria == 'train':
        reverse = False
    elif selection_criteria == 'concordance':
        reverse = True
    elif selection_criteria == 'ibs':
        reverse = False
    elif selection_criteria == 'inll':
        reverse = False
    best_trial = sorted(trials.results, key=lambda x: x['test_loss'], reverse=reverse)[0]  # low to high
    net_init_params = best_trial['net_init_params']
    print(best_trial)
    print(net_init_params)

if __name__ == '__main__':
    dataset_id = 5
    bs = 500
    dataset_string = datasets[dataset_id]
    # load_path = f'./{dataset_string}_{s}/'
    # get_best_model_param(load_path, 'train')
    for s in [1,2,3,4]:
        load_path = f'./{dataset_string}_{s}/'
        get_best_model_param(load_path, 'train')