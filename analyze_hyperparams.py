from serious_run import *
from hyperopt import space_eval
def generate_h_param_space(dataset):
    d_x = dataset_d_x[dataset]
    w_x = dataset_w_x[dataset]
    d = dataset_d[dataset]
    w = dataset_w[dataset]
    d_t = dataset_d_t[dataset]
    bs = dataset_bs[dataset]
    
    hyper_param_space = {
        # torch.nn.functional.elu,torch.nn.functional.relu,
        'bounding_op': [square],  # torch.sigmoid, torch.relu, torch.exp,
        'transformation': [torch.nn.functional.tanh],
        'depth_x': d_x,
        'width_x': w_x,
        'depth': d,
        'width': w,
        'depth_t': d_t,
        'width_t': [1,1,1,1,1],  # ads
        'bs': bs,
        'lr': [1e-3,1e-2,1e-1],
        'direct_dif': [False],
        'dropout': [0.0,0.1,0.2,0.3,0.4,0.5],
        'eps': [1e-3,1e-4,1e-5],
        'weight_decay': [1e-3,1e-4,1e-2,0.1,0]

    }

    h_params_list = ['bounding_op', 'transformation', 'depth_x', 'width_x', 'depth_t', 'width_t', 'depth', 'width', 'bs', 'lr',
     'direct_dif', 'dropout', 'eps', 'weight_decay']
    hyperparameter_space = {}
    for string in h_params_list:
        hyperparameter_space[string] = hp.choice(string, hyper_param_space[string])
    return hyperparameter_space

def load_database_into_df(path,fold_idx,dataset,o,net_type):

    T = pickle.load(open(path, "rb"))
    results = T.results
    trials = T.trials
    hspace = generate_h_param_space(dataset)
    data = []
    for el,t in zip(results,trials):
        val = t['misc']['vals']
        for key,item in val.items():
            val[key] = item[0]
        actual_h_params = space_eval(hspace,val)
        row = dict(el, **actual_h_params)
        row['fold_idx'] = fold_idx
        row['dataset'] = dataset
        row['o'] = o
        row['net_type'] = net_type
        del row['net_init_params'],row['bounding_op'],row['transformation']
        data.append(row)
    df  = pd.DataFrame(data)
    return df

folder = 'ibs_eval'
objective = ['S_mean', 'hazard_mean']
criteria = ['test_loglikelihood', 'test_conc', 'test_ibs', 'test_inll']
model = ['survival_net_basic']
result_name = 'survival_net_ibs_eval'
c = criteria[2]
cols = ['objective', 'model', 'dataset']

if __name__ == '__main__':
    concat  = []
    for o in objective:
        for net_type in model:
            for d in [0, 1, 2, 3]:
                d_str = datasets[d]
                row = [o, net_type, d_str]
                desc_df = []
                for s in [1337]:
                    for f_idx in [0, 1, 2, 3, 4]:
                        # get_best_params(f'./{d_str}_{s}/hyperopt_database.p',c)
                        vals = load_database_into_df(f'./{folder}/{d_str}_seed={s}_fold_idx={f_idx}_objective={o}_{net_type}/hyperopt_database.p', f_idx,d,o,net_type)
                        concat.append(vals)

    res = pd.concat(concat)
    res.to_csv(f"{result_name}_analysis.csv")


