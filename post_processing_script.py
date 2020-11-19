from serious_run import *

def post_processing_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, nargs='?', default=-1, help='which dataset to run')
    parser.add_argument('--seeds', type=int, nargs='?', help='selects the seed to split the data on')
    return parser

if __name__ == '__main__':
    args = vars(post_processing_parser().parse_args())
    d_str = datasets[args['dataset']]
    seeds = [i+1 for i in range(args['seeds'])]
    df = []
    for s in seeds:
        load_str = f'./{d_str}_{s}/best_results.csv'
        mini_df = pd.read_csv(load_str)
        df.append(mini_df)
    df = pd.concat(df)
    print(df.describe())

