from utils.dataloaders import *


datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            'weibull',
            'checkboard',
            'normal'
            ]

if __name__ == '__main__':
    for s in datasets:
        surival_dataset(str_identifier=s)


