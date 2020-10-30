from pycox.datasets import kkbox,support,metabric,gbsg,flchain
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from .toy_data_generation import toy_data_class

class surival_dataset(Dataset):
    def __init__(self,str_identifier,seed=1337):

        super(surival_dataset, self).__init__()
        if str_identifier=='support':
            data = support
        elif str_identifier=='metabric':
            data = metabric
        elif str_identifier=='gbsg':
            data = gbsg
        elif str_identifier == 'flchain':
            data = flchain
        elif str_identifier=='kkbox':
            data = kkbox
        elif str_identifier=='weibull':
            data = toy_data_class(str_identifier)
        elif str_identifier=='checkboard':
            data = toy_data_class(str_identifier)
        elif str_identifier=='normal':
            data = toy_data_class(str_identifier)

        self.event_col = data.col_event
        self.duration_col = data.col_duration
        df_train = data.read_df()
        df_test = df_train.sample(frac=0.2,random_state=seed)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.25,random_state=seed)
        df_train = df_train.drop(df_val.index)

        self.split(df_train,'train')
        self.split(df_val,'val')
        self.split(df_test,'test')
        self.set('train')

    def split(self,df,mode='train'):

        setattr(self,f'{mode}_delta', torch.from_numpy(df[self.event_col].values).float())
        setattr(self,f'{mode}_y', torch.from_numpy(df[self.duration_col].values).unsqueeze(-1).float())
        setattr(self, f'{mode}_X', torch.from_numpy(df.drop([self.event_col, self.duration_col], axis=1).values).float())

    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        self.delta = getattr(self,f'{mode}_delta')

    def __getitem__(self, index):
        return self.X[index,:],self.y[index],self.delta[index]

    def __len__(self):
        return self.X.shape[0]

def get_dataloader(str_identifier,bs,seed):
    d = surival_dataset(str_identifier,seed)
    dat = DataLoader(dataset=d,batch_size=bs,shuffle=True)
    return dat
