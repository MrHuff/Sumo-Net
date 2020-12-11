from pycox.datasets import kkbox,support,metabric,gbsg,flchain
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pycox.preprocessing.feature_transforms import *
import torch
from .toy_data_generation import toy_data_class
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
class surival_dataset(Dataset):
    def __init__(self,str_identifier,seed=1337):

        super(surival_dataset, self).__init__()
        if str_identifier=='support':
            data = support
            cont_cols = ['x0','x3','x7','x8','x9','x10','x11','x12','x13']
            binary_cols = ['x1','x4','x5']
            cat_cols = ['x2','x6']

        elif str_identifier=='metabric':
            data = metabric
            cont_cols = ['x0', 'x1', 'x2', 'x3', 'x8']
            binary_cols = ['x4', 'x5', 'x6', 'x7']
            cat_cols = []

        elif str_identifier=='gbsg':
            data = gbsg
            cont_cols = ['x3','x4','x5','x6']
            binary_cols = ['x0', 'x2']
            cat_cols = ['x1']
        elif str_identifier == 'flchain':
            data = flchain
            cont_cols = ['sample.yr','age','kappa','lambda','creatinine']
            binary_cols = ['sex','mgus']
            cat_cols = ['flc.grp']
        elif str_identifier=='kkbox':
            data = kkbox
            cont_cols = ['n_prev_churns','log_days_between_subs','log_days_since_reg_init','log_payment_plan_days','log_plan_list_price','log_actual_amount_paid','age_at_start']
            binary_cols = ['is_auto_renew','is_cancel','strange_age','nan_days_since_reg_init','no_prev_churns']
            cat_cols = ['city','payment_method_id','gender','registered_via']
        elif str_identifier=='weibull':
            data = toy_data_class(str_identifier)
            cont_cols = ['x1']
            binary_cols = []
            cat_cols = []
        elif str_identifier=='checkboard':
            data = toy_data_class(str_identifier)
            cont_cols = ['x1']
            binary_cols = []
            cat_cols = []
        elif str_identifier=='normal':
            data = toy_data_class(str_identifier)
            cont_cols = ['x1']
            binary_cols = []
            cat_cols = []
        df_train = data.read_df()
        df_train = df_train.dropna()
        if str_identifier=='kkbox':
            self.event_col = 'event'
            self.duration_col = 'duration'
            df_train = df_train.drop(['msno'],axis=1)
        else:
            self.event_col = data.col_event
            self.duration_col = data.col_duration
        c = OrderedCategoricalLong()
        for el in cat_cols:
            df_train[el] = c.fit_transform(df_train[el])
        standardize = [([col], MinMaxScaler()) for col in cont_cols]
        leave = [(col, None) for col in binary_cols]
        self.cat_cols = cat_cols
        self.x_mapper = DataFrameMapper(standardize+leave)
        self.duration_mapper = MinMaxScaler()

        if self.cat_cols:
            self.unique_cat_cols = df_train[cat_cols].max(axis=0).tolist()
            self.unique_cat_cols = [el+1 for el in self.unique_cat_cols]
            for el in cat_cols:
                print(f'column {el}:', df_train[el].unique().tolist())
            print(self.unique_cat_cols)
        else:
            self.unique_cat_cols = []

        df_train, df_test, y_train, y_test = train_test_split(df_train, df_train[self.event_col], test_size = 0.2, random_state = seed,stratify=df_train[self.event_col])
        df_train, df_val, y_train, y_val = train_test_split(df_train, df_train[self.event_col], test_size = 0.2, random_state = seed,stratify=df_train[self.event_col])

        # if str_identifier not in ['gbsg']:
        x_train = self.x_mapper.fit_transform(df_train[cont_cols+binary_cols]).astype('float32')
        x_val = self.x_mapper.transform(df_val[cont_cols+binary_cols]).astype('float32')
        x_test = self.x_mapper.transform(df_test[cont_cols+binary_cols]).astype('float32')
        # else:
        #     x_train =df_train[cont_cols + binary_cols].values.astype('float32')
        #     x_val = df_val[cont_cols + binary_cols].values.astype('float32')
        #     x_test = df_test[cont_cols + binary_cols].values.astype('float32')

        y_train = self.duration_mapper.fit_transform(df_train[self.duration_col].values.reshape(-1,1)).astype('float32')
        y_val = self.duration_mapper.transform(df_val[self.duration_col].values.reshape(-1,1)).astype('float32')
        y_test = self.duration_mapper.transform(df_test[self.duration_col].values.reshape(-1,1)).astype('float32')

        self.split(X=x_train,delta=df_train[self.event_col],y=y_train,mode='train',cat=cat_cols,df=df_train)
        self.split(X=x_val,delta=df_val[self.event_col],y=y_val,mode='val',cat=cat_cols,df=df_val)
        self.split(X=x_test,delta=df_test[self.event_col],y=y_test,mode='test',cat=cat_cols,df=df_test)
        self.set('train')

    def split(self,X,delta,y,cat=[],mode='train',df=[]):

        setattr(self,f'{mode}_delta', torch.from_numpy(delta.astype('float32').values).float())
        setattr(self,f'{mode}_y', torch.from_numpy(y).float())
        setattr(self, f'{mode}_X', torch.from_numpy(X).float())
        if self.cat_cols:
            setattr(self, f'{mode}_cat_X', torch.from_numpy(df[cat].astype('int64').values).long())

    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        self.delta = getattr(self,f'{mode}_delta')
        if self.cat_cols:
            self.cat_X = getattr(self,f'{mode}_cat_X')
        else:
            self.cat_X = []
        self.min_duration = self.y.min().numpy()
        self.max_duration = self.y.max().numpy()

    def transform_x(self,x):
        return self.x_mapper.transform(x)

    def invert_duration(self,duration):
        return self.duration_mapper.inverse_transform(duration)

    def transform_duration(self,duration):
        return self.duration_mapper.transform(duration)

    def __getitem__(self, index):
        if self.cat_cols:
            return self.X[index,:],self.cat_X[index,:],self.y[index],self.delta[index]
        else:
            return self.X[index,:],self.cat_X,self.y[index],self.delta[index]

    def __len__(self):
        return self.X.shape[0]

class chunk_iterator():
    def __init__(self,X,delta,y,cat_X,shuffle,batch_size):
        self.X = X
        self.delta = delta
        self.y = y
        self.cat_X = cat_X
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.valid_cat = not isinstance(self.cat_X, list)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.delta = self.delta[self.perm]
            self.y = self.y[self.perm,:]
            if self.valid_cat: #F
                self.cat_X = self.cat_X[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_delta = torch.chunk(self.delta,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        if self.valid_cat:  # F
            self.it_cat_X = torch.chunk(self.cat_X,self.chunks)
        else:
            self.it_cat_X = []
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            if self.valid_cat:
                result = (self.it_X[self._index],self.it_cat_X[self._index],self.it_y[self._index],self.it_delta[self._index])
            else:
                result = (self.it_X[self._index],[],self.it_y[self._index],self.it_delta[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

class super_fast_iterator():
    def __init__(self,X,delta,y,cat_X,batch_size):
        self.X = X
        self.delta = delta
        self.y = y
        self.cat_X = cat_X
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.valid_cat = not isinstance(self.cat_X, list)
        self._index = 0
        self.rand_range = self.n - self.batch_size - 1
        if self.rand_range<0:
            self.rand_range=1

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.chunks:
            i = np.random.randint(0, self.rand_range)
            i_end = i+self.batch_size
            if self.valid_cat:
                result = (self.X[i:i_end,:],self.cat_X[i:i_end,:],self.y[i:i_end,:],self.delta[i:i_end])
            else:
                result = (self.X[i:i_end,:],[],self.y[i:i_end,:],self.delta[i:i_end])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False,super_fast=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.super_fast = super_fast
    def __iter__(self):
        if self.super_fast:
            return super_fast_iterator(X =self.dataset.X,
                                  delta = self.dataset.delta,
                                  y = self.dataset.y,
                                  cat_X = self.dataset.cat_X,
                                  batch_size=self.batch_size)
        else:
            return chunk_iterator(X =self.dataset.X,
                                  delta = self.dataset.delta,
                                  y = self.dataset.y,
                                  cat_X = self.dataset.cat_X,
                                  shuffle = self.shuffle,
                                  batch_size=self.batch_size)


def get_dataloader(str_identifier,bs,seed):
    d = surival_dataset(str_identifier,seed)
    # dat = DataLoader(dataset=d,batch_size=bs,shuffle=True,pin_memory=True)
    dat = custom_dataloader(dataset=d,batch_size=bs,shuffle=True,super_fast=False)
    return dat
