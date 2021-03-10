from pycox.datasets import kkbox,support,metabric,gbsg,flchain
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from pycox.preprocessing.feature_transforms import *
import torch
from .toy_data_generation import toy_data_class
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from lifelines import KaplanMeierFitter
import pycox.utils as utils

class toy_dataset():
    def __init__(self,name):
        self.load_path = f'./{name}/{name}.csv'
        self.left_int = 'l'
        self.right_int = 'r'

    def read_df(self):
        df = pd.read_csv(self.load_path,index_col=0)
        return df


class tooth_dataset():
    def __init__(self):
        self.load_path = f'tooth.csv'
        self.left_int = 'left'
        self.right_int = 'rightInf'

    def read_df(self):
        df = pd.read_csv(self.load_path,index_col=0)
        return df

class essIncData_dataset():
    def __init__(self):
        self.load_path = f'essIncData.csv'
        self.left_int = 'inc_l'
        self.right_int = 'inc_u'

    def read_df(self):
        df = pd.read_csv(self.load_path,index_col=0)
        return df

def calc_km(durations,events):
    km = utils.kaplan_meier(durations, 1 - events)
    return km

class LogTransformer(BaseEstimator, TransformerMixin): #Scaling is already good. This leaves network architecture...
    def __init__(self):
        pass

    def fit_transform(self, input_array, y=None):
        return np.log(input_array)

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return np.log(input_array)

    def inverse_transform(self,input_array):
        return np.exp(input_array)

class IdentityTransformer(BaseEstimator, TransformerMixin): #Scaling is already good. This leaves network architecture...
    def __init__(self):
        pass

    def fit_transform(self, input_array, y=None):
        return input_array

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array * 1

    def inverse_transform(self,input_array):
        return input_array

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class surival_dataset_interval(Dataset):
    def __init__(self,str_identifier,seed=1337,fold_idx=0):
        print('fold_idx: ', fold_idx)
        super(surival_dataset_interval, self).__init__()
        if str_identifier=='essIncData':
            data = essIncData_dataset()
            cont_cols = []
            binary_cols = []
            cat_cols = ['cntry','eduLevel']

        elif str_identifier=='tooth':
            data = tooth_dataset()
            cont_cols = []
            binary_cols = []
            cat_cols = ['sex', 'dmf']

        elif str_identifier in ['interval_checkboard','interval_weibull','interval_normal']:
            data = toy_dataset(str_identifier)
            cont_cols = ['X']
            binary_cols = []
            cat_cols = []

        df_full = data.read_df()
        df_full = df_full.dropna()
        df_full =df_full.replace([np.inf,-np.inf], np.nan)
        self.left_int = data.left_int
        self.right_int = data.right_int
        self.inf_marker = 'is_inf_bool'
        df_full[self.inf_marker] = np.isnan(df_full[self.right_int])
        print(f'{str_identifier} max',df_full[self.right_int].max())
        print(f'{str_identifier} min',df_full[self.left_int].min())
        c = OrderedCategoricalLong()
        for el in cat_cols:
            df_full[el] = c.fit_transform(df_full[el])
        standardize = [([col], MinMaxScaler()) for col in cont_cols]
        leave = [(col,None) for col in binary_cols]
        self.cat_cols = cat_cols
        self.x_mapper = DataFrameMapper(standardize+leave)
        if self.cat_cols:
            self.unique_cat_cols = df_full[cat_cols].max(axis=0).tolist()
            self.unique_cat_cols = [el+1 for el in self.unique_cat_cols]
            # for el in cat_cols:
            #     print(f'column {el}:', df_full[el].unique().tolist())
            # print(self.unique_cat_cols)
        else:
            self.unique_cat_cols = []

        folder = StratifiedKFold(n_splits=5,  shuffle=True, random_state=seed)
        splits = list(folder.split(df_full,df_full[self.inf_marker]))
        tr_idx,tst_idx = splits[fold_idx]
        df_train = df_full.iloc[tr_idx,:]
        df_test = df_full.iloc[tst_idx,:]
        df_train, df_val, _, _ = train_test_split(df_train, df_train[self.left_int], test_size = 0.25,stratify=df_train[self.inf_marker])
        if cont_cols or binary_cols:
            self.regular_X = True
            x_train = self.x_mapper.fit_transform(df_train[cont_cols+binary_cols]).astype('float32')
            x_val = self.x_mapper.transform(df_val[cont_cols+binary_cols]).astype('float32')
            x_test = self.x_mapper.transform(df_test[cont_cols+binary_cols]).astype('float32')
        else:
            self.regular_X = False
            x_train = []
            x_val = []
            x_test = []
        self.duration_mapper = MinMaxScaler()
        subset = df_train[~df_train[self.inf_marker]]
        self.duration_mapper.fit(subset[self.right_int].values.reshape(-1,1))

        y_train_right = self.duration_mapper.transform(df_train[self.right_int].values.reshape(-1,1)).astype('float32')
        y_train_left = self.duration_mapper.transform(df_train[self.left_int].values.reshape(-1,1)).astype('float32')
        y_val_right = self.duration_mapper.transform(df_val[self.right_int].values.reshape(-1, 1)).astype(
            'float32')
        y_val_left = self.duration_mapper.transform(df_val[self.left_int].values.reshape(-1, 1)).astype(
            'float32')
        y_test_right = self.duration_mapper.transform(df_test[self.right_int].values.reshape(-1, 1)).astype(
            'float32')
        y_test_left = self.duration_mapper.transform(df_test[self.left_int].values.reshape(-1, 1)).astype(
            'float32')


        self.split(X=x_train,inf_indicator=df_train[self.inf_marker],y_left=y_train_left,y_right=y_train_right,mode='train',cat=cat_cols,df=df_train)
        self.split(X=x_val,inf_indicator=df_val[self.inf_marker],y_left=y_val_left,y_right=y_val_right,mode='val',cat=cat_cols,df=df_val)
        self.split(X=x_test,inf_indicator=df_test[self.inf_marker],y_left=y_test_left,y_right=y_test_right,mode='test',cat=cat_cols,df=df_test)
        self.set('train')

    def split(self,X,inf_indicator,y_left,y_right,cat=[],mode='train',df=[]):
        remove_left_mask = np.isnan(y_left).squeeze()
        y_left = y_left[~remove_left_mask]
        inf_indicator = inf_indicator[~remove_left_mask]
        y_right = y_right[~remove_left_mask]

        setattr(self,f'{mode}_inf_indicator', torch.from_numpy(inf_indicator.astype('bool').values).bool())
        setattr(self,f'{mode}_y_left', torch.from_numpy(y_left).float())
        setattr(self,f'{mode}_y_right', torch.from_numpy(y_right).float())
        if self.regular_X:
            X = X[~remove_left_mask]
            setattr(self, f'{mode}_X', torch.from_numpy(X).float())
        if self.cat_cols:
            cat=cat[~remove_left_mask]
            setattr(self, f'{mode}_cat_X', torch.from_numpy(df[cat].astype('int64').values).long())

    def set(self,mode='train'):
        self.y_left = getattr(self,f'{mode}_y_left')
        self.y_right = getattr(self,f'{mode}_y_right')
        self.inf_indicator = getattr(self,f'{mode}_inf_indicator')
        if self.cat_cols:
            self.cat_X = getattr(self,f'{mode}_cat_X')
        else:
            self.cat_X = []
        if self.regular_X:
            self.X = getattr(self, f'{mode}_X')
        else:
            self.X = []
        self.min_duration = self.y_left.min().numpy()
        self.max_duration = self.y_right[~self.inf_indicator].max().numpy()

    def transform_x(self,x):
        return self.x_mapper.transform(x)

    def invert_duration(self,duration):
        return self.duration_mapper.inverse_transform(duration)

    def transform_duration(self,duration):
        return self.duration_mapper.transform(duration)

    def __getitem__(self, index):
        #X,x_cat,y_left,y_right,inf_indicator
        if self.cat_cols:
            if self.regular_X:
                return self.X[index, :], self.cat_X[index, :], self.y_left[index], self.y_right[index], \
                       self.inf_indicator[index]
            else:
                return [], self.cat_X[index, :], self.y_left[index], self.y_right[index], \
                       self.inf_indicator[index]
        else:
            return self.X[index,:],[],self.y_left[index],self.y_right[index],self.inf_indicator[index]


    def __len__(self):
        return self.X.shape[0]

class chunk_iterator():
    def __init__(self,dataset,shuffle,batch_size):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = dataset.y_left.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        self.X  = dataset.X
        self.cat_X  = dataset.cat_X
        self.y_left  = dataset.y_left
        self.y_right  = dataset.y_right
        self.inf_indicator  = dataset.inf_indicator
        self.valid_cat = not isinstance(self.cat_X, list)
        self.regular_X = not isinstance(self.X, list)
        if self.shuffle:
            self.inf_indicator = self.inf_indicator[self.perm]
            self.y_left = self.y_left[self.perm,:]
            self.y_right = self.y_right[self.perm,:]
            if self.valid_cat: #F
                self.cat_X = self.cat_X[self.perm,:]
            if self.regular_X:
                self.X = self.X[self.perm, :]
        self._index = 0
        self.it_inf_indicator = torch.chunk(self.inf_indicator,self.chunks)
        self.it_y_left = torch.chunk(self.y_left,self.chunks)
        self.it_y_right = torch.chunk(self.y_right,self.chunks)
        if self.valid_cat:  # F
            self.it_cat_X = torch.chunk(self.cat_X,self.chunks)
        else:
            self.it_cat_X = []

        if self.regular_X:
            self.it_X = torch.chunk(self.X, self.chunks)
        else:
            self.it_X = []
        self.true_chunks = len(self.it_y_left)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            if self.valid_cat:
                if self.regular_X:
                    result = (self.it_X[self._index], self.it_cat_X[self._index], self.it_y_left[self._index],
                              self.it_y_right[self._index], self.it_inf_indicator[self._index])
                else:
                    result = ([], self.it_cat_X[self._index], self.it_y_left[self._index],
                              self.it_y_right[self._index], self.it_inf_indicator[self._index])
            else:
                result = (self.it_X[self._index],[],self.it_y_left[self._index],self.it_y_right[self._index], self.it_inf_indicator[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.y_left.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return chunk_iterator(dataset=self.dataset,shuffle=self.shuffle,batch_size=self.batch_size)
    def __len__(self):
        self.n = self.dataset.y_left.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len

def get_dataloader_interval(str_identifier,bs,seed,fold_idx,shuffle=True):
    d = surival_dataset_interval(str_identifier, seed, fold_idx=fold_idx)
    dat = custom_dataloader(dataset=d,batch_size=bs,shuffle=shuffle)
    return dat
