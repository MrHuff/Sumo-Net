from pycox.datasets import kkbox,support,metabric,gbsg,flchain
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
from .toy_data_generation import toy_data_class
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
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
            cont_cols = ['age','kappa','lambda','creatinine']
            binary_cols = ['sex','mgus']
            cat_cols = []
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

        if str_identifier=='kkbox':
            c = OrderedCategoricalLong(min_per_category=1,return_series=True)
            self.event_col = 'event'
            self.duration_col = 'duration'
            df_train = df_train.drop(['msno'],axis=1)
            df_train['gender'] = c.fit_transform(df_train['gender'])

        else:
            self.event_col = data.col_event
            self.duration_col = data.col_duration

        leave = binary_cols + cat_cols
        standardize = [([col], MinMaxScaler()) for col in cont_cols]
        leave = [(col, None) for col in leave]
        self.cat_cols = cat_cols

        self.x_mapper = DataFrameMapper(standardize + leave)
        self.duration_mapper = MinMaxScaler()
        if self.cat_cols:
            self.unique_cat_cols = df_train[cat_cols].unique().values().tolist()
        else:
            self.unique_cat_cols = []

        df_test = df_train.sample(frac=0.2,random_state=seed)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.25,random_state=seed)
        df_train = df_train.drop(df_val.index)

        x_train = self.x_mapper.fit_transform(df_train)
        x_val = self.x_mapper.transform(df_val)
        x_test = self.x_mapper.transform(df_test)

        y_train = self.duration_mapper.fit_transform(df_train[self.duration_col].values.reshape(-1,1))
        y_val = self.duration_mapper.transform(df_val[self.duration_col].values.reshape(-1,1))
        y_test = self.duration_mapper.transform(df_test[self.duration_col].values.reshape(-1,1))
        self.split(X=x_train,delta=df_train[self.event_col],y=y_train,mode='train',cat=cat_cols,df=df_train)
        self.split(X=x_val,delta=df_val[self.event_col],y=y_val,mode='val',cat=cat_cols,df=df_val)
        self.split(X=x_test,delta=df_test[self.event_col],y=y_test,mode='test',cat=cat_cols,df=df_test)
        self.set('train')

    def split(self,X,delta,y,cat=[],mode='train',df=[]):

        setattr(self,f'{mode}_delta', torch.from_numpy(delta.values).float())
        setattr(self,f'{mode}_y', torch.from_numpy(y).float())
        setattr(self, f'{mode}_X', torch.from_numpy(X).float())
        if self.cat_cols:
            setattr(self, f'{mode}_cat_X', torch.from_numpy(df[cat].values).long())

    def set(self,mode='train'):
        self.X = getattr(self,f'{mode}_X')
        self.y = getattr(self,f'{mode}_y')
        self.delta = getattr(self,f'{mode}_delta')
        if self.cat_cols:
            self.cat_X = getattr(self,f'{mode}_cat_X')
        else:
            self.cat_X = 0

    def transform_x(self,x):
        return self.x_mapper.transform(x)

    def invert_duration(self,duration):
        return self.duration_mapper.inverse_transform(duration)

    def __getitem__(self, index):
        if self.cat_cols:
            return self.X[index,:],self.cat_X[index,:],self.y[index],self.delta[index]
        else:
            return self.X[index,:],self.cat_X,self.y[index],self.delta[index]

    def __len__(self):
        return self.X.shape[0]

def get_dataloader(str_identifier,bs,seed):
    d = surival_dataset(str_identifier,seed)
    dat = DataLoader(dataset=d,batch_size=bs,shuffle=True)
    return dat
