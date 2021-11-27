import pandas as pd
import numpy as np
from serious_run import  *

def pivot_to_df(table_mean):
    table_mean = table_mean.reset_index().rename_axis(None, axis=1)
    table_mean = table_mean.set_index(table_mean['net'])
    table_mean = table_mean.drop(['net'], axis=1)

    return table_mean
datasets = ['support',
            'metabric',
            'gbsg',
            'flchain',
            'kkbox',
            ]
df = pd.read_csv('complexity_test_extend_epoch.csv')
table_mean = df.pivot_table(values='epoch_time',index='net',columns='dataset',aggfunc=np.mean)
table_std = df.pivot_table(values='epoch_time',index='net',columns='dataset',aggfunc=np.std)
table_mean = pivot_to_df(table_mean)
table_std = pivot_to_df(table_std)
tab = table_mean.applymap(lambda x: '$'+str(round(x,3))+'\pm')+table_std.applymap(lambda x: str(round(x,3))+'$')
table_mean_t = tab.transpose()
table_mean_t = table_mean_t.rename(index = {0:'SUPPORT',1:'METABRIC',2:'GBSG',3:'FLCHAIN',4:'KKBOX'})
table_mean_t.to_latex(buf=f"timings_transposed_reb_2.tex",escape=False)
# speedup = table_mean.iloc[0,:]/table_mean.iloc[1,:]
# table_mean = tab.append(speedup.apply(lambda x: '$'+str(round(x)))+'$',ignore_index=True)
# table_mean.columns = [el.upper() for el in datasets]
# table_mean = table_mean.rename(index = {0:'Cox-Time',1:'SuMo-net',2:'Speedup factor'})
# table_mean.to_latex(buf=f"timings.tex",escape=False)
# table_mean_t = table_mean.transpose()
# table_mean_t.to_latex(buf=f"timings_transposed.tex",escape=False)

# median_1 = round(df['Cox-Time'].median(),2)
# median_2 = round(df['SuMo-net'].median(),2)
# nobs = [median_1,median_2]
# boxplot = df.boxplot(column=['Cox-Time','SuMo-net'])
# boxplot.set_ylabel("Time (s)")
# # Add it to the plot
# pos = range(len(nobs))
# for tick, label in zip(pos, boxplot.get_xticklabels()):
#     boxplot.text(pos[tick]+1, nobs[tick] + 0.35, nobs[tick],
#             horizontalalignment='center', size='x-small', color='b', weight='semibold')
# boxplot.figure.savefig('boxplot.png')
