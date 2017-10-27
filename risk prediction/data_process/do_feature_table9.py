import pandas as pd


datafile9 = '..\\..\\data\\BDCI2017-liangzi-10.12\\9recruit.csv'
data_9 = pd.read_csv(datafile9)
data_9['IS_WZ1'] = data_9['WZCODE'].apply(lambda x: 1 if x=='ZP01' else 0)
data_9['IS_WZ2'] = data_9['WZCODE'].apply(lambda x: 1 if x=='ZP02' else 0)
data_9['IS_WZ3'] = data_9['WZCODE'].apply(lambda x: 1 if x=='ZP03' else 0)
data_9['WZ1_RECRNUM'] = data_9['IS_WZ1'] * data_9['RECRNUM']
data_9['WZ2_RECRNUM'] = data_9['IS_WZ2'] * data_9['RECRNUM']
data_9['WZ3_RECRNUM'] = data_9['IS_WZ3'] * data_9['RECRNUM']

data_9_new = data_9.groupby('EID').sum()
data_9_new['WZ_COUNT'] = data_9_new['IS_WZ1'] + data_9_new['IS_WZ2'] + data_9_new['IS_WZ3']
data_9_new['IS_RECR'] = data_9_new['RECRNUM'].apply(lambda x: 1 if x>0 else 0)

print(data_9_new)
data_9_new.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\9recruit_add.csv')