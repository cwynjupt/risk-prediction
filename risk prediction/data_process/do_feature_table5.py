import pandas as pd


datafile5 = '..\\..\\data\\BDCI2017-liangzi-10.12\\5right.csv'
data_5 = pd.read_csv(datafile5)

data_5 = data_5.fillna(0)

data_5['RIGHT_COUNT'] = 1
data_5['RIGHT_11_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==11 else 0)
data_5['RIGHT_12_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==12 else 0)
data_5['RIGHT_20_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==20 else 0)
data_5['RIGHT_30_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==30 else 0)
data_5['RIGHT_40_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==40 else 0)
data_5['RIGHT_50_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==50 else 0)
data_5['RIGHT_60_COUNT'] = data_5['RIGHTTYPE'].apply(lambda x: 1 if x==60 else 0)
data_5['FUYU_RIGHT'] = data_5['FBDATE'].apply(lambda x: 1 if x!=0 else 0)
data_5['WFUYU_RIGHT'] = data_5['FBDATE'].apply(lambda x: 1 if x==0 else 0)

data_5_new = data_5.groupby('EID').sum()
data_5_new['IS_EXIST_WFU'] = data_5_new['WFUYU_RIGHT'].apply(lambda x: 1 if x>0 else 0)
data_5_new = data_5_new.drop('RIGHTTYPE', axis=1)

print(data_5)
print(data_5_new)

data_5_new.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\5right_add.csv')