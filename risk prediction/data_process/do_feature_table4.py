import pandas as pd
import numpy as np

datafile4 = '..\\..\\data\\BDCI2017-liangzi-10.12\\4invest.csv'
data_4 = pd.read_csv(datafile4)
data_4 = data_4.fillna(0)

data_4['BTCOUNT'] = 1
data_4['BS_BTCOUNT'] = data_4['IFHOME'].apply(lambda x: 1 if x==1 else 0)
data_4['WS_BTCOUNT'] = data_4['IFHOME'].apply(lambda x: 1 if x!=1 else 0)
data_4['BT_END_COUNT'] = data_4['BTENDYEAR'].apply(lambda x: 1 if x!=0 else 0)
data_4['BT_NORMAL_COUNT'] = data_4['BTENDYEAR'].apply(lambda x: 1 if x==0 else 0)
data_4['BS_BT_END'] = data_4['BS_BTCOUNT'] & data_4['BT_END_COUNT']
data_4['WS_BT_END'] = data_4['WS_BTCOUNT'] & data_4['BT_END_COUNT']
btlive = data_4['BTENDYEAR'] - data_4['BTYEAR']
data_4['BT_LIVE'] = btlive[btlive>=0]
data_4['BS_BT_LIVE'] = data_4['BS_BTCOUNT'] * data_4['BT_LIVE']
data_4['WS_BT_LIVE'] = data_4['WS_BTCOUNT'] * data_4['BT_LIVE']
data_4['BS_BT_BL'] = data_4['BS_BTCOUNT'] * data_4['BTBL']
data_4['WS_BT_BL'] = data_4['WS_BTCOUNT'] * data_4['BTBL']
print(data_4.tail(10))

data_4_new = data_4.groupby('EID').sum()

data_4_new['BS_BT_LIVE_AVG'] = data_4_new['BS_BT_LIVE']/data_4_new['BS_BTCOUNT']
data_4_new['WS_BT_LIVE_AVG'] = data_4_new['WS_BT_LIVE']/data_4_new['WS_BTCOUNT']
data_4_new['BT_LIVE_AVG'] = data_4_new['BT_LIVE']/data_4_new['BTCOUNT']
data_4_new['BT_BL_AVG'] = data_4_new['BTBL']/data_4_new['BTCOUNT']
data_4_new['BS_BT_BL_AVG'] = data_4_new['BS_BT_BL']/data_4_new['BS_BTCOUNT']
data_4_new['WS_BT_BL_AVG'] = data_4_new['WS_BT_BL']/data_4_new['WS_BTCOUNT']
data_4_new['IS_EXIST_BT'] = data_4_new['BTCOUNT'].apply(lambda x: 1 if x>0 else 0)
data_4_new['IS_EXIST_NORMAL_BT'] = data_4_new['BT_NORMAL_COUNT'].apply(lambda x: 1 if x>0 else 0)

data_4_new = data_4_new.drop(['BTEID', 'IFHOME', 'BTYEAR', 'BTENDYEAR'], axis=1)
print(data_4_new.tail(10))

data_4_new.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\4invest_add.csv')