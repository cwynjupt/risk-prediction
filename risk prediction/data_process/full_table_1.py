import pandas as pd
import numpy as np

datafile_all = '..\\data\\BDCI2017-liangzi-10.12\\all_tablename.csv'
datafile_train = '..\\data\\BDCI2017-liangzi-10.12\\train.csv'
datafile_test = '..\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv'
datafile1 = '..\\data\\BDCI2017-liangzi-10.12\\1entbase.csv'
datafile2 = '..\\data\\BDCI2017-liangzi-10.12\\2alter_count.csv'
datafile3 = '..\\data\\BDCI2017-liangzi-10.12\\3branch_count.csv'
datafile4 = '..\\data\\BDCI2017-liangzi-10.12\\4invest_count.csv'
datafile5 = '..\\data\\BDCI2017-liangzi-10.12\\5right_count.csv'
datafile6 = '..\\data\\BDCI2017-liangzi-10.12\\6project_count.csv'
datafile7 = '..\\data\\BDCI2017-liangzi-10.12\\7lawsuit_count.csv'
datafile8 = '..\\data\\BDCI2017-liangzi-10.12\\8breakfaith_count.csv'
datafile9 = '..\\data\\BDCI2017-liangzi-10.12\\9recruit_count.csv'

data_train = pd.read_csv(datafile_train)
data_all = pd.read_csv(datafile_all)
data_test = pd.read_csv(datafile_test)
data_1 = pd.read_csv(datafile1)
data_2 = pd.read_csv(datafile2)
data_3 = pd.read_csv(datafile3)
data_4 = pd.read_csv(datafile4)
data_5 = pd.read_csv(datafile5)
data_6 = pd.read_csv(datafile6)
data_7 = pd.read_csv(datafile7)
data_8 = pd.read_csv(datafile8)
data_9 = pd.read_csv(datafile9)

# data_all = data_1
data_all['EID'] = data_1['EID']
# print(data_all['EID'].isin(data_train['EID']))
# print(data_train[data_all['EID'].isin(data_train['EID'])])
data_target = data_train.set_index(data_all[data_all['EID'].isin(data_train['EID'])].index)
data_all['TARGET'] = data_target['TARGET']
print(data_all.head())
table1_col = ['RGYEAR',	'HY',	'ZCZB', 'ETYPE',	'MPNUM',	'INUM', 'FINZB',	'FSTINUM',	'TZINUM']
table2_col = ['ALTCOUNT']
table3_col = ['WSBR',	'WSENDBR',	'BSBR',	'BSENDBR',	'COUNTBR',	'ENDBR']
table4_col = ['WSIN',	'WSEND',	'BSIN',	'BSEND',	'INCOUNT',	'ENDCOUNT']
table5_col = ['RIGHTCOUNT']
table6_col = ['WSPROCOUNT',	'BSPROCOUNT',	'PROCOUNT']
table7_col = ['LSCOUNT',	'LAWAMOUNT']
table8_col = ['FBCOUNT',	'FBEND']
table9_col = ['WZCODE',	'RECRNUM']

data_1_new = data_1[data_1['EID'].isin(data_all['EID'])].drop(['EID'], axis=1)
data_1_new = data_1_new.set_index(np.arange(len(data_1_new)))
data_all[table1_col] = data_1_new
print(data_all.head())

# print(len(data_all))   # 重新改变一下index就对了
# print(data_all['EID'].isin(data_2['EID']))
# print(data_all[data_all['EID'].isin(data_2['EID'])].index)
data_2 = data_2.set_index(data_all[data_all['EID'].isin(data_2['EID'])].index)
# print(data_2)
data_all[table2_col] = data_2.drop(['EID'], axis=1)
print(data_all[table2_col])

data_3 = data_3.set_index(data_all[data_all['EID'].isin(data_3['EID'])].index)
data_all[table3_col] = data_3.drop(['EID'], axis=1)
print(data_all.head())

data_4 = data_4.set_index(data_all[data_all['EID'].isin(data_4['EID'])].index)
data_all[table4_col] = data_4.drop(['EID'], axis=1)
print(data_all.head())

data_5= data_5.set_index(data_all[data_all['EID'].isin(data_5['EID'])].index)
data_all[table5_col] = data_5.drop(['EID'], axis=1)
print(data_all.head())

data_6 = data_6.set_index(data_all[data_all['EID'].isin(data_6['EID'])].index)
data_all[table6_col] = data_6.drop(['EID'], axis=1)

data_7 = data_7.set_index(data_all[data_all['EID'].isin(data_7['EID'])].index)
data_all[table7_col] = data_7.drop(['EID'], axis=1)
print(data_all.head())

data_8 = data_8.set_index(data_all[data_all['EID'].isin(data_8['EID'])].index)
data_all[table8_col] = data_8.drop(['EID'], axis=1)
print(data_all.head())

data_9 = data_9.set_index(data_all[data_all['EID'].isin(data_9['EID'])].index)
data_all[table9_col] = data_9.drop(['EID'], axis=1)
print('data', data_all.head())

# data_all.to_csv('C:\\Users\\dell\\Desktop\\大数据竞赛\\data fountain\\企业预测\\data\\BDCI2017-liangzi-10.12\\all_table.csv', index=False)
