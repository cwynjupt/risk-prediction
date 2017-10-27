import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

# # 读取整个数据集
# all_data = pd.read_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\all_table.csv')
# print(all_data.head())
# all_feature = all_data.drop('TARGET', axis=1)
# # 填充0
# all_feature = all_feature.fillna(0)

# train_target = pd.read_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\train.csv')
# test_target = pd.read_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv')

# 观察特征， 对特征进行处理
datafile1 = '..\\..\\data\\BDCI2017-liangzi-10.12\\1entbase.csv'
datafile2 = '..\\..\\data\\BDCI2017-liangzi-10.12\\2alter.csv'
datafile3 = '..\\..\\data\\BDCI2017-liangzi-10.12\\3branch.csv'
datafile4 = '..\\..\\data\\BDCI2017-liangzi-10.12\\4invest.csv'
datafile5 = '..\\..\\data\\BDCI2017-liangzi-10.12\\5right.csv'
datafile6 = '..\\..\\data\\BDCI2017-liangzi-10.12\\6project.csv'
datafile7 = '..\\..\\data\\BDCI2017-liangzi-10.12\\7lawsuit.csv'
datafile8 = '..\\..\\data\\BDCI2017-liangzi-10.12\\8breakfaith.csv'
datafile9 = '..\\..\\data\\BDCI2017-liangzi-10.12\\9recruit.csv'
data_1 = pd.read_csv(datafile1)
data_2 = pd.read_csv(datafile2)
data_3 = pd.read_csv(datafile3)
data_4 = pd.read_csv(datafile4)
data_5 = pd.read_csv(datafile5)
data_6 = pd.read_csv(datafile6)
data_7 = pd.read_csv(datafile7)
data_8 = pd.read_csv(datafile8)
data_9 = pd.read_csv(datafile9)

t = time.time()
# add feature for table3
data_3 = data_3.fillna(0)

data_3['BR_COUNT'] = 1
data_3['NORMAL_BR'] = data_3['B_ENDYEAR'].apply(lambda x: 1 if x==0 else 0)
data_3['END_BR'] = data_3['B_ENDYEAR'].apply(lambda x: 1 if x!=0 else 0)
data_3['WS_BR'] = data_3['IFHOME'].apply(lambda x: 0 if x==1 else 1)
data_3['BS_BR'] = data_3['IFHOME'].apply(lambda x: 1 if x==1 else 0)
data_3['BS_NORMAL_BR'] = data_3['BS_BR']
data_3['BS_NORMAL_BR'] = (data_3['B_ENDYEAR'].apply(lambda x: 1 if x==0 else 0)) & data_3['BS_NORMAL_BR']
# data_3['BS_NORMAL_BR'] = 0
# index = data_3[data_3['IFHOME']==1].index & data_3[data_3['B_ENDYEAR']==0].index
# data_3.iloc[index] = 1
data_3['BS_END_BR'] = data_3['BS_BR'] & (data_3['B_ENDYEAR'].apply(lambda x: 1 if x!=0 else 0))
data_3['WS_NORMAL_BR'] = data_3['WS_BR'] & (data_3['B_ENDYEAR'].apply(lambda x: 1 if x==0 else 0))
data_3['WS_END_BR'] = data_3['WS_BR'] & (data_3['B_ENDYEAR'].apply(lambda x: 1 if x!=0 else 0))
brlive = data_3['B_ENDYEAR'] - data_3['B_REYEAR']
data_3['BRLIVE'] = brlive[brlive>=0]
data_3['BS_END_BR_LIVE'] = data_3['BRLIVE'] * data_3['BS_END_BR']
data_3['WS_END_BR_LIVE'] = data_3['BRLIVE'] * data_3['WS_END_BR']

data_3 = data_3.groupby('EID').sum()
data_3['BRLIVE_AVG'] = data_3['BRLIVE'] / data_3['END_BR']
data_3['BS_END_BR_LIVE_AVG'] = data_3['BS_END_BR_LIVE'] / data_3['BS_END_BR']
data_3['WS_END_BR_LIVE_AVG'] = data_3['WS_END_BR_LIVE'] / data_3['WS_END_BR']
print(data_3)
#
data_3['EID'] = data_3.index
cols = list(data_3)
cols.insert(0, cols.pop(cols.index('EID')))
data_3 = data_3.ix[:, cols]
print(data_3.head())
data_3 = data_3.drop(['IFHOME', 'B_REYEAR', 'B_ENDYEAR'], axis=1)

data_3.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\3branch_add.csv', index=False)

print('run time is :', time.time() - t)







