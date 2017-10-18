import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取整个数据集
all_data = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\all_table.csv')
print(all_data.head())
all_feature = all_data.drop('TARGET', axis=1)
# 填充0
all_feature = all_feature.fillna(0)

predictor = ['RGYEAR', 'HY',	'ZCZB', 'ETYPE',
             'MPNUM',	'INUM', 'FINZB',	'FSTINUM',
             'TZINUM','ALTCOUNT','WSBR',	'WSENDBR',
             'BSBR',	'BSENDBR',	'COUNTBR',	'ENDBR',
             'WSIN',	'WSEND',	'BSIN',	'BSEND',
             'INCOUNT',	'ENDCOUNT','RIGHTCOUNT','WSPROCOUNT',
             'BSPROCOUNT',	'PROCOUNT','LSCOUNT',	'LAWAMOUNT','FBCOUNT',
             'FBEND','WZCODE',	'RECRNUM']
train_target = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\train.csv')
test_target = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv')
train_feature = pd.merge(train_target, all_feature, how='left', on='EID')
# 观察特征， 对特征进行处理






# 在对数据进行处理之后，划分为训练数据和测试数据，进行建模预测
train_feature = train_feature.drop('TARGET', axis=1)
test_feature = pd.merge(test_target, all_feature, how='left', on='EID')
# print(train_feature.head())
# print(test_feature.head())
label = train_target['TARGET']
train = train_feature[predictor]
test = test_feature[predictor]