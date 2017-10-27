import pandas as pd
import numpy as np

# add feature for table 8
datafile8 = '..\\..\\data\\BDCI2017-liangzi-10.12\\8breakfaith.csv'
data_8 = pd.read_csv(datafile8)

# do with no time
data_8['BF_COUNT'] = 1
print(type(data_8.iloc[1, 3]))
data_8['BF_END'] = data_8['SXENDDATE'].apply(lambda x: 1 if x is not np.nan else 0)
data_8['BF_IN'] = data_8['SXENDDATE'].apply(lambda x: 1 if x is np.nan else 0)
# change to time type
data_8['SXENDDATE'] = pd.to_datetime(data_8['SXENDDATE'])
data_8['FBDATE'] = pd.to_datetime(data_8['FBDATE'])

# bflive = data_8['SXENDDATE'] - data_8['FBDATE']
# # print(np.timedelta64(1, 'ns'))
# print(bflive)
# data_8['BF_LIVE'] = bflive[bflive >= np.timedelta64(0, 'ns')]
#
data_8 = data_8.groupby('EID').sum()

data_8['IS_EXIST_INBF'] = data_8['BF_IN'].apply(lambda x: 1 if x>0 else 0)
data_8['IS_INDYEND'] = data_8['BF_IN'] - data_8['BF_END']
data_8['IS_INDYEND'] = data_8['IS_INDYEND'].apply(lambda x: 1 if x>0 else 0)
data_8['BF_END_RATE'] = data_8['BF_END']/data_8['BF_COUNT']
data_8['BF_IN_RATE'] = data_8['BF_IN']/data_8['BF_COUNT']

data_8 = data_8.drop('TYPECODE', axis=1)
data_8.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\8breakfaith_add.csv')

print(data_8)

# deal with time
