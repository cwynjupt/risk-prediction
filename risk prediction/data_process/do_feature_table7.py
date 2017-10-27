import pandas as pd


datafile7 = '..\\..\\data\\BDCI2017-liangzi-10.12\\7lawsuit.csv'
data_7 = pd.read_csv(datafile7)

# do with no time
data_7['LAW_COUNT'] = 1
data_7['LAW_YMONEY_COUNT'] = data_7['LAWAMOUNT'].apply(lambda x: 1 if x!=0 else 0)
data_7['LAW_WMONEY_COUNT'] = data_7['LAWAMOUNT'].apply(lambda x: 1 if x==0 else 0)

data_7 = data_7.groupby('EID').sum()
print(data_7)
data_7['LAW_AMOUNT_AVG'] = data_7['LAWAMOUNT']/data_7['LAW_COUNT']
data_7['LAW_AMOUNT_AVG_YB'] = data_7['LAWAMOUNT']/data_7['LAW_YMONEY_COUNT']
data_7['IS_YBDYWB'] = data_7['LAW_YMONEY_COUNT']-data_7['LAW_WMONEY_COUNT']
data_7['IS_YBDYWB'] = data_7['IS_YBDYWB'].apply(lambda x: 1 if x>0 else 0)
data_7['LAW_YB_RATE'] = data_7['LAW_YMONEY_COUNT'] / data_7['LAW_COUNT']
data_7['LAW_WB_RATE'] = data_7['LAW_WMONEY_COUNT'] / data_7['LAW_COUNT']
data_7['IS_EXIST_YB'] = data_7['LAW_YMONEY_COUNT'].apply(lambda x: 1 if x>0 else 0)
data_7['IS_EXIST_WB'] = data_7['LAW_WMONEY_COUNT'].apply(lambda x: 1 if x>0 else 0)

# cols = list(data_7)
# data_7 = data_7.ix[:, cols]
data_7 = data_7.drop(['TYPECODE'], axis=1)

print(data_7.tail(10))

# data_7.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\7lawsuit_add.csv')

# deal with time