import pandas as pd


datafile6 = '..\\..\\data\\BDCI2017-liangzi-10.12\\6project.csv'
data_6 = pd.read_csv(datafile6)

data_6['PROCOUNT'] = 1
data_6['BS_PROCOUNT'] = data_6['IFHOME']
data_6['WS_PROCOUNT'] = data_6['IFHOME'].apply(lambda x: 1 if x==0 else 0)
data_6_new = data_6.groupby('EID').sum()
data_6_new['BS_RATE'] = data_6_new['BS_PROCOUNT']/data_6_new['PROCOUNT']
data_6_new = data_6_new.drop(['TYPECODE','IFHOME'], axis=1)

data_6_new.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\6project_add.csv')
print(data_6_new)