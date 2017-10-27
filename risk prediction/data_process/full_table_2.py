import pandas as pd


# datafile_all = '..\\..\\data\\BDCI2017-liangzi-10.12\\all_tablename.csv'
datafile_train = '..\\..\\data\\BDCI2017-liangzi-10.12\\train.csv'
datafile_test = '..\\..\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv'
datafile1 = '..\\..\\data\\BDCI2017-liangzi-10.12\\1entbase.csv'
datafile2 = '..\\..\\data\\BDCI2017-liangzi-10.12\\2alter_add.csv'
datafile3 = '..\\..\\data\\BDCI2017-liangzi-10.12\\3branch_add.csv'
datafile4 = '..\\..\\data\\BDCI2017-liangzi-10.12\\4invest_add.csv'
datafile5 = '..\\..\\data\\BDCI2017-liangzi-10.12\\5right_add.csv'
datafile6 = '..\\..\\data\\BDCI2017-liangzi-10.12\\6project_add.csv'
datafile7 = '..\\..\\data\\BDCI2017-liangzi-10.12\\7lawsuit_add.csv'
datafile8 = '..\\..\\data\\BDCI2017-liangzi-10.12\\8breakfaith_add.csv'
datafile9 = '..\\..\\data\\BDCI2017-liangzi-10.12\\9recruit_add.csv'
data_train = pd.read_csv(datafile_train)
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
all = pd.merge(data_1, data_2, how='left', on='EID')
all = pd.merge(all, data_3,  how='left', on='EID')
all = pd.merge(all, data_4,  how='left', on='EID')
all = pd.merge(all, data_5,  how='left', on='EID')
all = pd.merge(all, data_6,  how='left', on='EID')
all = pd.merge(all, data_7,  how='left', on='EID')
all = pd.merge(all, data_8,  how='left', on='EID')
all = pd.merge(all, data_9,  how='left', on='EID')
all = pd.merge(all, data_train,  how='left', on='EID')

print(all.head())
all.to_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\all_4.csv', index=False)