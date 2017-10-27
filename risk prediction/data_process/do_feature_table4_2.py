import pandas as pd
import numpy as np

datafile4 = '..\\..\\data\\BDCI2017-liangzi-10.12\\4invest.csv'
data_4 = pd.read_csv(datafile4)

end_table = data_4.dropna()

new_table = end_table
new_table['ISEND'] = 1
new_table = new_table.drop(['EID', 'IFHOME', 'IFHOME', 'BTYEAR', 'BTENDYEAR', 'BTBL'], axis=1)
new_table['EID'] = new_table['BTEID']
print(sorted(new_table['EID']))

train_target = pd.read_csv('..\\..\\data\\BDCI2017-liangzi-10.12\\train.csv')
print(train_target)
comp_table = pd.merge(new_table, train_target, on='EID', how='left')
print(comp_table)
