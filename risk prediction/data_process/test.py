import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
datafile3 = '..\\..\\data\\BDCI2017-liangzi-10.12\\3branch.csv'
data_3 = pd.read_csv(datafile3)
data = data_3.groupby(['EID', 'B_REYEAR']).count()
print(data)