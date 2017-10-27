import pandas as pd
# from data_process import do_feature
from data_process import split_data
# from model_process import forest_model
# from model_process import regression_model
from model_process import gbm_model
from model_process import xgboost_model
from model_process import gbm_gridsearch
from model_process import gbm_gridsearch_sk


# read data
data_all = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\all_3.csv')
data_all = data_all.drop('TARGET', axis=1)
# print('the feature sparsity is {0:.3f}%'.format(data_all.nnz/float(data_all.shape[0]*data_all.shape[1])*100))

# do feature
data_all = data_all.fillna(0)

train_target = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\train.csv')
test_target = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv')
train_feature = pd.merge(train_target, data_all, how='left', on='EID')
train_feature = train_feature.drop('TARGET', axis=1)
test_feature = pd.merge(test_target, data_all, how='left', on='EID')

# split train data to train data and validation data
label = train_target['TARGET']
train = train_feature
test = test_feature
# train = train_feature.drop('EID', axis=1)
# test = test_feature.drop('EID', axis=1)
X_train, y_train, X_validation, y_validation = split_data.splits_data(train, label)

# fit model
gbm_model.gbm_model(X_train, y_train, X_validation, y_validation, test, test_target)
# xgboost_model.xgb_model(X_train, y_train, X_validation, y_validation, test, test_target)
# gbm_gridsearch.gridsearchCV(train, label)
