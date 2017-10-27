import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

all_data = pd.read_csv('..\\data\\BDCI2017-liangzi-10.12\\all_table.csv')
print(all_data.head())
all_feature = all_data.drop('TARGET', axis=1)
# 填充0
all_feature = all_feature.fillna(0)
# all_feature = all_feature.drop('RGYEAR', axis=1)
# train_feature = all_data.drop('TARGET')
# train_target = all_data[all_data['EID']]['TARGET']

import lightgbm as lgb
from sklearn.model_selection import train_test_split
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
train_feature = train_feature.drop('TARGET', axis=1)
test_feature = pd.merge(test_target, all_feature, how='left', on='EID')
print(train_feature.head())
print(test_feature.head())
label = train_target['TARGET']
train = train_feature[predictor]
test = test_feature[predictor]

X_train, y_train, X_validation, y_validation = train_test_split(train, label, test_size=0.3, random_state=10000)
print(type(y_validation))

lgb_train = lgb.Dataset(X_train, X_validation)
lgb_eval = lgb.Dataset(y_train, y_validation)

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': {'auc',},
    'num_leaves': 2**6,
    'learning_rate': 0.05,
    'verbose': 0,
}
print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=30)
y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

test_target['PROB'] = y_pred
test_target['FORTARGET'] = test_target['PROB'].apply(lambda x: int(x*2))
test_target.to_csv('result.csv', index=False)

print('Calculate feature importance...')
print('Feature name:', gbm.feature_name())
print('Feature importance:', list(gbm.feature_importance()))
