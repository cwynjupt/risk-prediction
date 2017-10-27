import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def gbm_model( X_train, y_train, X_validation, y_validation, test, test_target):
    '''

    :param X_train:
    :param y_train:
    :param X_validation:
    :param y_validation:
    :param test:
    :param test_target:
    :return:
    '''

    params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': {'auc',},
        'num_leaves': 64,
        'learning_rate': 0.01,
        'verbose': 0,
        'subsample': 0.8,
        'min_data_in_leaf': 60,
        'feature_fraction': 0.9,
        'lambda_l1': 2,
        'lambda_l2': 1,
        # 'ignore_column': {61,67,68,70,74},
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_threads': 4,
        # 'is_unbalance': True,
    }

    # params['is_unbalance'] = 'true'
    print('Start cv-ing...')

    # data for no validation:need split
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_validation, y_validation)
    gbm = lgb.train(params, lgb_train, num_boost_round=1500, valid_sets=lgb_eval, early_stopping_rounds=30)

    # data for validation :not split any more
    # X_train_1 = pd.concat([X_train, X_validation])
    # y_label = pd.concat([y_train, y_validation])
    # lgb_train = lgb.Dataset(X_train_1, label=y_label)
    # bst = lgb.cv(params, lgb_train, num_boost_round=1500, nfold=5, stratified=True, early_stopping_rounds=30)
    # print(len(bst['auc-mean']))
    # print(bst['auc-mean'])
    # print('Start training...')
    # gbm = lgb.train(params, lgb_train, num_boost_round=len(bst['auc-mean']))
    # train_pred = gbm.predict(lgb_train, num_iteration=gbm.best_iteration)
    # auc_score = metrics.roc_auc_score(y_label, train_pred)
    # print('auc_score:', auc_score)

    y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

    test_target['FORTARGET'] = y_pred
    test_target['FORTARGET'] = test_target['FORTARGET'].apply(lambda x: 1 if x>0.22 else 0)
    test_target['PROB'] = y_pred
    test_target.to_csv('result_gbm.csv', index=False)

    print('Calculate feature importance...')
    print('num of feature:', len(gbm.feature_name()))
    pd.set_option('max_rows', 400)
    importance = pd.DataFrame({'feature': gbm.feature_name(), 'importance': list(gbm.feature_importance())})
    print(importance)
    # print('Feature name:', gbm.feature_name())
    # print('Feature importance:', list(gbm.feature_importance()))

    # feature_img = pd.Series(gbm.feature_importance(), gbm.feature_name()).sort_values(ascending=False)
    # feature_img.plot(kind='bar', title='feature importance')
    # plt.show()
