import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics


def gridsearchCV(train, label):
    # train = lgb.Dataset(train, label=label)
    params_test = {
        # 'min_data_in_leaf': list(range(20, 60, 10)),
        'lambda_l1': list(range(0, 6, 1)),
        'lambda_l2': list(range(0, 6, 1)),
        # 'num_leaves': list(range(20, 120, 10)),
    }
    # params_class = {
    #     # 'task': 'train',
    #     # 'boosting': 'gbdt',
    #     # 'objective': 'binary',
    #     # 'metric': {'auc', },
    #     'num_leaves': 64,
    #     'num_threads': 4,
    #     'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'subsample': 0.8,
    #     'verbose': 0,
    # }
    estimator = LGBMClassifier(
        # params_class,
        # num_leaves=64,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        num_boost_round=1000,
        learning_rate=0.2,
        lambda_l1=2,
        lambda_l2=1,

        min_data_in_leaf=60,    # 30
        num_leaves=64,   # 20
    )
    gsearch = GridSearchCV(estimator=estimator, param_grid=params_test, scoring='roc_auc', cv=5, verbose=0)
    gsearch.fit(train, label)
    print('best score is :\n', gsearch.best_score_)
    print('best params is:\n')
    parameter = gsearch.best_estimator_.get_params()
    for param_name in sorted(params_test.keys()):
        print('\t%s:%r\n' %(param_name, parameter[param_name]))
    return gsearch.best_params_


# def gbm_model(train, label, test, test_target):
#     params = gridserachCV(train, label)
#     params['boosting_type'] = 'gbdt'
#     params['objective'] = 'binary'
#
#     train = lgb.Dataset(train, label=label)
#
#     gbm = lgb.cv(params, train, num_boost_round=1000, nfold=5, stratified=True, early_stopping_rounds=30)
#     lgbm = lgb.train(params, train, num_boost_round=len(gbm['mean-auc']))
#     y_pred = lgbm.predict(test, num_iteration=lgbm.best_iteration)
#
#     test_target['FORTARGET'] = y_pred
#     test_target['FORTARGET'] = test_target['FORTARGET'].apply(lambda x: 1 if x > 0.22 else 0)
#     test_target['PROB'] = y_pred
#     test_target.to_csv('result_gbm.csv', index=False)
#
#     print('Calculate feature importance...')
#     print('num of feature:', len(lgbm.feature_name()))
#     print('Feature name:', lgbm.feature_name())
#     print('Feature importance:', list(lgbm.feature_importance()))
