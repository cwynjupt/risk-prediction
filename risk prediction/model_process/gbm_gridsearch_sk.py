from sklearn.grid_search import GridSearchCV
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from lightgbm.sklearn import LGBMModel


def gridsearchCV(train, label):
    params_test = {
        # 'learning_rate': 0.01,
        # 'verbose': 0,
        # 'subsample': 0.8,
        # 'min_data_in_leaf': 60,
        # 'feature_fraction': 0.9,
        # 'lambda_l1': 2,
        # 'lambda_l2': 1,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 5,

        'learning_rate': range(0.01, 0.2, 0.02),
        'min_data_in_leaf': range(20, 60, 5),
        'lambda_l1': range(0, 6),
        'lambda_l2': range(0, 6),
        # 'is_unbalance': True,
    }
    params_class = {
        # 'task': 'train',
        # 'boosting': 'gbdt',
        # 'objective': 'binary',
        # 'metric': {'auc', },

        'num_leaves': 64,
        'num_threads': 4,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'subsample': 0.8,
        'verbose': 0,
    }
    estimator = LGBMClassifier(params_class, n_jobs=4)
    gsearchCV = GridSearchCV(estimator, param_grid=params_test, scoring='roc_auc', cv=5, verbose=0)
    gsearchCV.fit(train, label)
    print('bset score is\n', gsearchCV.best_score_)
    print(gsearchCV.best_params_)
    best_params = gsearchCV.best_estimator_.get_params()
    for params_name in sorted(params_test.keys()):
        print('\t %s:%r\n' % (params_name, best_params[params_name]))
    return gsearchCV.best_params_


# def gbm_model(train, label, test, test_target):
#     params = gridsearchCV(train, label)
#     params['boosting_type'] = 'gbdt'
#     params['objective'] = 'binary'
#     gbm = LGBMModel(params)
#     gbm.fit(train, label, early_stopping_rounds=30)
#     y_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)
#     print(gbm.best_score_)
#     print('feature name:', list(train.columns))
#     print('feature importance: ', gbm.feature_importances_)
#     # print(y_pred)
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