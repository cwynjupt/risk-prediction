import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import operator


def xgb_model(X_train, y_train, X_validation, y_validation, test, test_target):
    '''

    :param X_train:
    :param y_train:
    :param X_validation:
    :param y_validation:
    :param test:
    :param test_target:
    :return:
    '''
    # params = {
    #     'min_child_weight': 100,
    #     'eta': 0.2,
    #     'colsample_bytree': 0.7,
    #     'max_depth': 12,
    #     'subsample': 0.7,
    #     'alpha': 1,
    #     'gamma': 1,
    #     'silent': 1,
    #     'verbose_eval': True,
    #     'seed': 12,
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'auc',
    # }

    params = {
        # 'boosting_type': 'gbdt',
        # 'objective': 'binary',
        # 'metric': {'auc'},
        # 'num_leaves': 128,
        # 'learning_rate': 0.08,
        # 'feature_fraction': 0.8,
        # 'bagging_fraction': 0.8,
        # 'bagging_freq': 10,
        # 'verbose': 0,
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'max_depth': 4,
        'lambda': 130,
        'eta': 0.3,
        'silent': 1,
        'min_child_weight': 4,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'auc'
    }

    rounds = 500
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgvalidation = xgb.DMatrix(X_validation, label=y_validation)
    evallist = [(xgvalidation, 'eval'), (xgtrain, 'train')]
    bst = xgb.train(params, xgtrain, evals=evallist, num_boost_round=rounds, early_stopping_rounds=20)

    xgtest = xgb.DMatrix(test)
    print('bstiteration:', bst.best_iteration)
    y_pred = bst.predict(xgtest, ntree_limit=bst.best_iteration)
    test_target['FORTARGET'] = y_pred
    test_target['FORTARGET'] = test_target['FORTARGET'].apply(lambda x: int(x * 2))
    test_target['PROB'] = y_pred
    test_target.to_csv('result_xgb.csv', index=False)

    features = [x for x in X_train.columns if x not in ['EID']]
    importance = bst.get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore']/df['fscore'].sum()

    print('Calculate feature importance...')
    print('Feature name:', df['feature'])
    print('Feature importance:', df['fscore'])

    # df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 8))
    # plt.title('xgboost feature importance')
    # plt.xlabel('relative importance')
    # plt.show()




