import lightgbm as lgb
from sklearn.model_selection import train_test_split


def gbm_model(label, train, test, test_target):
    '''
    :param label: 样本标签
    :param train: 训练样本
    :param test: 测试样本
    :param test_target: 测试样本目标
    :return:
    '''
    X_train, y_train, X_validation, y_validation = train_test_split(train, label, test_size=0.3, random_state=10000)

    lgb_train = lgb.Dataset(X_train, X_validation)
    lgb_eval = lgb.Dataset(y_train, y_validation)

    params = {
        'task': 'train',
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': {'auc', },
        'num_leaves': 2 ** 6,
        'learning_rate': 0.05,
        'verbose': 0,
    }
    print('Start training...')
    gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=30)
    y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

    test_target['PROB'] = y_pred
    test_target['FORTARGET'] = test_target['PROB'].apply(lambda x: int(x * 2))
    test_target.to_csv('result_gbm.csv', index=False)

    print('Calculate feature importance...')
    print('Feature name:', gbm.feature_name())
    print('Feature importance:', list(gbm.feature_importance()))
