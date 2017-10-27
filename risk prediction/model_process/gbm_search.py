import lightgbm as lgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.grid_search import GridSearchCV


def print_best_score(gsearch, param_test):
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def lightGBM_CV(feature, label):
    # data, labels = make_train_set(24000000, 25000000)
    # values = data.values
    param_test = {
        'max_depth': range(5, 15, 2),
        'num_leaves': range(10, 40, 5),
    }
    estimator = LGBMRegressor(
        num_leaves=50,  # cv调节50是最优值
        max_depth=13,
        learning_rate=0.1,
        n_estimators=1000,
        objective='regression',
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=7,
    )
    gsearch = GridSearchCV(estimator, param_grid=param_test, scoring='roc_auc', cv=5)
    gsearch.fit(feature, label)   # 运行网格搜索
    print(gsearch.best_params_)    # 最佳结果的参数组合
    print(gsearch.best_score_)      # 最好的评分
    print_best_score(gsearch, param_test)


def gbm_search(feature, label):
    lightGBM_CV(feature, label)