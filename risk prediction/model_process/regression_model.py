from data_process import firstdeal
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np


def linear_regression(train_eid, train_feature, train_target, evaluation_eid, evaluation_feature):
    alg = LinearRegression()
    kf = KFold(train_feature.shape[0], n_folds=3, random_state=1)
    train_predictions = []
    for train, test in kf:
        train_feature_kf_train = (train_feature.iloc[train, :])
        train_target_kf_train = (train_target.iloc[train])
        alg.fit(train_feature_kf_train, train_target_kf_train)
        train_feature_kf_test = (train_feature.iloc[test, :])
        train_target_kf_test = alg.predict(train_feature_kf_test)
        train_predictions.append(train_target_kf_test)

    train_predictions = np.concatenate(train_predictions, axis=0)
    train_predictions[train_predictions>.5] = 1
    train_predictions[train_predictions<=.5] = 0
    accuracy = sum(train_predictions[train_predictions == train_target])/len(train_predictions)
    print(train_predictions)
    print('the accuracy of linear_regression is:', accuracy)


from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression   # 逻辑回归


def logistic_regression(train_eid, train_feature, train_target, evaluation_eid, evaluation_feature):
    # Initialize our algorithm
    alg=LogisticRegression(random_state=1)
    # Compute the accuracy score for all the cross validation folds.(much simpler than what we did before!)
    scores = cross_validation.cross_val_score(alg,train_feature,train_target,cv=3)
    # Take the mean of the scores (because we have one for each fold)
    print(scores.mean())


