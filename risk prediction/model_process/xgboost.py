import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV


def xgb_model(label, train, test, test_target):
    '''

    :param label:
    :param train:
    :param test:
    :param test_target:
    :return:
    '''
    X_train, y_train, X_validation, y_validation = train_test_split(train, label, random_state=10000, test_size=0.3)
    params = {
        'objective': 'binary',
        'eval_metric': 'auc',


    }

