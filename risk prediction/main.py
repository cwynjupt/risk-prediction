import pandas as pd
from data_process import firstdeal
from model_process import forest_model
from model_process import regression_model
import matplotlib.pyplot as plt
import seaborn as sns

train_eid, train_feature, train_target, evaluation_eid, evaluation_feature = firstdeal.get_data()
# print(train_eid.head(100))
# print(train_feature.head(100))
# print(train_target.head(100))
regression_model.linear_regression(train_eid, train_feature, train_target, evaluation_eid, evaluation_feature)
regression_model.logistic_regression(train_eid, train_feature, train_target, evaluation_eid, evaluation_feature)