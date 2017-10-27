from sklearn.linear_model import LogisticRegression   # 逻辑回归
from sklearn.base import ClassifierMixin


def logistic_regression(X_train, y_train, X_validation, y_validation, test, test_target):
    alg=LogisticRegression(random_state=1)
    alg.fit(X_train, y_train)
    y_validation_pred = alg.predict(X_validation)
    auc_score = ClassifierMixin.score(y_validation, y_validation_pred)





