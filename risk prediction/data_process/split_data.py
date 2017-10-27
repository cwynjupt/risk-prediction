from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split


def splits_data(X, y):
    X_train, X_validation,  y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=10000)
    return X_train, y_train, X_validation, y_validation
    # skf = StratifiedKFold(n_splits=3)
    # X_split, y_split = skf.split(X, y)
    # X_train, y_train, X_validation, y_validation = X_split[1, :], y_split[1, :], X_split[2, :], y_split[2, :]
    # return X_train, y_train, X_validation, y_validation
