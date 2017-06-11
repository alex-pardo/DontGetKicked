import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, model_selection

T = pd.read_csv('./training.csv')

text_cols = [column for column in T.columns if T[column].dtype not in ['int64', 'float64']]
for col in text_cols:
    print col
    lb = preprocessing.LabelBinarizer()
    T[col].fillna('-', inplace=True)
    T[col].apply(str)
    T[col] = lb.fit_transform(T[col])


y = np.array(T['IsBadBuy'])
X = np.array(T.drop('IsBadBuy', axis=1))

X = np.where(np.isnan(X), np.ma.array(X, mask=np.isnan(X)).mean(axis=0), X)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

kf = model_selection.KFold(n_splits=3)
for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    sc = preprocessing.StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = svm.SVC()

    clf = model_selection.GridSearchCV(clf, parameters, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)
    break