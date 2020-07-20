# -*- coding: utf-8 -*-
# @Time: 2020/3/25 15:07
# @Author:
import time

from sklearn import metrics
from xgboost.sklearn import XGBRegressor

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations
bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')
preds = bst.predict(dtest)
import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])

from sklearn.metrics import precision_score

print(precision_score(y_test, best_preds, average='macro'))

exit()

start = time.time()
reg = XGBRegressor()
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'XGBRegressor'
history.append([name,loss,end-start])

start = time.time()
reg = XGBRegressor(max_depth=4, n_estimators=500, min_child_weight=10,
				   subsample=0.7, colsample_bytree=0.7, reg_alpha=0, reg_lambda=0.5)
reg.fit(X_train, y_train)
end = time.time()
y_pred = reg.predict(X_test)
loss = metrics.mean_squared_error(y_test, y_pred)
name = 'XGBRegressor_s'
history.append([name,loss,end-start])