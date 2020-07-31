#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:00:10 2019

@author: sameerhimati
"""
import csv
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV,  train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor, MLPClassifier
from pygam import LogisticGAM
from pygam import LinearGAM, s, f, te, GAM, ExpectileGAM
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn import svm
import pandas
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
import math
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.feature_selection import RFE

train = pd.read_csv("trainingdata2.csv")
test = pd.read_csv("test_predictors2.csv")
train.head()
test.head()
tol = 1e-10
max_iter = 10000
y = train['y'].values
y = y.reshape(-1, 1)
predictors = train.drop(['y', 'X8'], axis = 1)
test_predictors = test.drop(['id', 'X8'], axis = 1)
state = np.random.randint(0, 100)
Xtrain, XVal, Ytrain, YVal = train_test_split(predictors, y, random_state=state)
lasso = Lasso(alpha = 0.01)
lasso.fit(predictors, y)
predicted_lasso = lasso.predict(test_predictors)




ridge = Ridge(alpha = 0.01)
ridge.fit(Xtrain, Ytrain)
predicted_ridge = ridge.predict(XVal)
#numpy.savetxt('sameerSubmission.csv', predicted_lasso, delimiter=',', header = 'y')

gam = GAM().fit(Xtrain, Ytrain)
gam2 = ExpectileGAM(s(10) + s(15) + s(102)).fit(Xtrain, Ytrain)
print gam
#
#x = Xtrain.values
##gam3 = GAM().gridsearch(x, Ytrain, weights = lasso.coef_)
#print gam
#print gam2
predicted_gam = gam.predict(XVal)
predicted_gam2 = gam2.predict(XVal)

rfe = RFE(lasso, 56)
fit = rfe.fit(Xtrain, Ytrain)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

target = fit.ranking_
potential = []
for i in range(0, len(target)):
    if target[i] == 1:
        potential.append(i)
        print i

predictor_lst = []
for i in potential:
    predictor_lst.append('X%d' % (i+2))

print predictor_lst
print len(predictor_lst)

xNewTrain = Xtrain.filter(items = predictor_lst)
xNewVal = XVal.filter(items = predictor_lst)


classifiers = [
    linear_model.LassoCV(),
    linear_model.ElasticNetCV(),
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, loss='huber', alpha = 0.1),
    linear_model.ARDRegression(alpha_1 = 25, alpha_2 = 25),
    linear_model.LinearRegression(),
    linear_model.MultiTaskElasticNetCV(),
    linear_model.MultiTaskLassoCV(),]

for item in classifiers:
    print(item)
    clf = item
    clf.fit(xNewTrain, Ytrain)
    mse = metrics.mean_squared_error(YVal, clf.predict(xNewVal))
    rmse = math.sqrt(mse)
    print mse
    print rmse
    print "\n"
    
ard = linear_model.ARDRegression(alpha_1 = 25, alpha_2 = 25)
ard.fit(predictors, y)
predicted_ard = ard.predict(test_predictors)
np.savetxt('sameerSubmission.csv', predicted_lasso, delimiter=',', header = 'y')

m = np.mean(y)
mse1 = metrics.mean_squared_error(YVal, predicted_lasso)
rmse1 = math.sqrt(mse1)
print m
print mse1
print rmse1

mse2 = metrics.mean_squared_error(YVal, predicted_ridge)
rmse2 = math.sqrt(mse2)
print mse2
print rmse2

#mse3 = metrics.mean_squared_error(YVal, predicted_gbc)
#rmse3 = math.sqrt(mse3)
#print mse3
#print rmse3
mse3 = metrics.mean_squared_error(YVal, predicted_gam)
rmse3 = math.sqrt(mse3)
print mse3
print rmse3
#
mse3 = metrics.mean_squared_error(YVal, predicted_gam2)
rmse3 = math.sqrt(mse3)
print mse3
print rmse3