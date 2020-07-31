#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:00:10 2019

@author: sameerhimati
"""
import csv
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from pygam import LogisticGAM
from pygam import LinearGAM, s, f, te, GAM
from sklearn import metrics
import numpy
import pandas as pd
import math
import matplotlib.pyplot as plt

train = pd.read_csv("trainingdata2.csv")
test = pd.read_csv("test_predictors2.csv")
train.head()
test.head()

y = train['y'].values
y = y.reshape(-1, 1)
predictors = train.drop(['y', 'X8'], axis = 1)
print predictors
test_predictors = test.drop(['id', 'X8'], axis = 1)
print test

lasso = Lasso(alpha = 0.01)
lasso.fit(predictors, y)

predicted_lasso = lasso.predict(predictors)
print predicted_lasso
#numpy.savetxt('sameerSubmission.csv', predicted_lasso, delimiter=',', header = 'y')

gam = LinearGAM().fit(predictors, y)
print gam
gam2 = GAM().gridsearch(predictors, y)
print gam2
predicted_gam = gam.predict(predictors)
m = numpy.mean(y)


mse1 = metrics.mean_squared_error(y, predicted_lasso)
rmse1 = math.sqrt(mse1)
print m
print mse1
print rmse1

mse2 = metrics.mean_squared_error(y, predicted_gam)
rmse2 = math.sqrt(mse2)
print mse2
print rmse2