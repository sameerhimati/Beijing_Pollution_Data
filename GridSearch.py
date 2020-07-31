#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 01:00:10 2019

@author: sameerhimati
"""
import csv
from sklearn.model_selection import GridSearchCV,  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from pygam import LinearGAM, s, f, te, GAM, l
from sklearn import metrics
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns

train = pd.read_csv("trainingdata2.csv")
test = pd.read_csv("test_predictors2.csv")
train.head()
test.head()

y = train['y'].values
y = y.reshape(-1, 1)
predictors = train.drop(['y', 'X8'], axis = 1)
test_predictors = test.drop(['id', 'X8'], axis = 1)
#state = np.random.randint(0, 100)
#print state
#Xtrain, XVal, Ytrain, YVal = train_test_split(predictors, y, random_state=state)

lasso = Lasso(alpha = 0.01)
lasso.fit(predictors, y)
clf = LinearRegression()


rfe = RFE(lasso, 20)
fit = rfe.fit_transform(predictors, y)
lasso.fit(fit, y)
print("Selected Features: %s" % (rfe.support_))
print("Feature Ranking: %s" % (rfe.ranking_))

rfe2 = RFE(clf, 20)
fit2 = rfe2.fit_transform(predictors, y)
clf.fit(fit2, y)
#print("Num Features: %s" % (fit2.n_features_))
print("Selected Features: %s" % (rfe2.support_))
print("Feature Ranking: %s" % (rfe2.ranking_))

#no of features
nof_list=np.arange(1,109)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(predictors, y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))


#no of features
nof_list=np.arange(1,109)            
high_score=0
#Variable to store the optimum features
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(predictors, y, test_size = 0.3, random_state = 0)
    model = Lasso(alpha = 0.01)
    rfe = RFE(model,nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))