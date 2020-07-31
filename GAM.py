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
state = np.random.randint(0, 100)
print state
Xtrain, XVal, Ytrain, YVal = train_test_split(predictors, y, random_state=state)
lasso = Lasso(alpha = 0.01)
lasso.fit(predictors, y)
clf = LinearRegression()


rfe = RFE(lasso, 50)
fit = rfe.fit(Xtrain, Ytrain)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

rfe2 = RFE(clf, 50)
fit2 = rfe2.fit(Xtrain, Ytrain)
print("Num Features: %s" % (fit2.n_features_))
print("Selected Features: %s" % (fit2.support_))
print("Feature Ranking: %s" % (fit2.ranking_))






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

print xNewTrain
print xNewVal

plt.figure(figsize=(50,150))
cor = xNewTrain.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

for i in predictor_lst:
    cor_target = abs(cor[i])
#Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.529]
    print "relavant features for %s are:" %i
    print relevant_features
    print np.dtype(relevant_features)

print relevant_features
#pol = l(8)
#for i in potential:
#    pol+= l(i)
#    pol += s(i)
    
lam1 = 0.3
lam2 = 0.5
lam3 = 0.7
d =  'categorical'
pol = s(10, lam = lam3) + s(12, lam = lam3) + s(91, lam = lam1, dtype = d) + l(8, lam = lam3)
pol += l(9, lam = lam3) + l(11, lam = lam3) + l(13, lam = lam3) + l(14, lam = lam3)
pol += l(15, lam = lam3) + l(17, lam = lam3)
pol += l(29, lam = lam1) + l(35, lam = lam1) + l(49, lam = lam1) + l(61, lam = lam2)
pol += l(37, lam = lam2) + s(21, lam = lam1) + l(16, lam = lam1) + l(7, lam = lam1)
pol += l(60, lam = lam1) + l(67, lam = lam1)

pol += te(3, 4, lam = lam3) + te(5, 6)
pol += te(29, 37) + te(29, 38)
pol += te(35, 36)+ te(35, 83) + te(37, 83)

pol += s(3, lam=lam2) + s(4, lam = lam3) + s(5, lam = lam3)



#lasso.fit(Xtrain, Ytrain)
#predicted_lasso = lasso.predict(XVal)
#numpy.savetxt('sameerSubmission.csv', predicted_lasso, delimiter=',', header = 'y')

print pol

gam = GAM(pol).fit(Xtrain, Ytrain)
print gam
gam2 = GAM().fit(xNewTrain, Ytrain)
print gam2
#gam2 = GAM(poly).fit(Xtrain, Ytrain)
#print gam2
predicted_gam = gam.predict(XVal)
predicted_gam2 = gam2.predict(xNewVal)
gam3 = GAM().fit(Xtrain, Ytrain)
print gam3
predicted_gam3 = gam3.predict(XVal)
#lam = np.logspace(-3, 5, 5)
#lams = [lam] * 108
#
#gam.gridsearch(Xtrain, Ytrain, lam=lams)
#gam.summary()

#np.savetxt('sameerSubmission.csv', predicted_gam, delimiter=',', header = 'y')


#mse1 = metrics.mean_squared_error(YVal, predicted_lasso)
#rmse1 = math.sqrt(mse1)
#print mse1
#print rmse1

mse2 = metrics.mean_squared_error(YVal, predicted_gam)
rmse2 = math.sqrt(mse2)
print mse2
print rmse2

mse3 = metrics.mean_squared_error(YVal, predicted_gam2)
rmse3 = math.sqrt(mse3)
print mse3
print rmse3

mse4 = metrics.mean_squared_error(YVal, predicted_gam3)
rmse4 = math.sqrt(mse4)
print mse4
print rmse4