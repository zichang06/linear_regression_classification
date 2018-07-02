# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:12:10 2018

@author: 曹蕊
"""

import pandas as pd   
import numpy as np
import random
from sklearn import linear_model
from sklearn.linear_model import LinearRegression  


# read in data
train = pd.read_csv('data/train.csv') 
#random.shuffle(train)
train_X = train.iloc[:, 1:385]
train_y = train.reference 
test = pd.read_csv('data/test.csv') 
test_X = train.iloc[:, 1:385]

train_X = train_X.drop(['value002'], 1)
train_X = train_X.drop(['value003'], 1)
train_X = train_X.drop(['value008'], 1)
train_X = train_X.drop(['value017'], 1)
train_X = train_X.drop(['value019'], 1)
train_X = train_X.drop(['value022'], 1)

train_X = train_X.drop(['value027'], 1)
train_X = train_X.drop(['value028'], 1)
train_X = train_X.drop(['value031'], 1)
train_X = train_X.drop(['value039'], 1)
train_X = train_X.drop(['value044'], 1)
train_X = train_X.drop(['value053'], 1)
train_X = train_X.drop(['value061'], 1)

train_X = train_X.drop(['value062'], 1)
train_X = train_X.drop(['value069'], 1)
train_X = train_X.drop(['value073'], 1)
train_X = train_X.drop(['value086'], 1)
train_X = train_X.drop(['value089'], 1)
train_X = train_X.drop(['value096'], 1)
train_X = train_X.drop(['value099'], 1)

train_X = train_X.drop(['value103'], 1)
train_X = train_X.drop(['value105'], 1)
train_X = train_X.drop(['value106'], 1)
train_X = train_X.drop(['value111'], 1)
train_X = train_X.drop(['value118'], 1)
train_X = train_X.drop(['value121'], 1)
train_X = train_X.drop(['value131'], 1)

train_X = train_X.drop(['value135'], 1)
train_X = train_X.drop(['value139'], 1)
train_X = train_X.drop(['value140'], 1)
train_X = train_X.drop(['value147'], 1)
train_X = train_X.drop(['value148'], 1)
train_X = train_X.drop(['value153'], 1)
train_X = train_X.drop(['value154'], 1)

train_X = train_X.drop(['value156'], 1)
train_X = train_X.drop(['value170'], 1)
train_X = train_X.drop(['value174'], 1)
train_X = train_X.drop(['value186'], 1)
train_X = train_X.drop(['value188'], 1)
train_X = train_X.drop(['value191'], 1)
train_X = train_X.drop(['value192'], 1)

train_X = train_X.drop(['value193'], 1)
train_X = train_X.drop(['value199'], 1)
train_X = train_X.drop(['value202'], 1)
train_X = train_X.drop(['value203'], 1)
train_X = train_X.drop(['value208'], 1)
train_X = train_X.drop(['value215'], 1)
train_X = train_X.drop(['value225'], 1)

train_X = train_X.drop(['value228'], 1)
train_X = train_X.drop(['value241'], 1)
train_X = train_X.drop(['value242'], 1)
train_X = train_X.drop(['value251'], 1)
train_X = train_X.drop(['value252'], 1)
train_X = train_X.drop(['value255'], 1)
train_X = train_X.drop(['value260'], 1)

train_X = train_X.drop(['value261'], 1)
train_X = train_X.drop(['value294'], 1)
train_X = train_X.drop(['value295'], 1)
train_X = train_X.drop(['value301'], 1)
train_X = train_X.drop(['value306'], 1)
train_X = train_X.drop(['value309'], 1)
train_X = train_X.drop(['value318'], 1)

train_X = train_X.drop(['value323'], 1)
train_X = train_X.drop(['value324'], 1)
train_X = train_X.drop(['value335'], 1)
train_X = train_X.drop(['value337'], 1)
train_X = train_X.drop(['value347'], 1)
train_X = train_X.drop(['value369'], 1)
train_X = train_X.drop(['value370'], 1)

train_X = train_X.drop(['value372'], 1)
train_X = train_X.drop(['value376'], 1)
train_X = train_X.drop(['value377'], 1)
train_X = train_X.drop(['value379'], 1)
train_X = train_X.drop(['value383'], 1)

test_X = test_X.drop(['value002'], 1)
test_X = test_X.drop(['value003'], 1)
test_X = test_X.drop(['value008'], 1)
test_X = test_X.drop(['value017'], 1)
test_X = test_X.drop(['value019'], 1)
test_X = test_X.drop(['value022'], 1)

test_X = test_X.drop(['value027'], 1)
test_X = test_X.drop(['value028'], 1)
test_X = test_X.drop(['value031'], 1)
test_X = test_X.drop(['value039'], 1)
test_X = test_X.drop(['value044'], 1)
test_X = test_X.drop(['value053'], 1)
test_X = test_X.drop(['value061'], 1)

test_X = test_X.drop(['value062'], 1)
test_X = test_X.drop(['value069'], 1)
test_X = test_X.drop(['value073'], 1)
test_X = test_X.drop(['value086'], 1)
test_X = test_X.drop(['value089'], 1)
test_X = test_X.drop(['value096'], 1)
test_X = test_X.drop(['value099'], 1)

test_X = test_X.drop(['value103'], 1)
test_X = test_X.drop(['value105'], 1)
test_X = test_X.drop(['value106'], 1)
test_X = test_X.drop(['value111'], 1)
test_X = test_X.drop(['value118'], 1)
test_X = test_X.drop(['value121'], 1)
test_X = test_X.drop(['value131'], 1)

test_X = test_X.drop(['value135'], 1)
test_X = test_X.drop(['value139'], 1)
test_X = test_X.drop(['value140'], 1)
test_X = test_X.drop(['value147'], 1)
test_X = test_X.drop(['value148'], 1)
test_X = test_X.drop(['value153'], 1)
test_X = test_X.drop(['value154'], 1)

test_X = test_X.drop(['value156'], 1)
test_X = test_X.drop(['value170'], 1)
test_X = test_X.drop(['value174'], 1)
test_X = test_X.drop(['value186'], 1)
test_X = test_X.drop(['value188'], 1)
test_X = test_X.drop(['value191'], 1)
test_X = test_X.drop(['value192'], 1)

test_X = test_X.drop(['value193'], 1)
test_X = test_X.drop(['value199'], 1)
test_X = test_X.drop(['value202'], 1)
test_X = test_X.drop(['value203'], 1)
test_X = test_X.drop(['value208'], 1)
test_X = test_X.drop(['value215'], 1)
test_X = test_X.drop(['value225'], 1)

test_X = test_X.drop(['value228'], 1)
test_X = test_X.drop(['value241'], 1)
test_X = test_X.drop(['value242'], 1)
test_X = test_X.drop(['value251'], 1)
test_X = test_X.drop(['value252'], 1)
test_X = test_X.drop(['value255'], 1)
test_X = test_X.drop(['value260'], 1)

test_X = test_X.drop(['value261'], 1)
test_X = test_X.drop(['value294'], 1)
test_X = test_X.drop(['value295'], 1)
test_X = test_X.drop(['value301'], 1)
test_X = test_X.drop(['value306'], 1)
test_X = test_X.drop(['value309'], 1)
test_X = test_X.drop(['value318'], 1)

test_X = test_X.drop(['value323'], 1)
test_X = test_X.drop(['value324'], 1)
test_X = test_X.drop(['value335'], 1)
test_X = test_X.drop(['value337'], 1)
test_X = test_X.drop(['value347'], 1)
test_X = test_X.drop(['value369'], 1)
test_X = test_X.drop(['value370'], 1)

test_X = test_X.drop(['value372'], 1)
test_X = test_X.drop(['value376'], 1)
test_X = test_X.drop(['value377'], 1)
test_X = test_X.drop(['value379'], 1)
test_X = test_X.drop(['value383'], 1)
'''
train_X = train_X.drop(['value002','value003','value008','value017','value019','value022'
,'value027','value028','value031','value039','value044','value053','value061',
'value062','value069','value073','value086','value089','value096','value099'
,'value103','value105','value106','value111','value118','value121','value131'
,'value135','value139','value140','value147','value148','value153','value154'
,'value156','value170','value174','value186','value188','value191','value192'
,'value193','value199','value202','value203','value208','value215','value225'
,'value228','value241','value242','value251','value252','value255','value260'
,'value261','value294','value295','value301','value306','value309','value318'
,'value323','value324','value335','value337','value347','value369','value370'
,'value372','value376','value377','value379','value383'], 1)
test_X = train_X.drop(['value002','value003','value008','value017','value019','value022'
,'value027','value028','value031','value039','value044','value053','value061',
'value062','value069','value073','value086','value089','value096','value099'
,'value103','value105','value106','value111','value118','value121','value131'
,'value135','value139','value140','value147','value148','value153','value154'
,'value156','value170','value174','value186','value188','value191','value192'
,'value193','value199','value202','value203','value208','value215','value225'
,'value228','value241','value242','value251','value252','value255','value260'
,'value261','value294','value295','value301','value306','value309','value318'
,'value323','value324','value335','value337','value347','value369','value370'
,'value372','value376','value377','value379','value383'], 1)
'''

'''
## linear regression edition
linreg = linear_model.LinearRegression()  
model=linreg.fit(train_X, train_y) 
y_pred = linreg.predict(test_X)  

head = ["reference"]
y_pred = pd.DataFrame (y_pred , columns = head)
y_pred.to_csv ("y_predect.csv" , encoding = "utf-8")
'''

## regularization edition with validation set (no cross validaion)
alphaArray = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
trai_num = 20000
vali_num = 5000
mse = []
trainX_trainx =  train_X[:trai_num]
trainX_trainy = train_y[:trai_num]
trainX_validationx =  train_X[trai_num:]
trainX_validationy = train_y[trai_num:]

for alphaVal in alphaArray:
    reg = linear_model.Ridge (alpha = alphaVal)
    model=reg.fit(trainX_trainx, trainX_trainy) 
    trainX_validation_predY = reg.predict(trainX_validationx)  
    errorArror = trainX_validation_predY - trainX_validationy
    errorMse = pow(np.linalg.norm(errorArror, ord=2), 2) / vali_num
    mse.append(errorMse)

alphaVal = alphaArray[mse.index(min(mse))]
reg = linear_model.Ridge (alpha = alphaVal)
model=reg.fit(train_X, train_y) 
y_pred = reg.predict(test_X)  

head = ["reference"]
y_pred = pd.DataFrame (y_pred , columns = head)
y_pred.to_csv ("y_predect.csv" , encoding = "utf-8")