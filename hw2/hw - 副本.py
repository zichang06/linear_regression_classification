import pandas as pd   
import numpy as np
import sklearn
from sklearn import preprocessing  
import csv
import random
import sys
import math

# trainSampleNum = 1719692
# testSampleNum = 449923
# train_dir = "train.txt"
# test_dir = "test.txt"
# alpha = 0.001 
# lambdaR = 1
# batchSize = 10000 
# epochMax = 100

trainSampleNum = 50
testSampleNum = 20
train_dir = "simple/train.txt"
test_dir = "simple/test.txt"
alpha = 0.001 
lambdaR = 1
batchSize = 10 
epochMax = 1000000


featureNum = 201
epsilon = 0.00001

def getData(dataDir = "train.txt", isTrain = True):
    '''
    readin dataset according to dataDir
    if train dataset is needed
    return augmented features[m, n+1] and label[m]
    if test dataset is needed
    return augmented features[m, n+1]

    '''
    if isTrain:
        features = np.zeros((trainSampleNum, featureNum + 1))
        for i in range(trainSampleNum):
            features[i][0] = 1
        if isTrain:
            label = np.zeros((trainSampleNum))
    else:
        features = np.zeros((testSampleNum, featureNum + 1))
        for i in range(testSampleNum):
            features[i][0] = 1
        if isTrain:
            label = np.zeros((testSampleNum))

    f = open(dataDir)  
    lines = f.readline() 
    sampleCount = 0
    while lines:  
        line = lines.split(' ')
        for index in range(len(line)):
            if index == 0:
                if isTrain:
                    label[sampleCount] = int(line[0])
                continue
            colon = line[index].index(':')
            features[sampleCount, int(line[index][:colon])] = float(line[index][colon+1:])
        lines = f.readline()  
        sampleCount += 1
    f.close()  
    print("finish load data %s" %(dataDir))
    
    if isTrain:
        return features, label 
    return features

class logisticRegression:
    def __init__(self, featureNum = 201):
        self.dim = featureNum+1
        self.theta = np.random.random(size = self.dim)

    def predict(self, x):
        yP = np.dot(x, self.theta)
        return yP

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def lossFunction(self, hypothesis, y):
        '''
        hypothesis: calculated hypothesis by model
        y: real value
        return the mean of one-batch loss
        '''
        hypothesis = np.clip(hypothesis, 10e-8, 1.0-10e-8)
        entropys = - y * np.log(hypothesis) - (1 - y) * np.log(1 - hypothesis)
        loss = np.mean(entropys)
        return loss

    def trainWithMiniBatch(self, features, label, alpha, lambdaR, batchSize, epochMax):
        '''
        every time load in a minibatch to train
        '''
        stepPerEpoch = math.ceil(trainSampleNum / batchSize) 
        currentLoss = 0
        lastLossPerEpoch = 0

        for epoch in range(epochMax):
            lossPerEpoch = 0
            for stepInEpoch in range(stepPerEpoch):
                start = stepInEpoch*batchSize
                end = min((stepInEpoch+1)*batchSize, trainSampleNum)
                currentLoss = self.oneStepGradientDescentRegulazed(x = features[start:end], 
                                                              y = label[start:end], 
                                                              alpha = alpha, 
                                                              lambdaR = lambdaR)
                #sys.stdout.write('\r>>epoch %d, step %d/%d, start = %d end = %d -- loss = %f .' % (epoch+1, stepInEpoch+1, stepPerEpoch, start, end, currentLoss))         
                lossPerEpoch += currentLoss
            if epoch % 2000 == 0:
                print('>>epoch %d, loss = %f .' % (epoch+1, currentLoss))               
            diff = abs(lastLossPerEpoch - lossPerEpoch)   
            lastLossPerEpoch = lossPerEpoch                              
            if diff < epsilon:
                return epoch+1
        return epoch+1
            


    def oneStepGradientDescentRegulazed(self, x, y, alpha, lambdaR):
        '''
        for a minibatch dataset 
        do gradient descent for a step 
        return the loss for this step
        '''
        z = np.dot(x, self.theta)
        hypothesis = self.sigmoid(z)
        error = hypothesis - y
        loss = self.lossFunction(hypothesis, y)
    
        x_transpose =  x.T
        item2 = np.dot(x_transpose, error) * alpha / featureNum

        scalar = 1 - alpha * lambdaR / featureNum
        tmp = self.theta[0]
        self.theta *= scalar
        self.theta[0] = tmp

        self.theta -= item2
        return loss
        

def writeCSV(predictLable):
    head = ["label"]
    y_pred = pd.DataFrame (predictLable , columns = head)
    y_pred.to_csv ("predictLable.csv" , encoding = "utf-8")


if __name__ == '__main__':
    trainFeatures, trainLabel  = getData(dataDir = train_dir, isTrain = True)
    min_max_scaler = preprocessing.MinMaxScaler()  
    trainFeatures = min_max_scaler.fit_transform(trainFeatures)  

    lr = logisticRegression()
    epoch, step = lr.trainWithMiniBatch(features = trainFeatures, 
                          label = trainLabel, 
                          alpha = alpha, 
                          lambdaR = lambdaR, 
                          batchSize = batchSize, 
                          epochMax = epochMax)
    print("finish training, with epoch = %d and step = %d." %(epoch, step))

    testFeatures  = getData(dataDir = test_dir, isTrain = False)
    testFeatures = min_max_scaler.fit_transform(testFeatures)

    print("start predict...")
    predictLable = lr.predict(testFeatures)

    print("write the result to csv and save...")
    writeCSV(predictLable)