import pandas as pd   
import numpy as np
import random
import sys
from numpy import genfromtxt

epsilon = 0.0001  

def getData(dataDir, featureNum):
    data = pd.read_csv(dataDir) 
    data.sample(frac=1)
    features = data.iloc[:, 1:featureNum]
    label = data.iloc[:, featureNum:]
    return features, label

def batchGradientDescentRegulazed(x, y, alpha, lambdaR, maxIterations):
    print("\nstart regulazed gradient descenting")
    m, n = np.shape(x)
    theta = np.zeros(n) 
    lastCost = 0
    cost = 0 

    iterNum = 0
    while iterNum < maxIterations:
        iterNum += 1
        hypothesis = np.array(np.dot(x, theta))
        loss = hypothesis - y

        lastCost = cost
        cost = np.linalg.norm(loss, ord=2)
        diff = abs(cost - lastCost)    
        if diff < epsilon:
            break
        sys.stdout.write('\r>> rate of progress %d/%d, cost = %f, cost difference = %f' % (iterNum, maxIterations, cost, diff))
    
        x_transpose =  x.T
        item2 = np.dot(x_transpose, loss) * alpha / m

        scalar = 1 - alpha * lambdaR / m
        tmp = theta[0]
        theta *= scalar
        theta[0] = tmp

        theta -= item2

    print("\nfinish training with alpha = %f, lambda = %f ,taltal Iterations = %d ." %(alpha, lambdaR, iterNum))
    return theta

def predict(x, theta):
    yP = np.dot(x, theta)
    return yP

def trainWithCrossValidation(Features, Label):
    print("\ntrain the model with cross validation...")
    m, n = np.shape(Features)
    lambdaArray = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_num = 20000
    vali_num = 5000
    mse = []
    alpha = 0.01
    maxIterations = 1000000

    for lambdaVal in lambdaArray: 
        errorMse = 0
        print("\nthe regulazation parameter lambda is %f..." %lambdaVal)

        for i in  range(1):
            print("\nnow for %dth validation..." %(i+1))
            start = i * vali_num
            end = start + vali_num 
            
            validationFeatures =  Features[start:end]
            validationLabel = Label[start:end]

            trainFeatures = np.zeros((train_num, n))
            trainLabel = np.zeros((train_num))
            trainFeatures[:start] = Features[:start]
            trainFeatures[start:] = Features[end:]
            trainLabel[:start] = Label[:start]
            trainLabel[start:] = Label[end:]

            theta = batchGradientDescentRegulazed(trainFeatures, trainLabel, alpha, lambdaVal, maxIterations)
            predictLable = predict(validationFeatures, theta)
            tmp = predictLable - validationLabel
            error = np.linalg.norm(tmp, ord=2) / vali_num
            print("the errorValue this turn is %f." %(error))
            errorMse += error
            
        mse.append(errorMse)
        print("the total errorMse for lambda %f is %f..." %(lambdaVal, errorMse))
    
    lambdaR = lambdaArray[mse.index(min(mse))]
    print("train the final model with the choosen lambda %f" %(lambdaR))
    theta = batchGradientDescentRegulazed(Features, Label, alpha, lambdaR, maxIterations)
    return theta


def writeCSV(predictLable):
    head = ["reference"]
    y_pred = pd.DataFrame (predictLable , columns = head)
    y_pred.to_csv ("predictLable.csv" , encoding = "utf-8")


def main():
    print("\nload in train data...")
    trainDataDir = "data/train.csv"
    features_, label_ = getData(trainDataDir, 385)
   
    features_ = features_.drop(['value002'], 1)
    features_ = features_.drop(['value003'], 1)
    features_ = features_.drop(['value008'], 1)
    features_ = features_.drop(['value017'], 1)
    features_ = features_.drop(['value019'], 1)
    features_ = features_.drop(['value022'], 1)

    features_ = features_.drop(['value027'], 1)
    features_ = features_.drop(['value028'], 1)
    features_ = features_.drop(['value031'], 1)
    features_ = features_.drop(['value039'], 1)
    features_ = features_.drop(['value044'], 1)
    features_ = features_.drop(['value053'], 1)
    features_ = features_.drop(['value061'], 1)

    features_ = features_.drop(['value062'], 1)
    features_ = features_.drop(['value069'], 1)
    features_ = features_.drop(['value073'], 1)
    features_ = features_.drop(['value086'], 1)
    features_ = features_.drop(['value089'], 1)
    features_ = features_.drop(['value096'], 1)
    features_ = features_.drop(['value099'], 1)

    features_ = features_.drop(['value103'], 1)
    features_ = features_.drop(['value105'], 1)
    features_ = features_.drop(['value106'], 1)
    features_ = features_.drop(['value111'], 1)
    features_ = features_.drop(['value118'], 1)
    features_ = features_.drop(['value121'], 1)
    features_ = features_.drop(['value131'], 1)

    features_ = features_.drop(['value135'], 1)
    features_ = features_.drop(['value139'], 1)
    features_ = features_.drop(['value140'], 1)
    features_ = features_.drop(['value147'], 1)
    features_ = features_.drop(['value148'], 1)
    features_ = features_.drop(['value153'], 1)
    features_ = features_.drop(['value154'], 1)

    features_ = features_.drop(['value156'], 1)
    features_ = features_.drop(['value170'], 1)
    features_ = features_.drop(['value174'], 1)
    features_ = features_.drop(['value186'], 1)
    features_ = features_.drop(['value188'], 1)
    features_ = features_.drop(['value191'], 1)
    features_ = features_.drop(['value192'], 1)

    features_ = features_.drop(['value193'], 1)
    features_ = features_.drop(['value199'], 1)
    features_ = features_.drop(['value202'], 1)
    features_ = features_.drop(['value203'], 1)
    features_ = features_.drop(['value208'], 1)
    features_ = features_.drop(['value215'], 1)
    features_ = features_.drop(['value225'], 1)

    features_ = features_.drop(['value228'], 1)
    features_ = features_.drop(['value241'], 1)
    features_ = features_.drop(['value242'], 1)
    features_ = features_.drop(['value251'], 1)
    features_ = features_.drop(['value252'], 1)
    features_ = features_.drop(['value255'], 1)
    features_ = features_.drop(['value260'], 1)

    features_ = features_.drop(['value261'], 1)
    features_ = features_.drop(['value294'], 1)
    features_ = features_.drop(['value295'], 1)
    features_ = features_.drop(['value301'], 1)
    features_ = features_.drop(['value306'], 1)
    features_ = features_.drop(['value309'], 1)
    features_ = features_.drop(['value318'], 1)

    features_ = features_.drop(['value323'], 1)
    features_ = features_.drop(['value324'], 1)
    features_ = features_.drop(['value335'], 1)
    features_ = features_.drop(['value337'], 1)
    features_ = features_.drop(['value347'], 1)
    features_ = features_.drop(['value369'], 1)
    features_ = features_.drop(['value370'], 1)

    features_ = features_.drop(['value372'], 1)
    features_ = features_.drop(['value376'], 1)
    features_ = features_.drop(['value377'], 1)
    features_ = features_.drop(['value379'], 1)
    features_ = features_.drop(['value383'], 1)
  
    m, n = np.shape(features_)
    features = np.ones((m, n+1))
    features[:, :-1] = features_
    label = np.array(label_).flatten()

    print("\nstart training...")
    theta = trainWithCrossValidation(features, label)

    print("\nload in test data...")
    testDataDir = "data/test.csv"
    testFeatures_ = pd.read_csv(testDataDir) 
    testFeatures_ = testFeatures_.iloc[:, 1:385]
    
    testFeatures_ = testFeatures_.drop(['value002'], 1)
    testFeatures_ = testFeatures_.drop(['value003'], 1)
    testFeatures_ = testFeatures_.drop(['value008'], 1)
    testFeatures_ = testFeatures_.drop(['value017'], 1)
    testFeatures_ = testFeatures_.drop(['value019'], 1)
    testFeatures_ = testFeatures_.drop(['value022'], 1)

    testFeatures_ = testFeatures_.drop(['value027'], 1)
    testFeatures_ = testFeatures_.drop(['value028'], 1)
    testFeatures_ = testFeatures_.drop(['value031'], 1)
    testFeatures_ = testFeatures_.drop(['value039'], 1)
    testFeatures_ = testFeatures_.drop(['value044'], 1)
    testFeatures_ = testFeatures_.drop(['value053'], 1)
    testFeatures_ = testFeatures_.drop(['value061'], 1)

    testFeatures_ = testFeatures_.drop(['value062'], 1)
    testFeatures_ = testFeatures_.drop(['value069'], 1)
    testFeatures_ = testFeatures_.drop(['value073'], 1)
    testFeatures_ = testFeatures_.drop(['value086'], 1)
    testFeatures_ = testFeatures_.drop(['value089'], 1)
    testFeatures_ = testFeatures_.drop(['value096'], 1)
    testFeatures_ = testFeatures_.drop(['value099'], 1)

    testFeatures_ = testFeatures_.drop(['value103'], 1)
    testFeatures_ = testFeatures_.drop(['value105'], 1)
    testFeatures_ = testFeatures_.drop(['value106'], 1)
    testFeatures_ = testFeatures_.drop(['value111'], 1)
    testFeatures_ = testFeatures_.drop(['value118'], 1)
    testFeatures_ = testFeatures_.drop(['value121'], 1)
    testFeatures_ = testFeatures_.drop(['value131'], 1)

    testFeatures_ = testFeatures_.drop(['value135'], 1)
    testFeatures_ = testFeatures_.drop(['value139'], 1)
    testFeatures_ = testFeatures_.drop(['value140'], 1)
    testFeatures_ = testFeatures_.drop(['value147'], 1)
    testFeatures_ = testFeatures_.drop(['value148'], 1)
    testFeatures_ = testFeatures_.drop(['value153'], 1)
    testFeatures_ = testFeatures_.drop(['value154'], 1)

    testFeatures_ = testFeatures_.drop(['value156'], 1)
    testFeatures_ = testFeatures_.drop(['value170'], 1)
    testFeatures_ = testFeatures_.drop(['value174'], 1)
    testFeatures_ = testFeatures_.drop(['value186'], 1)
    testFeatures_ = testFeatures_.drop(['value188'], 1)
    testFeatures_ = testFeatures_.drop(['value191'], 1)
    testFeatures_ = testFeatures_.drop(['value192'], 1)

    testFeatures_ = testFeatures_.drop(['value193'], 1)
    testFeatures_ = testFeatures_.drop(['value199'], 1)
    testFeatures_ = testFeatures_.drop(['value202'], 1)
    testFeatures_ = testFeatures_.drop(['value203'], 1)
    testFeatures_ = testFeatures_.drop(['value208'], 1)
    testFeatures_ = testFeatures_.drop(['value215'], 1)
    testFeatures_ = testFeatures_.drop(['value225'], 1)

    testFeatures_ = testFeatures_.drop(['value228'], 1)
    testFeatures_ = testFeatures_.drop(['value241'], 1)
    testFeatures_ = testFeatures_.drop(['value242'], 1)
    testFeatures_ = testFeatures_.drop(['value251'], 1)
    testFeatures_ = testFeatures_.drop(['value252'], 1)
    testFeatures_ = testFeatures_.drop(['value255'], 1)
    testFeatures_ = testFeatures_.drop(['value260'], 1)

    testFeatures_ = testFeatures_.drop(['value261'], 1)
    testFeatures_ = testFeatures_.drop(['value294'], 1)
    testFeatures_ = testFeatures_.drop(['value295'], 1)
    testFeatures_ = testFeatures_.drop(['value301'], 1)
    testFeatures_ = testFeatures_.drop(['value306'], 1)
    testFeatures_ = testFeatures_.drop(['value309'], 1)
    testFeatures_ = testFeatures_.drop(['value318'], 1)

    testFeatures_ = testFeatures_.drop(['value323'], 1)
    testFeatures_ = testFeatures_.drop(['value324'], 1)
    testFeatures_ = testFeatures_.drop(['value335'], 1)
    testFeatures_ = testFeatures_.drop(['value337'], 1)
    testFeatures_ = testFeatures_.drop(['value347'], 1)
    testFeatures_ = testFeatures_.drop(['value369'], 1)
    testFeatures_ = testFeatures_.drop(['value370'], 1)

    testFeatures_ = testFeatures_.drop(['value372'], 1)
    testFeatures_ = testFeatures_.drop(['value376'], 1)
    testFeatures_ = testFeatures_.drop(['value377'], 1)
    testFeatures_ = testFeatures_.drop(['value379'], 1)
    testFeatures_ = testFeatures_.drop(['value383'], 1)
    
    testFeatures = np.ones((m, n+1))
    testFeatures[:, :-1] = testFeatures_

    print("start predict...")
    predictLable = predict(testFeatures, theta)

    print("write the result to csv and save...")
    writeCSV(predictLable)

if __name__ == "__main__":
     main()