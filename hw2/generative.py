import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import sys
import time
import csv

rawTrainFile = sys.argv[1]
rawTestFile = sys.argv[2]
trainFile = sys.argv[3]
ansFile = sys.argv[4]
testFile = sys.argv[5]
resultFile = sys.argv[6]
print(trainFile, ansFile, testFile, resultFile)

def sigmoid(z):
    return 1/(1+np.exp(-z))
'''
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)
'''

def writeResult():

    assert( len(res) == 16281 )
    with open(resultFile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['id', 'label'])
        for i in range(len(res)):
            writer.writerow( [str(i+1), res[i]] )

def normalize(df):
    #columns to be normalized
    return (df - df.min())/(df.max() - df.min())

alpha = 1.55
def doPredict(testDf):
    W = alpha*(mu1-mu0).T*inv
    b = alpha*(-.5*mu1.T*inv*mu1 + .5*mu0.T*inv*mu0) + np.log(len(x1df)/len(x0df))
    prob = sigmoid( np.matrix(testDf)*W.T+b )
    return  [1 if p >= 0.5 else 0 for p in prob]

xDf = pd.read_csv(trainFile)
yDf = pd.read_csv(ansFile, header=None)
testDf = pd.read_csv(testFile)

xDf['res'] = pd.Series(np.array(yDf).ravel())
x0df = xDf[xDf.res==0]
x1df = xDf[xDf.res==1]
x0df = x0df.drop( ["res"], axis=1)
x1df = x1df.drop( ["res"], axis=1)
xDf = xDf.drop( ['res'], axis=1)

mu0 = np.matrix(x0df.mean()).T
mu1 = np.matrix(x1df.mean()).T

cov = np.cov(np.matrix(xDf).T)
inv = np.linalg.pinv(cov)

res = doPredict(np.matrix(testDf))
writeResult()

