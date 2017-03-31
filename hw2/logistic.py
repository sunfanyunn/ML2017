import numpy as np
import pandas as pd
import sys
import time
import random
import scipy.optimize as opt
import csv
#from logistic import logisticClassifier

workclasses = [" Federal-gov", " Local-gov", " Never-worked", " Private", " Self-emp-inc", " Self-emp-not-inc", " State-gov", " Without-pay", "?_workclass"]
jobs = ['?_occupation', ' Adm-clerical', ' Armed-Forces', ' Craft-repair', ' Exec-managerial', ' Farming-fishing', ' Handlers-cleaners', ' Machine-op-inspct', ' Other-service', ' Priv-house-serv', ' Prof-specialty', ' Protective-serv', ' Sales', ' Tech-support', ' Transport-moving']
countries = ['?_native_country', ' Cambodia', ' Canada', ' China', ' Columbia', ' Cuba', ' Dominican-Republic', ' Ecuador', ' El-Salvador', ' England', ' France', ' Germany', ' Greece', ' Guatemala', ' Haiti', ' Holand-Netherlands', ' Honduras', ' Hong', ' Hungary', ' India', ' Iran', ' Ireland', ' Italy', ' Jamaica', ' Japan', ' Laos', ' Mexico', ' Nicaragua', ' Outlying-US(Guam-USVI-etc)', ' Peru', ' Philippines', ' Poland', ' Portugal', ' Puerto-Rico', ' Scotland', ' South', ' Taiwan', ' Thailand', ' Trinadad&Tobago', ' United-States', ' Vietnam', ' Yugoslavia']


rawTrainFile = sys.argv[1]
rawTestFile = sys.argv[2]
trainFile = sys.argv[3]
ansFile = sys.argv[4]
testFile = sys.argv[5]
resultFile = sys.argv[6]

def sigmoid(z):
    ret = np.minimum(0.99999999999, 1 / (1 + np.exp(-z)))
    ret = np.maximum(0.00000000001, ret)
    return ret

lamda = 0.001

def Cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log( 1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / len(X) + lamda*.5/len(X)*np.sum(
            np.square(theta[1:]))

def Gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        if i == parameters-1:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = np.sum(term) / len(X) + lamda*theta[0,i]/len(X)
    return grad

class logisticClassifier:
    theta=[]

    def train(self, X, y, iterations=100000, lr=1e-9):

        self.theta = np.matrix( np.zeros(X.shape[1]) ).T
        for _ in range(iterations):
            grad = Gradient(self.theta.T, X, y)
            self.theta = self.theta - lr*np.matrix(grad).T
            if _%100 == 0:
                lr *= 0.9999

    def doPredict(self, X):
        prob = sigmoid(X * self.theta.T)
        return [1 if pp >= .5 else 0 for pp in prob]

def writeResult():
    res = classifier.doPredict(np.matrix(testDf))
    assert( len(res) == 16281 )
    with open(resultFile, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['id', 'label'])
        for i in range(len(res)):
            writer.writerow( [str(i+1), res[i]] )

def normalize(df):
    #columns to be normalized
    return (df - df.min())/(df.max()-df.min())

xDf = pd.read_csv(trainFile)
yDf = pd.read_csv(ansFile, header=None)
testDf = pd.read_csv(testFile)
#Add bias term
xDf["b"] = np.ones(len(xDf))
testDf["b"] = np.ones(len(testDf))
#Add new features
continuous = ["age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]
for ff in continuous:
    xDf[ff+"*"] = np.log( 1+xDf[ff] )
    testDf[ff+"*"] = np.log( 1+testDf[ff] )
X = np.matrix(xDf)
Y = np.matrix(yDf)
classifier = logisticClassifier()
#classifier.train(X, Y)
# hard-code the parameter
classifier.theta = np.matrix(
[[  3.13049562e-02,  1.59115999e-06,  6.16374915e-01,  7.85782736e-04,
    1.60770490e-03,  3.71343782e-02,  5.84531568e-01, -1.76459461e-01,
   -3.25746025e-03, -6.74250905e-02,  2.08763705e-01, -5.32147965e-01,
   -2.04842102e-01, -2.75517715e-02, -3.81063034e-01, -7.25772519e-01,
   -7.44663631e-01, -2.53527104e-01, -2.50021467e-01, -4.15106946e-01,
   -8.79128044e-01, -5.53166989e-01,  9.33256115e-02,  1.43571708e-01,
    6.70941731e-01,  8.63255753e-01, -3.94085579e-01,  9.54613984e-01,
   -9.53833003e-02,  9.51273597e-01,  1.99952811e-02, -2.80791430e-01,
    7.65749159e-02,  1.13230727e+00, -1.17352622e-01, -9.01238457e-01,
   -2.74428221e-01, -2.31997979e-01,  4.07298821e-02, -8.06494168e-03,
    6.49189191e-02,  8.00785522e-01, -9.34022848e-01, -6.50227907e-01,
   -4.11223892e-01, -9.31523806e-01, -1.17799102e-01,  7.09841746e-01,
    3.96339898e-01,  3.89871540e-01,  6.85715439e-01, -2.57600435e-01,
   -3.84320054e-01,  2.33501704e-01,  4.99758464e-02, -3.88822920e-01,
   -1.03258896e+00, -4.79274597e-01,  1.11951699e+00, -2.06973318e-01,
    3.54034975e-02, -2.00044076e-01, -1.77304930e-01, -2.72398200e-02,
    3.46002233e-02,  5.94985589e-02, -5.52794963e-02, -7.75497116e-02,
    2.62013115e-02, -8.47644715e-02, -1.58213743e-02, -6.32413146e-02,
    4.87969424e-02,  4.34944729e-02,  8.09317410e-02, -5.51550392e-02,
   -3.24001737e-02, -1.10617722e-02, -1.27419013e-03, -5.17390988e-03,
    1.83286503e-03,  1.24204228e-03,  5.88377307e-03,  9.37298386e-03,
    1.54168263e-02,  6.01202522e-02, -3.76811376e-03,  4.88054532e-02,
   -1.01430553e-02, -5.40421241e-01, -3.57722942e-02, -2.19248732e-02,
   -2.30758389e-02,  1.27010355e-01, -1.45600330e-02, -3.15506631e-02,
   -7.82359082e-02,  3.83997108e-03, -8.70352165e-02,  2.92221543e-02,
   -1.01034488e-02, -1.14677641e-02,  1.79906739e-01, -7.38538548e-02,
    1.35997356e-02, -1.12461398e-01, -4.68385026e-01, -4.03769556e-01,
   -1.97041403e-01, -3.55440647e-01, -2.60788930e-01, -4.27257959e-01]] )

print(classifier.theta)
writeResult()
