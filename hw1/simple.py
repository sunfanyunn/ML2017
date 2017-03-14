# coding: utf-8
#import pandas as pd
from collections import Counter
import time
import copy
import math
import sys
import csv
import time
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''
trainFile=sys.argv[1]
testFile=sys.argv[2]
resFile=sys.argv[3]
'''
'''
train=pd.read_csv('train.csv', encoding='latin1')
train=train.drop(train.columns[[0,1]], axis=1)
train=train.rename(columns={train.columns[0]:'type'})
train=train[train.type=='PM2.5']
train
'''
startTime=time.time()
features=['PM2.5', 'PM10', 'O3', 'CH4', 'CO', 'NMHC', 'AMB_TEMP', 'NO', 'NO2',
        'NOx', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR',
        'RAINFALL']
dic = {'AMB_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5, 'NOx':6,
       'O3':7, 'PM10':8, 'PM2.5':9, 'RAINFALL':10, 'RH':11, 'SO2':12, 'THC':13,
       'WD_HR':14, 'WIND_DIREC':15, 'WIND_SPEED':16, 'WS_HR':17}
assert len(features) == 18
for ff in features:
    dic[ ff+"**2" ] = dic[ff]+18
assert len(dic)==36



#id of the chosen parameter
para=[]
#How many days of the corresponding parameter is used
paraCnt=[]
#allData before processed
allData=[[] for i in range(len(dic))]
#processed data used for training
trainData=[]
dataAns=[]
validateData=[]
validateDataAns=[]

normalizedTrainData=[]
#Final Answer
res=[]
#For normalization
mean=[]
deviation=[]

def f(X):
    global res
    ret=0.0
    assert len(res)==len(trainData[0])
    resLen=len(trainData[0])
    for i in range(resLen):
        ret += res[i]*float(X[i])
    return ret

def error():
    dataSize=len(trainData)
    tmp=np.array( [f(trainData[i]) for i in range(dataSize)] )
    tmp=tmp-dataAns
    return np.sqrt( sum(np.square(tmp))/dataSize )
def validateError():
    dataSize=len(validateData)
    tmp=np.array( [f(validateData[i]) for i in range(dataSize)] )
    tmp=tmp-validateDataAns
    return np.sqrt( sum(np.square(tmp))/dataSize )

def adagrad(totalMonth=12, iteration=3000, stepSize=1e-2, fudgeFactor=1e-6, lam=500):
    global res, startTime, trainData, dataAns
    cnt=sum(paraCnt)
    res=np.array([0.0 for i in range(cnt)])
    G=np.array([0.0 for i in range(cnt)])

    print("start training")
    #bG=0
    dataSize = len(trainData)

    assert dataSize%totalMonth==0
    dataSize = int(dataSize/totalMonth)

    for it in range(iteration):
        if time.time()-startTime > 598:
            return
        for mon in range(totalMonth):
            miniBatch = trainData[mon*dataSize:(mon+1)*dataSize,: ]
            miniBatchAns = dataAns[mon*dataSize:(mon+1)*dataSize]
            tmpAns=miniBatch.dot(res)
            assert len(tmpAns) == len(miniBatch)
            grad=np.array( [-2*sum( (miniBatch[:,i])*(miniBatchAns-tmpAns) ) for i in range(cnt)] )
            #bGrad=-2*sum(dataAns-tmpAns)
            #bG = bG + np.square(bGrad)
            G = G + np.square(grad)
            res=res-stepSize*(grad/np.sqrt(G))
            #b=b-stepSize*(bGrad/np.sqrt(bG))

        if it%100 ==0:
            print(error())
            print (res)

    print("end training")
    return

def linalg():
    global res
    res = ( ( np.linalg.inv (np.dot( np.transpose(trainData), trainData ) )
        ).dot( np.transpose(trainData) ) ).dot( dataAns )
    #print (res)
    #print ("Error on train data: %f" %(error()))
    #print ("Error on validation data: %f" %(validateError()))

def writeResult():
    global mean, deviation, res
    arrx=[[] for i in range(len(dic))]
    with open('test_X.csv', 'r', encoding = 'big5') as csvfile:
        csv_f = csv.reader(csvfile)
        for row in csv_f:
            if row[1] == 'RAINFALL':
                for i in range(2, len(row)):
                    if row[i] == 'NR':
                        arrx[ dic[row[1]] ].append(0.0)
                    else:
                        arrx[ dic[row[1]] ].append(float(row[i]))
            else:
                for i in range(2, len(row)):
                    arrx[ dic[row[1]] ].append(float(row[i]))
    for i in range(9*240):
        for ff in features:
            arrx[ dic[ff+'**2'] ].append( arrx[ dic[ff] ][i]**2 )

    assert len(arrx)==36
    for i in range(len(dic)):
        assert len(arrx[i])==9*240

    with open('res.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'value'])
        featureSize=sum(paraCnt)
        for i in range(240):
            endIndex=9*(i+1)
            X=[]
            for j in range(len(para)):
                X += arrx[para[j]][endIndex-paraCnt[j]:endIndex]
            assert len(X)==featureSize
            #Normalize
            '''
            for j in range(featureSize):
                X[j] = (X[j]-mean[j])/deviation[j]
            '''
            assert len(X) == featureSize
            writer.writerow( ('id_'+str(i), f( X ) ) )

def getData():
    f = open("train.csv", "r", encoding = 'big5')
    csv_f = csv.reader(f)
    next(csv_f)
    for row in csv_f:
        if row[2] == 'RAINFALL':
            for i in range(3, len(row)):
                if(row[i] == 'NR'):
                    allData[ dic[ row[2] ] ].append(0.0)
                else:
                    allData[ dic[ row[2] ] ].append(float(row[i]))
        else:
            for i in range(3, len(row)):
                allData[ dic[ row[2] ] ].append(float(row[i]))
    f.close()
    for i in range(480*12):
        for ff in features:
            allData[ dic[ff+'**2'] ].append( allData[ dic[ff] ][i]**2 )
    featureSize = len(dic)
    for i in range(featureSize):
        assert( len(allData[i]) == 480*12 )

def addData(term, days):
    para.append(dic[term])
    #para.append(int(term))
    paraCnt.append(days)

def processData():
    global trainData, dataAns, mean, deviation

    mxCnt = max(paraCnt)
    for mon in range(12):
        for eIndex in range(480*mon+mxCnt, 480*(mon+1)):
            tmp=[]
            for i in range(len(para)):
                tmp += allData[para[i]][eIndex-paraCnt[i]:eIndex]
            assert len(tmp)==sum(paraCnt)
            trainData.append(tmp)
            dataAns.append(allData[ dic['PM2.5'] ][eIndex])

    assert len(trainData)==len(dataAns)

    trainData=np.array(trainData)
    dataAns=np.array(dataAns)

    #Normalize
    '''
    featureSize=sum(paraCnt)
    for i in range(featureSize):
        mn=sum(trainData[:,i:i+1])/len(trainData)
        dev=trainData[:,i:i+1].std()
        mean.append(mn)
        deviation.append(dev)
        trainData[:,i:i+1] = (trainData[:,i:i+1]-mn)/dev
    assert abs( trainData[:,i:i+1].std() - 1) < 1e-8
    assert abs( sum(trainData[:,i:i+1])/len(trainData) ) < 1e-8
    '''

def calc():
    arr=[]
    for key, item in dic.items():
        arr.append( (stats.linregress(allData[item], allData[9])[2]**2, key) )

    arr.sort()
    for item in arr:
        print (item[0], item[1])

def crossValidation():
    global trainData, dataAns, validateData, validateDataAns
    print(len(trainData))
    dataSize=len(trainData)
    assert dataSize%12==0
    assert dataSize!=0
    monDataSize=int(dataSize/12)
    tmpTrainData=copy.deepcopy(trainData)
    tmpDataAns=copy.deepcopy(dataAns)
    validationError=[]
    for i in range(12):
        assert id(tmpTrainData)!=id(trainData)
        assert id(tmpDataAns)!=id(dataAns)

        validateData = trainData[monDataSize*i:monDataSize*(i+1)]
        validateDataAns = dataAns[monDataSize*i:monDataSize*(i+1)]

        trainData = np.delete(trainData, [i for i in range(monDataSize*i,monDataSize*(i+1))], 0)
        dataAns = np.delete(dataAns, [i for i in range(monDataSize*i,monDataSize*(i+1))], 0)

        assert( len(trainData)*12/11 == len(tmpTrainData) )

        linalg()
        #adagrad(11)
        validationError.append(validateError())

        trainData=copy.deepcopy(tmpTrainData)
        dataAns=copy.deepcopy(tmpDataAns)
    #print("Cross Validation: %f" %(sum(validationError)/12) )
    return (max(validationError), sum(validationError)/12)

getData()
addData('PM2.5', 9)
addData('PM2.5**2', 9)
addData('PM10', 9)
addData('PM10**2', 9)
addData('O3', 3)
addData('RAINFALL', 2)
addData('NOx', 1)
addData('NO2**2', 1)
addData('NO2', 3)
addData('SO2', 1)

processData()
#print(crossValidation())
#linalg()
#print(error())
#print(res)
adagrad()
print(error())
writeResult()
'''
res =[
 -0.0245953363197,
 0.00543147729409,
 0.107525100336,
 -0.136098380784,
 0.0197225252391,
 0.14834208727,
 -0.185823748118,
 0.137097587823,
 0.677825576718,

 0.00299175488044,
 -0.0040260831014,
 -0.000836853157474,
 0.0032621548918,

-0.0188549323735,
0.00135283847849,
0.00635244063715,
-0.0222112332366,
0.08227532717,

 -4.92491489646e-05,

 0.231419896634,

 -0.0443457852013,
 -0.00190880180795,
 0.0851226615262,

 -0.0585615659587,
 -0.089182982622,

 -0.405631318112
 ]
 '''

