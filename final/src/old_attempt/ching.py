import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import datetime

def rmsle(y_true, y_pred):
    y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(y_pred_log - y_true_log), axis = -1))


def kinetic_energy(column_df):
    probs=np.unique(column_df,return_counts=True)[1]/column_df.shape[0]
    kinetic=np.sum(probs**2)
    return kinetic

def replace_nan(column_df):
    #droping   rows from column df in order to compute  kinetic  for  existing  categories
    feature_without_nans=column_df.dropna(axis=0,inplace=False)
    kinetic_replace_value=kinetic_energy(feature_without_nans)
    column_df=column_df.fillna(value=kinetic_replace_value)
    return column_df

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
id_train =  train.id
id_test = test.id

train = train.drop(train[(train.product_type == 'Investment') & (train.price_doc <= 1e6)].index)

y_train = train["price_doc"]
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

x_all = pd.concat( [x_train, x_test] )
x_all = pd.get_dummies(x_all)


for col in x_all.columns :
    if x_all[col].isnull().values.any()==True:
        x_all[col]=replace_nan(x_all[col])

x_train = x_all.iloc[:x_train.shape[0]]
x_test = x_all.iloc[x_train.shape[0]:]
assert x_train.shape[0] + x_test.shape[0] == x_all.shape[0]

# for col in x_train.columns :
# 	if x_train[col].isnull().values.any()==True:
# 		x_train[col]=replace_nan(x_train[col])

# for col in x_test.columns :
# 	if x_test[col].isnull().values.any()==True:
# 		x_test[col]=replace_nan(x_test[col])

for c in x_train.columns:
    assert x_train[c].dtype != 'object'
    assert x_test[c].dtype != 'object'

import sys
import os
from sklearn.cluster import KMeans
import pickle

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values

Cluster_cnt = int(sys.argv[1])
kmeans = KMeans(n_clusters=Cluster_cnt, random_state=0).fit(x_train)
data_x = []
data_y = []
for i in range(Cluster_cnt):
    data_x.append([])
    data_y.append([])
for i in range(x_train.shape[0]):
    m = kmeans.predict(x_train[i].reshape(1, -1))[0]
    data_x[ m ].append(x_train[i])
    data_y[ m ].append(y_train[i])
    print('\r%d/%d'%(i, x_train.shape[0]), end='')
print('')
os.system('mkdir kmeans')
for i in range(Cluster_cnt):
    pickle.dump(data_x[i], open('kmeans/data_%d_x'%(i), 'wb'))
    pickle.dump(data_y[i], open('kmeans/data_%d_y'%(i), 'wb'))

idx = np.random.permutation(x_train.shape[0])
x_train = x_train[idx]
y_train = y_train[idx]





import keras
import keras.backend as K
import h5py
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Highway
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

model = Sequential()

model.add(Dense(256, activation='relu', input_dim=x_train.shape[1]))
# model.add(Dropout(.1))

model.add(Dense(256, activation='relu'))
# model.add(Dropout(.1))

model.add(Dense(256, activation='relu'))
# model.add(Dropout(.1))

model.add(Dense(1))

model.summary()
model.compile(loss=rmsle, optimizer='adamax')


Cluster_cnt = int(sys.argv[1])
for i in range(Cluster_cnt):
    x_train = pickle.load(open('kmeans/data_%d_x'%(i), 'rb'))
    y_train = pickle.load(open('kmeans/data_%d_y'%(i), 'rb'))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    earlystopping = EarlyStopping(monitor='val_loss', patience = 10)
    checkpoint = ModelCheckpoint(filepath='best_%d.h5'%(i), save_best_only=True, monitor='val_loss')
    model.fit(x_train, y_train, batch_size=256, validation_split=.1, epochs=100, callbacks=[earlystopping, checkpoint], verbose=True)
data = []
pred = []
for i in range(Cluster_cnt): data.append([])
for i in range(x_test.shape[0]):
    m = kmeans.predict(x_test[i].reshape(1, -1))[0]
    data[ m ].append(x_test[i])
for i in range(Cluster_cnt):
    x_test = np.array(data[i])
    model = load_model('best_%d.h5'%(i), custom_objects={'rmsle': rmsle})
    pred.extend(model.predict(x_test).flatten())
output = pd.DataFrame({'id': id_test, 'price_doc': pred})
output.to_csv('out.csv', index=False)
#exit()
#model = load_model('best.h5', custom_objects={'rmsle': rmsle})
#y_pred = model.predict(x_test).flatten()

#output = pd.DataFrame({'id': id_test, 'price_doc': y_pred})
#output.head()

#output.to_csv('out.csv', index=False)
