import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.regularizers import l1, l2

def expand(x_train, y_train):

    fx_train = np.empty((x_train.shape[0]*12, 48, 48, 1))
    fy_train = np.empty(len(y_train)*12)

    for ind in range(0, len(x_train)):
        tmp = x_train[ind]
        i = 12*ind
        fx_train[i+0] = np.pad(tmp[0:42,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+1] = np.pad(tmp[6:48,0:42,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+2] = np.pad(tmp[6:48,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+3] = np.pad(tmp[0:42,6:48,:], ((3,3),(3,3),(0,0)), 'constant')
        fx_train[i+4] = np.pad(tmp[3:45,3:45,:], ((3,3),(3,3),(0,0)), 'constant')
        for j in range(5):
            fx_train[i+5+j] = np.fliplr(fx_train[i+j])
        fx_train[i+10] = tmp
        fx_train[i+11] = np.fliplr(tmp)
        fy_train[i:i+12] = y_train[ind]

    return fx_train, fy_train

def load_data():
    number = 4000

    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("ans.csv")

    #test_df = train_df.iloc[0:number]
    #train_df = train_df.iloc[number:]
    x_train = np.array( [ list(map(float,train_df["feature"].iloc[i].split())) for i in range(len(train_df)) ] )
    x_test = np.array( [ list(map(float, test_df["feature"].iloc[i].split())) for i in range(len(test_df)) ] )
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train.reshape( x_train.shape[0], 48, 48, 1)
    x_test = x_test.reshape( x_test.shape[0], 48, 48, 1)
    x_train/=255
    x_test/=255


    y_train = np.array( train_df["label"] )
    y_test = np.array( test_df["label"] )
    x_train, y_train = expand(x_train, y_train)


    y_train = np_utils.to_categorical(y_train, 7)
    y_test = np_utils.to_categorical(y_test, 7)

    assert( len(x_train) == len(y_train))
    print( len(x_train), len(y_train))
    print( len(x_test), len(y_test))
    return (x_train, y_train, x_test, y_test)

def main():
    (x_train, y_train, x_test, y_test) = load_data()

    data_dim = (48, 48, 1)
    nb_class = 7

    model = Sequential()
    '''
    model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=data_dim))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))


    model.add(Conv2D(128, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    '''

    model.add(Flatten(input_shape=data_dim))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    model.add(Dense(nb_class))

    model.add(Activation('softmax'))

    model.summary()

    #sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
    sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    model.fit(x_train, y_train, batch_size=50, epochs=150, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)
    print ('Train Acc:', score[1] )
    model.save("dnn_model")

main()
