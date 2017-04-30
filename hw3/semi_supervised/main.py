import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import keras.backend as K


from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

unlabeled_x = None


class History(Callback):
    def on_train_begin(self,logs={}):
        self.tr_losses=[]
        self.val_losses=[]
        self.tr_accs=[]
        self.val_accs=[]

    def on_epoch_end(self,epoch,logs={}):
        '''
        self.tr_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.tr_accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        '''
        self.tr_losses = [ logs.get('loss') ]
        self.val_losses = [ logs.get('val_loss') ]
        self.tr_accs = [ logs.get('acc') ]
        self.val_accs = [ logs.get('val_acc') ]
        dump_history(self)

def dump_history(logs,store_path='./'):
    with open(os.path.join(store_path,'train_loss'),'a') as f:
        for loss in logs.tr_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'train_accuracy'),'a') as f:
        for acc in logs.tr_accs:
            f.write('{}\n'.format(acc))
    with open(os.path.join(store_path,'valid_loss'),'a') as f:
        for loss in logs.val_losses:
            f.write('{}\n'.format(loss))
    with open(os.path.join(store_path,'valid_accuracy'),'a') as f:
        for acc in logs.val_accs:
            f.write('{}\n'.format(acc))

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

lamda = 0.1
def semi_loss( y_test, y_pred ):
    labeled_loss = K.categorical_crossentropy(y_test, y_pred)
    unlabeled_pred = model.predict( unlabeled_x ).flatten()
    unlabeled_loss = sum(np.log( unlabeled_pred ) * unlabeled_pred)
    return labeled_loss + lamda*unlabeled_loss


def load_data():
    number = 4000

    train_df = pd.read_csv("../train.csv")
    test_df = pd.read_csv("../ans.csv")

    test_df = train_df.iloc[0:number]
    train_df = train_df.iloc[number:]
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

    unlabeled_size = 10000
    global unlabeled_x
    unlabeled_x = x_train[:unlabeled_size]

    x_train = x_train[unlabeled_size:]
    y_train = y_train[unlabeled_size:]

    return (x_train, y_train, x_test, y_test)

def main():
    (x_train, y_train, x_test, y_test) = load_data()


    data_dim = (48, 48, 1)
    nb_class = 7
    global model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=data_dim))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    '''
    model.add(Conv2D(64, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    '''

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

    model.add(Flatten())
    model.add(Dense(2048))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))

    model.summary()

    #sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
    sgd = SGD(lr=0.005, decay=0.00001, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = History()
    #checkpointer = ModelCheckpoint(filepath="best_model", verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=30, epochs=50, \
            validation_data=(x_test, y_test), callbacks=[history])
    dump_history(history)

    score = model.evaluate(x_test, y_test)
    print ('Train Acc:', score[1] )
    model.save("model_tmp")

model = None
main()
