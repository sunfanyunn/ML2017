import pandas as pd
import re
import numpy as np
import csv
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline

#import word2vec
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers import LSTM
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.regularizers import l1, l2

import tensorflow as tf
trainFile='train_data.csv'
testFile='test_data.csv'
outputFile='res.csv'
EMBEDDING_DIM=100
MAX_SEQUENCE_LENGTH=300


def f1score(y_true, y_pred):

    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def np_f1score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    num_tp = sum(y_true*y_pred)
    num_fn = sum(y_true*(1-y_pred))
    num_fp = sum((1-y_true)*y_pred)
    num_tn = sum((1-y_true)*(1-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def load_embedding():
    embeddings_index = {}
    with open('embeddings/glove.6B.100d.txt', 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def load_data():

    x_train ,y_train, x_test = [], [], []

    engine = re.compile('(.*?),"(.*?)",(.*)')
    with open(trainFile, 'r') as f:
        line = f.readline()
        for line in f:
            res = engine.findall(line)[0]
            assert len(res)==3
            y_train.append( res[1].split(',') )
            x_train.append( res[2] )

    engine = re.compile('\d*,(.*)')
    with open(testFile, 'r') as f:
        line = f.readline()
        for line in f:
            text = engine.findall(line)[0]
            x_test.append(text)

    assert(len(x_train) == len(y_train))

    x_train = np.array(x_train)
    x_test  = np.array(x_test)
    y_train = np.array(y_train)

    tokenizer = Tokenizer(num_words=1000)
    all_txt = (np.concatenate((x_train.flatten(), x_test.flatten())))
    print(all_txt.shape)
    tokenizer.fit_on_texts(all_txt)
    word_index = tokenizer.word_index
    print("len of word_index", len(word_index))

    mxlen=MAX_SEQUENCE_LENGTH
    x_train = tokenizer.texts_to_sequences( x_train )
    x_train = pad_sequences(x_train, mxlen)
    x_test = tokenizer.texts_to_sequences( x_test )
    x_test = pad_sequences(x_test, mxlen)
    print(x_train.shape)
    print(x_test.shape)

    mlb = MultiLabelBinarizer()
    mlb.fit_transform(y_train)
    y_train = mlb.transform(y_train)
    print(len(x_train), len(x_test))
    assert(len(x_train) == len(y_train))

    nb_valid = 500
    return x_train[nb_valid:], y_train[nb_valid:], x_train[:nb_valid], \
y_train[:nb_valid], x_test, word_index, mlb

def writeResult(mlb, y_pred):
    y_pred = np.array(y_pred)
    y_pred = mlb.inverse_transform(y_pred)

    with open(outputFile, 'w') as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(['id', 'tags'])
        for i in range(len(y_pred)):
            csv_writer.writerow([i, ' '.join(y_pred[i])])

def Model():
    model = Sequential()
    model.add(Conv1D(128, 5, activation='relu', input_shape=(300,100)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(256, 5, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(512, 5, activation='relu'))
    model.add(LSTM(512))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', f1score])
    return model

def main():
    x_train, y_train, x_valid, y_valid, x_test, word_index, mlb = load_data()
    nb_class = len(y_train[0])
    print("nb_class", nb_class)
    embeddings_index = load_embedding()
    print ("finish loading embedding")

    embedding_matrix = np.zeros(( len(word_index) + 1, EMBEDDING_DIM) )
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

    x_train = np.array([ [embedding_matrix[i] for i in x] for x in x_train ])
    x_test = np.array([ [embedding_matrix[i] for i in x] for x in x_test ])
    print(x_train.shape)
    print(x_test.shape)
    '''
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=100)
    multi_target_forest = MultiOutputClassifier(forest)
    pred = multi_target_forest.fit(x_train, y_train).predict(x_valid)
    print( [ sum(x) for x in pred ] )
    print( pred.shape )
    print( np_f1score(y_valid, pred) )
    '''
    from sklearn.cluster import KMeans
    model = []
    for cl in range(38):
        x_train0, x_train1 = [], []
        for i,x in enumerate(x_train):
            if y_train[i][cl] == 1:
                x_train1.append(x)
            else:
                x_train0.append(x)

        x_train0 = np.array(x_train0)
        x_train1 = np.array(x_train1)
        xx_train = np.concatenate( (x_train0[:len(x_train1)], x_train1))
        yy_train = np.array( [0]*len(x_train1) + [1]*len(x_train1) )
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2)
        x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
        print(x_train.shape)
        X_tsne = tsne.fit_transform(x_train[:3000])
        for i in range(0, 3000, 300):
            if y_train[i][cl] == 1:
                for j in range(300):
                    plt.scatter( X_tsne[i+j][0], X_tsne[i+j][1] , color=['red'])
            else:
                for j in range(300):
                    plt.scatter( X_tsne[i+j][0], X_tsne[i+j][1] , color=['blue'])
        plt.savefig("tmp")
        '''
        print("all", len(x_train), "ones", sum(x_train1))
        print( sum(x_train1)*1./len(x_train))

#        print("performing KMeans clustering")
#        kmeans = KMeans(n_clusters=len(x_train1), random_state=0).fit(x_train0)
#        x_train0 = kmeans.cluster_centers_
#        print("done clustering")
        '''


        yy_valid = []
        for i,x in enumerate(x_valid):
            if y_valid[i][cl] == 1:
                yy_valid.append(1)
            else:
                yy_valid.append(0)

        print("all", len(yy_valid), "ones", sum(yy_valid))
        print( sum(yy_valid)*1./len(yy_valid))

        tmp_model = Model(embedding_layer)
        tmp_model.fit(xx_train, yy_train, validation_data=(x_valid, yy_valid),
                epochs=5, batch_size=128)

        pred = tmp_model.predict( x_valid )
        print(sum(pred))
#        print( tmp_model.evalutate(x_valid, yy_valid) )
        print( np_f1score(yy_valid, pred) )

#        model.fit(tmp_model)
#        training = [ x_train[i]

    '''
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=50, batch_size=128)

    tmp  = model.predict(x_valid)
    pred = []
    for res in tmp:
        pred.append( [1 if x > 0.5 else 0 for x in res] )
    print("validation f1score", np_f1score(y_valid, np.array(pred)))

    y_pred = model.predict(x_test)
    pred = []
    for res in y_pred:
        pred.append( [1 if x >0.5 else 0 for x in res] )
    print(y_pred)
    print(y_pred[0])
    '''

    writeResult(mlb, pred)

if __name__=='__main__':
    main()
