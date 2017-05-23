import numpy as np
import string
import sys
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout, Activation
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU

from keras.layers import LSTM
#from sklearn.metrics import f1_score
import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

try:
   import cPickle as pickle
except:
   import pickle

def load_tokenizer(tokenizer_path):
  return pickle.load(open(tokenizer_path, "rb"))

def save_tokenizer(tokenizer, tokenizer_path):
    pickle.dump(tokenizer, open(tokenizer_path, "wb"), pickle.HIGHEST_PROTOCOL)

def save_embedding_dict(dic, path):
    with open(path, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_embedding_dict(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def np_func(y_true, y_pred):
    ret = []
    for true, pred in zip(y_true, y_pred):


        tp = sum(true*pred)
        fn = sum(true*(1-pred))
        fp = sum((1-true)*pred)
        tn = sum((1-true)*(1-pred))
        if tp==0: f1=0
        else:
            p = tp*1./(tp+fp)
            r = tp*1./(tp+fn)
            f1 = 2.*p*r/(p+r)

        ret.append(f1)

    return np.mean(ret)

def np_f1score(y_true, y_pred):

    global thresh
    precision = 0.001
    (left, right) = (0.1, 0.9)

    while True:
        if right - left <= precision:
            thresh = (left+right)/2
            break

        left_third = ((2 * left) + right) / 3
        right_third = (left + (2 * right)) / 3
        cond = (np_func(y_true, y_pred, left_third) < np_func(y_true, y_pred, right_third))
        if sess.run(cond):
            (left, right) = (left_third, right)
        else:
            (left, right) = (left, right_third)

    print(thresh)

    return np_func(y_true, y_pred, thresh)

'''
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
'''
train_path = 'train_data.csv'
test_path = 'test_data.csv'
output_path = 'bog.csv'

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128
thresh = 0.4


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]

                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            articles.append(article)

        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list):
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_data = X[indices]
    Y_data = Y[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################


def maximize(func, a, b, precision):
    """
    Find the point in the given range which maximises the input function with an
    error at most equal to `precision` / 2.
    The fuction to be maximized should either strictly increasing and then
    strictly decreasing or vice versa.
    >>> def x(x):
    ...   return x
    ...
    >>> print("{0:.0f}".format(maximize(x, 0, 10, 1e-3)))
    10
    >>> print("{0:.0f}".format(maximize(abs, -2, 20, 1e-3)))
    20
    """
    (left, right) = (a, b)
    while True:
        if right - left <= precision:
            return (left + right) / 2

        left_third = ((2 * left) + right) / 3
        right_third = (left + (2 * right)) / 3

        if func(left_third) < func(right_third):
            (left, right) = (left_third, right)
        else:
            (left, right) = (left, right_third)

def func(y_true, y_pred, thresh):

    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    ret = K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
    return ret

def f1_score(y_true,y_pred):
    global thresh
    return func(y_true, y_pred, thresh)
    '''
    precision = 0.001
    (left, right) = (0.1, 0.9)

    while True:
        if right - left <= precision:
            thresh = (left+right)/2
            break

        left_third = ((2 * left) + right) / 3
        right_third = (left + (2 * right)) / 3
        cond = tf.less(func(y_true, y_pred, left_third) , func(y_true, y_pred, right_third))
        if sess.run(cond):
            (left, right) = (left_third, right)
        else:
            (left, right) = (left, right_third)

    print(thresh)
    return func(y_true, y_pred, thresh)

    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
    '''

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))

    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    save_tokenizer(tokenizer, "bog_tokenizer_pickle")
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
#    train_sequences = tokenizer.texts_to_sequences(X_data)
    train_sequences = tokenizer.texts_to_matrix(X_data, "count")
    test_sequences = tokenizer.texts_to_matrix(X_test, "count")
    print(test_sequences.shape)

    ###
    train_tag = to_multi_categorical(Y_data,tag_list)

    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

    print ('Building model.')
    model = Sequential()
#    model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_article_length, trainable=False))
#    model.add(GRU(1024,activation='tanh',dropout=0.1))

    model.add(Dense(512,input_shape=(len(train_sequences[0]),) ))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    '''

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
#    model.add(LeakyReLU(alpha=0.3))

    model.add(Dense(512))
    model.add(Activation('relu'))
#    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.1))
    '''

    model.add(Dense(256))
#    model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128))
#    model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64))
#    model.add(LeakyReLU(alpha=0.3))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])

    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath='bog.hdf5',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_f1_score',
                                 mode='max')
    hist = model.fit(X_train, Y_train,
                     validation_data=(X_val, Y_val),
                     epochs=nb_epoch,
                     batch_size=batch_size,
                     callbacks=[earlystopping,checkpoint])


    model.load_weights('bog.hdf5')

    Y_pred = model.predict(X_train)
    Y_pred = (Y_pred > thresh).astype('int')
    print( np_f1score(Y_train, Y_pred) )
    V_pred = model.predict(X_val)
    V_pred = (V_pred > thresh).astype('int')
    print( np_f1score(Y_val, V_pred))

    Y_pred = model.predict(test_sequences)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    print(Y_pred)

    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
#        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            if len(labels) == 0:
                labels.append( tag_list[ np.argmax(Y_pred[index]) ] )
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
