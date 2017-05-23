import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import numpy as np
import string
import sys
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.layers import LSTM

try:
   import cPickle as pickle
except:
   import pickle

def np_func(y_true, y_pred, thresh):
    y_pred = (y_pred > thresh).astype('int')

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
    print(np_func(y_true, y_pred, 0.4))

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
        if cond==1:
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

'''
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
'''
#train_path = 'train_data.csv'
test_path = sys.argv[1]
output_path = sys.argv[2]

tag_list = [
        "SCIENCE-FICTION",
        "SPECULATIVE-FICTION",
        "FICTION",
        "NOVEL",
        "FANTASY",
        "CHILDREN'S-LITERATURE",
        "HUMOUR",
        "SATIRE",
        "HISTORICAL-FICTION",
        "HISTORY",
        "MYSTERY",
        "SUSPENSE",
        "ADVENTURE-NOVEL",
        "SPY-FICTION",
        "AUTOBIOGRAPHY",
        "HORROR",
        "THRILLER",
        "ROMANCE-NOVEL",
        "COMEDY",
        "NOVELLA",
        "WAR-NOVEL",
        "DYSTOPIA",
        "COMIC-NOVEL",
        "DETECTIVE-FICTION",
        "HISTORICAL-NOVEL",
        "BIOGRAPHY",
        "MEMOIR",
        "NON-FICTION",
        "CRIME-FICTION",
        "AUTOBIOGRAPHICAL-NOVEL",
        "ALTERNATE-HISTORY",
        "TECHNO-THRILLER",
        "UTOPIAN-AND-DYSTOPIAN-FICTION",
        "YOUNG-ADULT-LITERATURE",
        "SHORT-STORY",
        "GOTHIC-FICTION",
        "APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION",
        "HIGH-FANTASY"]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128
thresh=0.35


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
def f1_score(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred)

    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))

#########################
###   Main function   ###
#########################
def main():
    assert len(tag_list) == 38

    (_, X_test,_) = read_data(test_path,False)
    tokenizer = load_tokenizer( "rnn/tokenizer_pickle")
    word_index = tokenizer.word_index

    ### convert word sequences to index sequence
    test_sequences = tokenizer.texts_to_sequences(X_test)
    print(type(test_sequences))

    ### padding to equal length
    test_sequences = pad_sequences(test_sequences)
    print(test_sequences.shape)
    print ('Padding sequences.')
    max_article_length = test_sequences.shape[1]
    print(" max_article_length", max_article_length)
#    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)

    ### split data into training set and validation set
#    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

    ### get mebedding matrix from glove
    print ('Get embedding dict from glove.')
#    embedding_dict = get_embedding_dict('embeddings/glove.6B.%dd.txt'%embedding_dim)
#    save_embedding_dict(embedding_dict, 'embedding_dict')
    embedding_dict = load_embedding_dict('rnn/embedding_dict')

#    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embedding_matrix],
                        input_length=max_article_length,
                        trainable=False))
    model.add(GRU(512,activation='tanh',dropout=0.1))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    model.load_weights('rnn/best.hdf5')
    Y_pred = model.predict(test_sequences)
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
