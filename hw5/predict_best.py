import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from keras.models import Sequential, load_model
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
thresh=0.9

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

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

    tokenizer = load_tokenizer( "best/tokenizer_pickle")
    word_index = tokenizer.word_index
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences = pad_sequences(test_sequences)
    max_article_length = test_sequences.shape[1]
    print(" max_article_length", max_article_length)
    embedding_dict = load_embedding_dict('best/embedding_dict')
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
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(38,activation='sigmoid'))
    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1_score])

    model.load_weights('best/best.hdf5')
    #model = load_model('best/fuckin_model')

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
