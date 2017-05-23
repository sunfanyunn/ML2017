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
from sklearn.multioutput import MultiOutputClassifier

from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM
#from sklearn.metrics import f1_score
import tensorflow as tf

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

def np_f1score(y_true, y_pred):
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
'''
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
'''
train_path = 'train_data.csv'
test_path = 'test_data.csv'
output_path = 'rdf.csv'

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
def f1_score(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

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

    forest = RandomForestClassifier(n_estimators=10, random_state=1)
    multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    print("fitting ...")
    multi_target_forest.fit(X_train, Y_train)

    pred_val = multi_target_forest.predict_proba(X_val)
    print(pred_val.shape)
    pred_val = (pred_val > thresh).astype('int')
    print( pred_val.shape )
#    print( np_f1score(Y_val, pred_val) )

    Y_pred = multi_target_forest.predict(test_sequences)
    Y_pred_thresh = (Y_pred > thresh).astype('int')

    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            if len(labels) == 0:
                labels.append( tag_list[ np.argmax(Y_pred[index]) ] )
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()
