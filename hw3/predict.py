import csv
import sys
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.regularizers import l1, l2
#from keras.datasets import mnist
#categorical_crossentropy

def load_data():
    test_df = pd.read_csv("test.csv")
    x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
    #y_test = np.array( test_df["label"] )
    x_test/=255
    #y_test = np_utils.to_categorical(y_test, 7)
    return x_test#, y_test

def main():
    x_test = load_data()
    x_test = x_test.reshape( x_test.shape[0], 48, 48, 1)

    model = load_model('model_tmp')
    prd_class = model.predict_classes(x_test)

    with open('res.csv', 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id', 'label'])
        for i in range(len(x_test)):
            csv_writer.writerow([i]+[prd_class[i]])

main()
