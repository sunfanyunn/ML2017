'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.00001
set_session(tf.Session(config=config))

import pandas as pd
from scipy.misc import imsave
import numpy as np
import time
from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt


testFile="../test.csv"
def load_data():
    test_df = pd.read_csv(testFile)
    x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
    #y_test = np.array( test_df["label"] )
    #x_test/=255
    #y_test = np_utils.to_categorical(y_test, 7)
    return x_test#, y_test

x_test = load_data()
input_img_data = np.array(x_test[7])

# dimensions of the generated pictures for each filter.
img_width = 48
img_height = 48

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# build the VGG16 network with ImageNet weights
#model = vgg16.VGG16(weights='imagenet', include_top=False)
model = load_model('../model_tmp')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

input_img_data = input_img_data.reshape(1, 48, 48, 1)
pred = model.predict_classes(input_img_data)
target = K.mean(model.output[:, pred])

grads = K.gradients(target, input_img)[0]
fn = K.function([input_img, K.learning_phase()], [grads])

gradients = fn( [input_img_data,0] )[0]
gradients = np.array(gradients)
print(gradients)
print(gradients.max())
print(gradients.min())
gradients = (gradients-gradients.min())/(gradients.max()-gradients.min())
gradients = np.array(gradients).reshape(48,48)
print(gradients)
thres = 0.58

heatmap = gradients

see = input_img_data.reshape(48, 48)
see[np.where(heatmap <= thres)] = np.mean(see)

plt.figure()
plt.imshow(heatmap, cmap=plt.cm.jet)
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig("heatmap")

plt.figure()
plt.imshow(see,cmap='gray')
plt.colorbar()
plt.tight_layout()
fig = plt.gcf()
plt.draw()
fig.savefig("grey")
