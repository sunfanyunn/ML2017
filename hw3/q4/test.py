'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function


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


testFile="test.csv"
def load_data():
    test_df = pd.read_csv(testFile)
    x_test = np.array( [ list(map(float, test_df["feature"][i].split())) for i in range(len(test_df)) ] )
    #y_test = np.array( test_df["label"] )
    #x_test/=255
    #y_test = np_utils.to_categorical(y_test, 7)
    return x_test#, y_test

x_test = load_data()
input_img_data = x_test[55]
'''
imsave('original.png', input_img_data.reshape(48, 48))
plt.imshow(input_img_data.reshape(48, 48))
plt.savefig("original2.png")
'''


# dimensions of the generated pictures for each filter.
img_width = 48
img_height = 48

# the name of the layer we want to visualize
layer_name = 'activation_1'

# util function to convert a tensor into a valid image


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
model = load_model('model_tmp')
print('Model loaded.')

model.summary()

# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_outputs = []
for filter_index in range(64):
    # we only scan through the first 200 filters,
    # but there are actually 512 of them
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    if K.image_data_format() == 'channels_first':
        layer_output = (layer_output[:, filter_index, :, :])
    else:
        layer_output = (layer_output[:, :, :, filter_index])

    func = K.function([input_img], [layer_output])
    tmp = func( [input_img_data.reshape(1, 48, 48, 1)] )
    kept_outputs.append( deprocess_image(np.array(tmp).reshape(48, 48, 1)) )
    '''
    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    if K.image_data_format() == 'channels_first':
        input_img_data = input_img_data.reshape(1, 1, img_width, img_height)
    else:
        input_img_data = input_img_data.reshape(1, img_width, img_height, 1)
    # we run gradient ascent for 20 steps
    for i in range(100):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_outputs.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
    '''

# we will stich the best 64 filters on a 8 x 8 grid.
n = 7
# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
#kept_outputs.sort(key=lambda x: x[1], reverse=True)

print(len(kept_outputs))
kept_outputs = kept_outputs[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 5
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 1))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img = kept_outputs[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

# save the result to disk
imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters.reshape(width, \
    height))

