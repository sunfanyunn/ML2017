import sys
import math
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from CFModel import CFModel, DeepModel
from keras.layers import Embedding, Reshape, Input, Dot, Add
from keras.models import Model, Sequential

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

def create_model(n_users, m_items, k_factors):

    user_inp = Input(shape=(1,))
    user = Embedding(n_users, k_factors, input_length=1)(user_inp)
    user = Reshape((k_factors,))(user)

    movie_inp = Input(shape=(1,))
    movie = Embedding(m_items, k_factors, input_length=1)(movie_inp)
    movie = Reshape((k_factors,))(movie)

    user_b = Embedding(n_users, 1)(user_inp)
    user_b = Reshape((1,))(user_b)
    movie_b = Embedding(m_items, 1)(movie_inp)
    movie_b = Reshape((1,))(movie_b)

    l1 = Dot(axes=1)([user, movie])
    l2 = Add()([l1, user_b, movie_b])

    return Model([user_inp, movie_inp], l2)


train_file = 'input/train.csv'
K_FACTORS = 120
RNG_SEED = 1446557


#cell 5
ratings = pd.read_csv(train_file)
print(ratings.head())
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print( len(ratings), 'ratings loaded.')

shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
Users = shuffled_ratings['UserID'].values
Movies = shuffled_ratings['MovieID'].values
Ratings = shuffled_ratings['Rating'].values

std, mean = Ratings.std(), Ratings.mean()
Ratings = (Ratings-mean)/std
print(Users.shape, Movies.shape, Ratings.shape)

print("max_userid", max_userid)
print("max_movieid", max_movieid)

# model = create_model(max_userid + 1, max_movieid + 1, K_FACTORS)
model = CFModel(max_userid+1, max_movieid+1, K_FACTORS)
model.compile(loss='mse', optimizer='adamax')

callbacks = [EarlyStopping('val_loss', patience=2),
             ModelCheckpoint('deep_normalize.h5', save_best_only=True)]
history = model.fit([Users, Movies], Ratings, nb_epoch=30, validation_split=.1, verbose=2, callbacks=callbacks)
'''
loss = pd.DataFrame({'epoch': [ i + 1 for i in history.epoch ],
                     'training': [ math.sqrt(loss) for loss in history.history['loss'] ],
                     'validation': [ math.sqrt(loss) for loss in history.history['val_loss'] ]})
ax = loss.ix[:,:].plot(x='epoch', figsize={7,10}, grid=True)
ax.set_ylabel("root mean squared error")
ax.set_ylim([0.0,3.0]);
ax.savefig("k_{}".format(K_FACTORS))
'''

#cell 14
## Print best validation RMSE
min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
