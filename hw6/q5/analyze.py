import sys
import os
import pandas as pd
import numpy as np
# from CFModel import CFModel, DeepModel
from keras.layers import Embedding, Reshape, Input, Dot, Add, Merge
from keras.models import Model, Sequential

def create_model(n_users, m_items, k_factors):

    user_inp = Input(shape=(1,))
    user = Embedding(n_users, k_factors, input_length=1)(user_inp)
    user = Reshape((k_factors,))(user)

    movie_inp = Input(shape=(1,))
    movie = Embedding(m_items, k_factors, input_length=1)(movie_inp)
    movie = Reshape((k_factors,))(movie)


    l1 = Dot(axes=1)([user, movie])
    # l2 = Add()([l1, user_b, movie_b])

    return Model([user_inp, movie_inp], l1)



test_path='input/test.csv'
# output_path = sys.argv[2]

K_FACTORS = 120
RNG_SEED = 1446557

max_userid = 6040
max_movieid = 3952
K_FACTORS = 120

model = create_model(max_userid+1, max_movieid+1, K_FACTORS)
model.load_weights('models/cf_normalize.h5')
print(model.layers)

std, mean =1.11689766115, 3.58171208604

user_emb = np.array(model.layers[2].get_weights()).squeeze()
print('user embedding shape:', user_emb.shape)
movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie embedding shape', movie_emb.shape)
# np.save('user_emb.npy', user_emb)
# np.save('movie_emb.npy', movie_emb)

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

movie = pd.read_csv('input/movies.csv', delimiter='::')
X, Y = [], []
type1 = ['Thriller', 'Horror', 'Crime']
type2 = ['Drama', 'Musical']
for x in movie.iterrows():
    movie_id = x[1]['movieID']
    movie_category = x[1]['Genres']

    if any( t in movie_category for t in type1 ):
        if any( t in movie_category for t in type2 ):
            continue
        X.append( movie_emb[movie_id] )
        Y.append( 0 )
    elif any( t in movie_category for t in type2 ):
        X.append( movie_emb[movie_id] )
        Y.append( 1 )



tsne = TSNE(n_components=2)
vis_data = tsne.fit_transform(X)
print(type(vis_data))
print(vis_data.shape)
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

cm = plt.cm.get_cmap('RdBu')
sc = plt.scatter(vis_x, vis_y, c=Y, cmap=cm)
plt.colorbar(sc)
plt.savefig("q5.png")

