import sys
import os
import pandas as pd
import numpy as np
from CFModel import CFModel, DeepModel
from keras.layers import Embedding, Reshape, Input, Dot, Add
from keras.models import Model, Sequential

test_path=os.path.join(sys.argv[1], 'test.csv')
output_path = sys.argv[2]

K_FACTORS = 120
RNG_SEED = 1446557

max_userid = 6040
max_movieid = 3952
K_FACTORS = 120

MODEL_WEIGHTS_FILE = 'models/deep.h5'
trained_model = DeepModel(max_userid+1, max_movieid+1, K_FACTORS)
trained_model.load_weights(MODEL_WEIGHTS_FILE)

std, mean =1.11689766115, 3.58171208604

test_df = pd.read_csv(test_path)

print(test_df.head())
# test_df['Rating'] = test_df.apply( lambda x: predict_rating( x['UserID'], x['MovieID'])[0], axis=1 )
test_df['Rating'] = trained_model.predict( [np.array(test_df['UserID']), np.array(test_df['MovieID']) ])
# test_df['Rating'] = test_df['Rating']*std + mean
test_df['Rating'] = np.clip(test_df['Rating'], 1, 5)
test_df = test_df.drop( ['UserID', 'MovieID'] ,axis=1 )
print(test_df.head())
test_df.to_csv(output_path, index=False)
