import pandas as pd
import numpy as np

df = pd.read_csv('submission.csv')
factor=0.9923
df['price_doc'] = df['price_doc']*factor
df.to_csv('fuckin_test.csv', index=False)
print('mean', df['price_doc'].mean())
tmp = np.log1p(df['price_doc'])
print('sqrt( mean of log1p**2 )', np.sqrt( (tmp**2).mean()))

