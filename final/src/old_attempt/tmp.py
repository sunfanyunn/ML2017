import pandas as pd
import numpy as np


df = pd.read_csv('haha.csv')
df['price_doc'] = df['price_doc']*0.97
df.to_csv('haha0.97.csv', index=False)

'''
df2 = pd.read_csv('predict2015.csv')

nb = 4000

'''
df2 = df1.iloc[nb:]
df2['price_doc'] = df2['price_doc']*0.97
frames = [df1.iloc[:nb], df2]
df = pd.concat(frames)

print(df.head())

df.to_csv('rey_4000_0.95_rest_0.99.csv', index=False)
