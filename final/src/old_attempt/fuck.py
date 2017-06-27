import pandas as pd
import numpy as np

df = pd.read_csv('../input/reynaldo.csv')
test = pd.read_csv('../input/test.csv')
macro = pd.read_csv('../input/macro.csv')
'''
fit_feature = "rent_price_3room_eco"

macro_feature = pd.concat([test, macro], axis=1)[fit_feature]
#macro_feature.fillna(np.nan, inplace=True)

macro_feature = macro_feature.interpolate(method='akima')
macro_feature = macro_feature.interpolate(method='ffill')
macro_feature = macro_feature.interpolate(method='bfill')


assert macro_feature.isnull().any() == False

print(macro_feature.max(), macro_feature.min())
macro_feature = (macro_feature - macro_feature.min())/(macro_feature.max() - macro_feature.min())*0.04 + 0.93
print(macro_feature.max(), macro_feature.min())

assert len(df) == len(macro_feature)
'''
'''
nb = 5000
df1 = df.iloc[:nb].copy()
df2 = df.iloc[nb:].copy()
df2["price_doc"] = df2["price_doc"]*0.95
frames = [ df1, df2 ]
df = pd.concat(frames)
print(df.head())
print(df.tail())
'''

nb = 4000

df1 = df.iloc[:nb].copy()
#df2 = df.iloc[nb:2*nb].copy()

df1["price_doc"] = df1["price_doc"]*0.95
#df2["price_doc"] = df2["price_doc"]*0.99

df2 = df.iloc[nb:].copy()

frames = [ df1, df2 ]
df = pd.concat(frames)

tmp_df = pd.read_csv('CSVs/sub.csv')

df['price_doc'] = (df['price_doc'] + tmp_df['price_doc'])/2
print(df.head())

'''
#df["price_doc"] = df["price_doc"]*macro_feature
df["price_doc"] = df["price_doc"]*0.97
print(df.head())
'''

df.to_csv("ensemble.csv", index=False)

