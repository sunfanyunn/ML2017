
import pandas as pd
import numpy as np
df = pd.read_csv('final.csv')

df_lst = []
sz = int(df.shape[0]/3)
print(sz)

res = [-0.00726728669792, -0.0326138809059, np.log(0.99)/2 , 0.0236381324083]
print(res)

df['price_doc'] = np.log1p(df['price_doc'])

for i in range(4):

    if i == 0:
        tmp = df.iloc[:1000]
    elif i == 1:
        tmp = df.iloc[1000:sz]
    elif i == 2:
        tmp = df.iloc[sz:2*sz]
    elif i == 3:
        tmp = df.iloc[2*sz:3*sz]

    tmp['price_doc'] += res[i]
    df_lst.append( tmp )

#df = pd.concat(df_lst)
df['price_doc'] = np.exp(df['price_doc'])-1
df.to_csv('adjust.csv', index=False)
