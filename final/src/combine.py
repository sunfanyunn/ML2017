import os
import pandas as pd
import numpy as np
res = pd.Series(np.arange(7662))
lst = [2, 234, 34, 3451, 4]
for seed in lst:
    filename = '{}'.format(seed)
    df = pd.read_csv(os.path.join('CSVs',filename))
    res += df['price_doc']/len(lst)

    if seed == lst[-1]:
        print('damn')
        df['price_doc']= res
        df.to_csv('final.csv', index=False)

