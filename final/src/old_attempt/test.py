import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
import math
from sklearn.model_selection import train_test_split
np.random.seed(7122)

def calc(df):
    tmp = np.log1p(df['price_doc'])
    print('sqrt( mean of log1p**2 )', np.sqrt( (tmp**2).mean()))

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

# Based  on  : https://alexandrudaia.quora.com/Kinetic-things-by-Daia-Alexandru
def kinetic_energy(column_df):
    probs = (np.unique(column_df, return_counts=True)[1]).astype(float)/column_df.shape[0]
    kinetic=np.sum(probs**2)
    return kinetic

def replace_nan(column_df):
    #droping   rows from column df in order to compute  kinetic  for  existing  categories
    feature_without_nans=column_df.dropna(axis=0,inplace=False)
    kinetic_replace_value=kinetic_energy(feature_without_nans)
    column_df=column_df.fillna(value=kinetic_replace_value)
    return column_df

def nan2kinetic(train, test):
    for col  in   train.columns :
        if train[col].isnull().values.any()==True:
            train[col]=replace_nan(train[col])
    train.head()
    for col  in   test.columns :
        if test[col].isnull().values.any()==True:
            test[col]=replace_nan(test[col])
    test.head()

    print(" sentences: train is  containing nans is :",train.isnull().values.any())
    print("sentence :test is  containing nans is :",test.isnull().values.any())
    return train, test

def adjust_distribution(train, strategy = 1):
    if strategy == 1:
        train = train.drop(train[(train.price_doc <= 1e6) & (train.product_type == 'Investment' )].index)
    else:
        trainsub = train[train.timestamp < '2015-01-01']
        trainsub = trainsub[trainsub.product_type=="Investment"]

        train_index = set(train.index.copy())
        ind_1m = trainsub[trainsub.price_doc <= 1000000].index
        ind_2m = trainsub[trainsub.price_doc == 2000000].index
        ind_3m = trainsub[trainsub.price_doc == 3000000].index

        for ind, gap in zip([ind_1m, ind_2m, ind_3m], [10, 3, 2]):
            ind_set = set(ind)
            ind_set_cut = ind.difference(set(ind[::gap]))
            train_index = train_index.difference(ind_set_cut)


        train = train.loc[train_index]

    return train

def myadjust(x_train, x_test):
    useless = [ "ID_railroad_station_walk", "ID_big_road1",
            "ID_railroad_terminal", "ID_metro", "ID_railroad_station_avto",
            "ID_big_road2", "ID_bus_terminal" ]
    x_train = x_train.drop( useless, axis=1 )
    x_test = x_test.drop( useless, axis=1 )

    pp = [ "full_sq", "full_all"]
    for f in pp:
#        x_train[ f + "**2" ] = x_train[f]*x_train[f]
#        x_test[ f + "**2" ] = x_test[f]*x_test[f]
        x_train[ f + 'log' ] = np.log(1+x_train[f])
        x_test[ f + 'log' ] = np.log(1+x_test[f])


    return train, test

def add_macro_feature(train, test, macro):
    print(train.shape)
    tmp = pd.concat([train, macro], axis=1)
    train["usdrub"] = tmp["usdrub"]
    print(train.shape)
    tmp = pd.concat([test, macro], axis=1)
    print(test.shape)
    test["usdrub"] = tmp["usdrub"]
    print(test.shape)

    return train, test

train_path = '../input/train.csv'
test_path = '../input/test.csv'
macro = '../input/macro.csv'

######
# parameters
#####
nb_valid = 3000
random_split = True


## scale down factor
## factor = 0.95


def main():


    train = pd.read_csv('../input/train_clean.csv')
   # train.dropna(axis=1,inplace=True)
    print(train.shape)
    test = pd.read_csv('../input/test_clean.csv')
   # test.dropna(axis=1,inplace=True)
    print(test.shape)
    macro = pd.read_csv('../input/macro.csv')
    id_test = test.id

    # Add macro feature
    # train, test = add_macro_feature(train, test, macro)


    # Use data closer to 2016 ?
    # train = train[-5000:]


    #train = train.iloc[np.random.permutation(len(train))]
    print("before dropping", len(train))
    train = adjust_distribution(train)
    print("after dropping", len(train))

    #train, test = nan2kinetic(train, test)

    y_train = train["price_doc"]
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    '''
    x_all = pd.concat( [x_train, x_test] )
    x_all = pd.get_dummies(x_all)
    x_train = x_all.iloc[:x_train.shape[0]]
    x_test = x_all.iloc[x_train.shape[0]:]
    assert x_train.shape[0] + x_test.shape[0] == x_all.shape[0]
    '''


    #can't merge train with test because the kernel run for very long time
    from scipy import stats
    for c in x_train.columns:
        if x_train[c].dtype == 'object':
#            assert False
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values)+list(x_test[c].values))
            x_train[c] = lbl.transform(list(x_train[c].values))
            x_test[c] = lbl.transform(list(x_test[c].values))

        x = np.arange(len(x_train[c]))
        y = x_train[c]
        if all(not np.isnan(xx) for xx in x):
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            if r_value**2 > 0.1:
                print (c,"r-squared:", r_value**2)



    # starting from this point
    # x_train is not modified anymore


    print(x_train.head())
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    print("total len", len(x_train))


    '''
    if not random_split:
        x_valid = x_train[-nb_valid:]
        y_valid = y_train[-nb_valid:]
        x_train = x_train[:-nb_valid]
        y_train = y_train[:-nb_valid]
    else:
        x_train,x_valid,y_train,y_valid = train_test_split( x_train, y_train, test_size=0.1, random_state=42)
    '''

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)
#    dvalid = xgb.DMatrix(x_valid)

    cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=5000, early_stopping_rounds=20, verbose_eval=50, show_stdv=False)
    cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

    num_boost_rounds = len(cv_output)
    print("num_boost_rounds", num_boost_rounds)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=
            num_boost_rounds)

    '''
    # evaluate validation
    valid_pred = model.predict(dvalid)
    #print("Validation rmlse", rmsle(valid_pred, np.array(y_valid)))
    res = []
    for x in frange(0.9, 1.1, 0.01):
        error = rmsle(valid_pred*x, np.array(y_valid))
        res.append( (error,x) )
        print("Validation rmlse * {0:.2f}".format(x), rmsle(valid_pred*x, np.array(y_valid)))

    bst = min(res)
    print(bst)
    y_predict = model.predict(dtest)*bst[1]
    '''

    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    output.head()
    calc(output)

    output.to_csv('add_macro_with_scaledown.csv', index=False)
main()
