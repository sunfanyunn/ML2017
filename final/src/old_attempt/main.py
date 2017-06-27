import operator
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

def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

#load files
train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])
id_test = test.id

#clean data
bad_index = train[train.life_sq > train.full_sq].index
train.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.loc[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.loc[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.loc[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.loc[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.loc[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.loc[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index
train.loc[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index
test.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.loc[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.loc[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.loc[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.loc[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.loc[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.loc[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.loc[bad_index, "state"] = np.NaN
test.state.value_counts()

# brings error down a lot by removing extreme price per sqm
train.loc[train.full_sq == 0, 'full_sq'] = 50
train = train[train.price_doc/train.full_sq <= 600000]
train = train[train.price_doc/train.full_sq >= 10000]

# Add month-year
month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
train['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
test['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
train['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
test['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
train['month'] = train.timestamp.dt.month
train['dow'] = train.timestamp.dt.dayofweek

test['month'] = test.timestamp.dt.month
test['dow'] = test.timestamp.dt.dayofweek

# Other feature engineering
train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
test['room_size'] = test['life_sq'] / test['num_room'].astype(float)

################
# drop useless
#################
train = train.iloc[4000:]
train = train.drop(train[(train.price_doc <= 1e6) & (train.product_type == 'Investment' )].index)

print('before adjustment ...')
y_train = np.log1p(train["price_doc"])

train = train.drop(['price_doc'], axis=1)

print(train.shape, test.shape)
###################################
# Add my adjustment from here
###################################
useless = [ "ID_railroad_station_walk", "ID_big_road1",
        "ID_railroad_terminal", "ID_metro", "ID_railroad_station_avto",
        "ID_big_road2", "ID_bus_terminal" ]
train = train.drop( useless, axis=1 )
test = test.drop( useless, axis=1 )

#train['f-l'] = train['full_sq'] - train['life_sq']
#test['f-l'] = test['full_sq'] - test['life_sq']

print(train.shape, test.shape)

macro_features =  ["timestamp", "cpi","balance_trade","mortgage_rate","usdrub"]
my_macro=macro[macro_features]

df_all = pd.concat([train, test])
df_all = df_all.join(my_macro, on='timestamp', rsuffix='_macro')
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)
print(df_all.shape)

train = df_all.iloc[:train.shape[0]]
test = df_all.iloc[train.shape[0]:]
print(train.shape, test.shape)
print(train.head())
print(test.head())
#################################

x_train = train.drop(["id"], axis=1)
x_test = test.drop(["id"], axis=1)

num_train = len(x_train)
x_all = pd.concat([x_train, x_test])

for c in x_all.columns:
    if x_all[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_all[c].values))
        x_all[c] = lbl.transform(list(x_all[c].values))
        #x_train.drop(c,axis=1,inplace=True)


x_train = x_all[:num_train]
x_test = x_all[num_train:]
print(x_train.shape, x_test.shape)

xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
tmp = pd.read_csv('fuckin_test.csv')['price_doc']
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=20, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
#cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)
print('best num_boost_rounds = ', len(cv_output))
num_boost_rounds = len(cv_output)

train_rmsle = []
test_rmsle = []
c_train = []
c_test = []

cnt = 0
def evalerror(preds, dtrain):
    global cnt
    cnt += 1
    if cnt % 20 == 0:
        plt.plot(np.arange(len(train_rmsle)),train_rmsle, label='train')
        plt.plot(np.arange(len(test_rmsle)), test_rmsle, label='test')
        plt.legend()
        plt.savefig('rmsle')
        plt.figure()
        plt.plot(np.arange(len(c_train)), c_train, label='train')
        plt.plot(np.arange(len(c_test)), c_test, label='test')
        plt.legend()
        plt.savefig('custom metrics')
        plt.figure()
        print(cnt)

    labels = dtrain.get_label()
    r = rmsle(labels, preds)
    c = np.sqrt( (np.log1p(preds)**2).mean() )
    #print('rmsle', r)
    #print ('custom metrics', c)
    if preds.shape[0] > 10000:
        #print('train')
        train_rmsle.append(r)
        c_train.append(c)
    else:
        #print('test')
        test_rmsle.append(r)
        c_test.append(c)
        print('custom metrics on test',c)
    return ('custom metrics', -np.sqrt( (np.log1p(preds)**2).mean() ))

#watchlist = [(dtest, 'eval'), (dtrain, 'train')]
#num_boost_rounds = 422
#model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_rounds,maximize=True, verbose_eval=1,evals=watchlist,feval=evalerror)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_rounds)

importance = model.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] =100* df['fscore'] / df['fscore'].max()
df=df.sort_values(by="fscore",ascending=False)
df.head(50).plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
plt.title('XGBoost Feature Importance(50 significant)')
plt.xlabel('relative importance')
plt.savefig('feature_importance.png')

y_predict = model.predict(dtest)
y_predict = np.exp(y_predict)-1
# y_predict = np.round(y_predict)
df = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

tmp = np.log1p(df['price_doc'])
print('sqrt( mean of log1p**2 )', np.sqrt( (tmp**2).mean()))

df.to_csv('main{}.csv'.format(num_boost_rounds), index=False)

