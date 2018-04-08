import numpy
import seaborn as sns

from datetime import datetime 
from datetime import timedelta

from numpy import arange
from numpy import column_stack
from numpy import where
from numpy import mean
from numpy import abs 
from numpy import log1p

from matplotlib import pyplot

from pandas import read_csv
from pandas import set_option
from pandas import to_datetime
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error 

from xgboost import XGBRegressor
from xgboost import plot_importance

def encode_onehot(data):
  label_encoder = LabelEncoder()
  feature = label_encoder.fit_transform(data)
  feature = feature.reshape(feature.shape[0], 1)
  onehot_encoder = OneHotEncoder(sparse=False)
  feature = onehot_encoder.fit_transform(feature)
  return feature

def set_default_floor_and_unit_num(data):
  # default to 15th floor (simulate high floor privilege) 
  data.loc[:, "floor_num"] = data.loc[:, "floor_num"].fillna(15)

train_column_names = [
  "floor_area_sqm",
  "latitude",
  "longitude",
  "floor_num",
  "price"
]

test_column_names = [
  "index",
  "floor_area_sqm",
  "latitude",
  "longitude",
  "floor_num",
]

train_file = 'private_train_2.csv'
test_file = 'data/private_test.csv'

all_train_data = read_csv(train_file)
all_test_data = read_csv(test_file)

# assumption
# sellers as well as buyers usually need to know 
# the latest price of their desired house/apartment/condo.
# if latest price of exact place is not available,
# they need to know latest price of places nearby.
# they do not want the old price of exact place/nearby place
# because they want to buy or sell according to latest competitive market price.
# hence, we keep only 2017 data to train our model.
all_train_data = all_train_data[all_train_data.month.str.contains('2017')]

train_data = all_train_data[train_column_names]
test_data = all_test_data[test_column_names]

sns.regplot(x="floor_area_sqm", y="price", data=all_train_data)
pyplot.show()

print("A few records of training data : ")
print(train_data.head(20))
print("Shape of training data : ")
print(train_data.shape)
print("Shape of test data : ")
print(test_data.shape)
print("Data types of training data : ")
print(train_data.dtypes)
print("Count of training data : ")
print(train_data.count())

set_default_floor_and_unit_num(train_data)
set_default_floor_and_unit_num(test_data)
print(train_data.head(20))

train_price = train_data[["price"]]
print(train_price.head(5))

# scale all the values 
scaler = StandardScaler()
train_data[train_data.columns] = scaler.fit_transform(train_data[train_data.columns])

scalerY = StandardScaler()
train_price[train_price.columns] = scalerY.fit_transform(train_price[train_price.columns])

scalerT = StandardScaler()
test_data[test_data.columns] = scalerT.fit_transform(test_data[test_data.columns])

# train
dataset = train_data[[
  "floor_area_sqm",
  "latitude",
  "longitude",
  "floor_num",
  "price"
]].values

X = dataset[:, 0:3]
Y = dataset[:, 4]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# best parameters are obtained from separate long running script
model = XGBRegressor(
  nthreads=-1,
  subsample=0.5,
  n_estimators=1000,
  min_child_weight=2,
  max_depth=10, 
  learning_rate=0.1,
  colsample_bylevel=0.4
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pred_errors = mean_absolute_error(y_test, y_pred)
print("mean absolute error (scaled): %.2f " % pred_errors)

print(model)

y_pred_normal = scalerY.inverse_transform(y_pred)
print("Predicted prices (training)")
print(y_pred_normal)

y_test_normal = scalerY.inverse_transform(y_test)
print("Target prices (training)")
print(y_test_normal)

mape = mean(abs((y_test_normal - y_pred_normal)/y_test_normal))

print("MAPE (normal value): %.5f%%" % (mape * 100.0))
plot_importance(model)
pyplot.show() 

actual_test = test_data.values[:, 1:4]
actual_pred = model.predict(actual_test)
actual_pred_normal = scalerY.inverse_transform(actual_pred)
print("Predicted prices (to submit)")
print(actual_pred_normal)

all_test_data["price"] = actual_pred_normal
all_test_data.to_csv('private_predicted.csv',index=False)


