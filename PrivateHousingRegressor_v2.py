# explore the data set

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
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
from xgboost import plot_importance

def count_remaining_years(row):
  current_year = datetime.today().year
  tenure_years = int(row["tenure_year"])
  tenure_start = row["tenure_start_date"].year
  tenure_end = row["tenure_start_date"].year + tenure_years
  tenure_remain =  tenure_end - current_year
  return tenure_remain

def encode_onehot(data):
  label_encoder = LabelEncoder()
  feature = label_encoder.fit_transform(data)
  feature = feature.reshape(feature.shape[0], 1)
  onehot_encoder = OneHotEncoder(sparse=False)
  feature = onehot_encoder.fit_transform(feature)
  return feature

def encode_label(data):
  label_encoder = LabelEncoder()
  feature = label_encoder.fit_transform(data)
  return feature  

def set_default_floor_and_unit_num(data):
  # data = data.fillna(1) # default to 15th floor (simulate high floor privilege) with unit number 1
  data.loc[:, "floor_num"] = data.loc[:, "floor_num"].fillna(15)
  data.loc[:, "unit_num"] = data.loc[:, "unit_num"].fillna(1)

column_names = [
  "project_name",
  "address",
  "floor_area_sqm",
  "type_of_land",
  "price",
  "contract_date",
  "property_type",
  "tenure",
  "completion_date",
  "type_of_sale",
  "postal_district",
  "postal_sector",
  "postal_code",
  "region",
  "area",
  "month",
  "latitude",
  "longitude",
  "floor_num",
  "unit_num"
]

train_file = 'private_train_2.csv'
test_file = 'private_test.csv'

# train_data = read_csv(train_file, usecols=column_names).head(150000)
train_data = read_csv(train_file, usecols=column_names)
test_data = read_csv(test_file)

print("A few records of training data : ", train_data.head(20))
print("Shape of training data : ", train_data.shape)
print("Shape of test data : ", test_data.shape)
print("Data types of training data : ", train_data.dtypes)
print("Count of training data : ", train_data.count())

# get address_block and address_street
blocks_streets_units = train_data.address.str.split("#")
blocks_streets = blocks_streets_units.str.get(0)
splitted_blocks_streets = blocks_streets.str.split(" ", 1)
blocks = splitted_blocks_streets.str.get(0)
streets = splitted_blocks_streets.str.get(1)
train_data["address_block"] = blocks
train_data["address_street"] = streets 

# get remaining tenure
now = datetime.now()
date_pattern = "\d\d/\d\d/\d\d\d\d"
train_data.tenure = train_data.tenure.str.strip()
train_data["tenure_year"] = where(train_data.tenure.str.contains(date_pattern),train_data.tenure.str.split(" ", 1).str.get(0), "999")
train_data["tenure_start"] = where(train_data.tenure.str.contains(date_pattern), train_data.tenure.str.rsplit(" ", 1).str.get(1), "01/01/2019")
train_data["tenure_start_date"] = to_datetime(train_data["tenure_start"], format="%d/%m/%Y")
train_data['tenure_remain'] = train_data.apply(count_remaining_years, axis=1)


print(train_data.head(20))
print(train_data.dtypes)

train_set = train_data[[
  "project_name",
  "floor_area_sqm",
  "type_of_land",
  "property_type",
  "completion_date",
  "type_of_sale",
  "postal_district",
  "postal_sector",
  "postal_code",
  "region",
  "area",
  "month",
  "latitude",
  "longitude",
  "floor_num",
  "unit_num",
  "address_block",
  "address_street",
  "tenure_remain",
  "price"]]

print(train_set.head(5))
print(train_set.shape)
print(train_set.dtypes)

train_set.project_name = encode_label(train_set.project_name)
# train_set.floor_area_sqm = encode_label(train_set.floor_area_sqm)
train_set.type_of_land = encode_label(train_set.type_of_land)
train_set.property_type = encode_label(train_set.property_type)
train_set.completion_date = encode_label(train_set.completion_date)
train_set.type_of_sale = encode_label(train_set.type_of_sale)
# train_set.postal_district = encode_label(train_set.postal_district)
# train_set.postal_sector = encode_label(train_set.postal_sector)
# train_set.postal_code = encode_label(train_set.postal_code)
train_set.region = encode_label(train_set.region)
train_set.area = encode_label(train_set.area)
train_set.month = encode_label(train_set.month)
# train_set.latitude = encode_label(train_set.latitude)
# train_set.longitude = encode_label(train_set.longitude)
# train_set.floor_num = encode_label(train_set.floor_num)
# train_set.unit_num = encode_label(train_set.unit_num)
train_set.address_block = encode_label(train_set.address_block)
train_set.address_street = encode_label(train_set.address_street)
# train_set.tenure_remain = encode_label(train_set.tenure_remain)

set_default_floor_and_unit_num(train_set)

print(train_set.head(5))
print(train_set.shape)
print(train_set.dtypes)

# descriptions
set_option("precision", 1)
print("Description of training data : ")
print(train_set.describe())

# correlation
set_option("precision", 2)
print("Correlation of training data : ")
print(train_set.corr(method="pearson"))

# some plots
sns.regplot(x="floor_area_sqm", y="price", data=train_set)
pyplot.show()

price_plot = sns.distplot(train_set.price, label="skewness: %.2f" % train_set.price.skew())
price_plot = price_plot.legend(loc="best")
pyplot.show()

train_set.loc[:,"price"] = train_set.price.map(lambda x: log1p(x) if x > 0 else 0)
price_log_plot = sns.distplot(train_set.price, label="skewness: %.2f" % train_set.price.skew())
price_log_plot = price_log_plot.legend(loc="best")
pyplot.show()

skews = train_set.skew().sort_values(ascending = False)
print(skews)

tenure_remain_plot = sns.distplot(train_set.tenure_remain, label="skewness: %.2f" % train_set.tenure_remain.skew())
tenure_remain_plot = tenure_remain_plot.legend(loc="best")
pyplot.show()

train_set.loc[:,"tenure_remain"] = train_set.tenure_remain.map(lambda x: log1p(x) if x > 0 else 0)
tenure_remain_log_plot = sns.distplot(train_set.tenure_remain, label="skewness: %.2f" % train_set.tenure_remain.skew())
tenure_remain_log_plot = tenure_remain_log_plot.legend(loc="best")
pyplot.show()

dataset = train_set[[
  "floor_area_sqm",
  "type_of_land",
  "property_type",
  "area",
  "latitude",
  "longitude",
  "floor_num",
  "tenure_remain",
  "price"
]].values

X = dataset[:, 0:7]
Y = dataset[:, 8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

model = XGBRegressor()

scoring = "neg_mean_absolute_error"
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

print(model)

print("Mean absolute error from kfold: ", results.mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mape = mean(abs((y_test - y_pred)/y_test))

print("MAPE: %2.f%%" % (mape * 100.0))

plot_importance(model)
pyplot.show() 

