# explore the data set

import numpy

from datetime import datetime 
from datetime import timedelta

from numpy import arange
from numpy import column_stack
from numpy import where
from numpy import mean
from numpy import abs 

from matplotlib import pyplot

from pandas import read_csv
from pandas import set_option
from pandas import to_datetime
from pandas.plotting import scatter_matrix

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

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

# assumed that run time python script and data files are in same folder
# must use 'private_train_2.csv' because original data file has one row with missing data
# 'private_train_2.csv' has removed that row.  row no. 52830 with "n" for address value.
train_file = 'private_train_2.csv'
test_file = 'private_test.csv'

train_data = read_csv(train_file, usecols=column_names) #.head(52829)
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

print(train_set.shape)
print(train_set.head(5))

dataset = train_set.values
X = dataset[:, 0:18]
Y = dataset[:, 19]

features = []
# features.append(encode_onehot(train_set.project_name.values))
features.append(train_set.floor_area_sqm.values)
features.append(encode_onehot(train_set.type_of_land.values))
features.append(encode_onehot(train_set.property_type.values))
# features.append(encode_onehot(train_set.completion_date.values))
features.append(encode_onehot(train_set.type_of_sale.values))
# features.append(train_set.postal_district.values)
# features.append(train_set.postal_sector.values)
# features.append(train_set.postal_code.values)
features.append(encode_onehot(train_set.region.values))
features.append(encode_onehot(train_set.area.values))
# features.append(encode_onehot(train_set.month.values))
# features.append(train_set.latitude.values)
# features.append(train_set.longitude.values)
features.append(train_set.floor_num.values)
features.append(train_set.unit_num.values)
features.append(encode_onehot(train_set.address_block.values))
features.append(encode_onehot(train_set.address_street.values))
features.append(train_set.tenure_remain.values)

encoded_x = column_stack(features)

print("X shape: ", encoded_x.shape)

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(encoded_x, Y, test_size=test_size, random_state=seed)

kfold = KFold(n_splits=5, random_state=seed)

model = XGBRegressor()
model.fit(X_train, y_train)

scoring = "neg_mean_absolute_error"
results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

print(model)

print("Mean absolute error from kfold: ", results.mean())

y_pred = model.predict(X_test)

mape = mean(abs((y_test - y_pred)/y_test))

print("MAPE: %2.f%%" % (mape * 100.0))







