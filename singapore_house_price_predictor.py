# how to use
# make sure you have data set in the same folder as this program
# to explore datasets
#   uncomment the explore_hdb, or explore_private methods
# to predict
#   uncomment the predict_hdb, or predict_private methods, or both
# from terminal type:
#   py singapore_house_price_predictor
# 
# References:
# http://danielhnyk.cz/how-to-use-xgboost-in-python/
# https://jessesw.com/XG-Boost/ 
# https://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/
# http://www.codeastar.com/win-big-real-estate-market-data-science/
# https://medium.com/@blazetamareborn/practicing-regression-techniques-on-house-prices-dataset-part-1-7266aec8b7d3
# https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
# https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
# https://pandas.pydata.org/pandas-docs/stable/merging.html#appending-rows-to-a-dataframe
# 

import numpy as np 
import pandas as pd 
import seaborn as sn 
import matplotlib.pyplot as plt 
import scipy.stats as st
import xgboost as xgb

from scipy.special import boxcox1p, inv_boxcox1p

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn.linear_model import ElasticNet 

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance

hdb_train_file = 'data/hdb_train.csv'
hdb_test_file = 'data/hdb_test.csv'

private_train_file = 'data/private_train_2.csv'
private_test_file = 'data/private_test.csv'

def explore(dataset_name, dataset_to_explore):
  print("---- %s ----" % dataset_name)
  print("Shape:")
  print(dataset_to_explore.shape)
  print(" ")
  print("Head:")
  print(dataset_to_explore.head())
  print(" ")
  print("Data Types:")
  print(dataset_to_explore.dtypes)
  print(" ")
  print("Data Summary:")
  print(dataset_to_explore.describe())
  print(" ")
  print("Data Count")
  print(dataset_to_explore.count())
  print("---- End Explore ----")

def encode_label(column_data):
  le = LabelEncoder()
  labels = le.fit_transform(column_data)
  mappings = {index: l for index, l in enumerate(le.classes_)}
  return labels

def set_hdb_data_types(data):
  data["block"] = data.block.astype("str").astype("category")
  data["flat_model"] = data.flat_model.astype("category")
  data["flat_type"] = data.flat_type.astype("category")
  data["lease_commence_date"] = data.lease_commence_date.astype("str").astype("category")
  data["month"] = data.month.astype("str").astype("category")
  data["storey_range"] = data.storey_range.astype("category")
  data["street_name"] = data.street_name.astype("category")
  data["town"] = data.town.astype("category")
  data["postal_code"] = data.postal_code.astype("str").astype("category")

def set_private_data_types(data):
  data["project_name"] = data.project_name.astype("category")
  data["address"] = data.address.astype("category")
  data["type_of_land"] = data.type_of_land.astype("category")
  data["contract_date"] = data.contract_date.astype("category")
  data["property_type"] = data.property_type.astype("category")
  data["tenure"] = data.tenure.astype("category")
  data["completion_date"] = data.completion_date.astype("str").astype("category")
  data["type_of_sale"] = data.type_of_sale.astype("category")
  data["region"] = data.region.astype("category")
  data["area"] = data.area.astype("category")
  data["month"] = data.month.astype("str").astype("category")
  data["floor_num"] = data.floor_num.astype("str").astype("category")
  data["unit_num"] = data.unit_num.astype("str").astype("category")

def convert_hdb_to_numeric(data):
  data["flat_model_label"] = encode_label(data["flat_model"])
  data["flat_type_label"] = encode_label(data["flat_type"])
  data["lease_commence_date_label"] = encode_label(data["lease_commence_date"])
  data["storey_range_label"] = encode_label(data["storey_range"])
  data["street_name_label"] = encode_label(data["street_name"])
  data["block_label"] = encode_label(data["block"])
  data["town_label"] = encode_label(data["town"])
  data["postal_code_label"] = encode_label(data["postal_code"])
  data["month_label"] = encode_label(data["month"])

def convert_private_to_numeric(data):
  data["project_name_label"] = encode_label(data["project_name"])
  data["address_label"] = encode_label(data["address"])
  data["type_of_land_label"] = encode_label(data["type_of_land"])
  data["contract_date_label"] = encode_label(data["contract_date"])
  data["property_type_label"] = encode_label(data["property_type"])
  data["tenure_label"] = encode_label(data["tenure"])
  data["completion_date_label"] = encode_label(data["completion_date"])
  data["type_of_sale_label"] = encode_label(data["type_of_sale"])
  data["region_label"] = encode_label(data["region"])
  data["area_label"] = encode_label(data["area"])
  data["month_label"] = encode_label(data["month"])
  data["floor_num"] = encode_label(data["floor_num"])
  data["unit_num"] = encode_label(data["unit_num"])

# explore HDB dataset
def explore_hdb():
  data = pd.read_csv(hdb_train_file)
  explore("HDB Training Dataset", data)

  set_hdb_data_types(data)

  prices = data[["resale_price"]]
  price_dist = sn.distplot(prices, label="Skewness : %.2f"%(prices.skew()))
  price_dist = price_dist.legend(loc="best")
  plt.show()

  fig,ax = plt.subplots()
  sn.pointplot(data=data[[
    "resale_price", 
    "flat_model", 
    "lease_commence_date"]],
    x="lease_commence_date",
    y="resale_price",
    hue="flat_model",
    ax=ax)
  ax.set(title="HDB Flat model prices in over the lease commence date")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.boxplot(data=data[[
    "resale_price",
    "flat_model"]],
    x="flat_model",
    y="resale_price",
    ax=ax)
  ax.set(title="HDB Flat model vs prices")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.pointplot(data=data[[
    "resale_price", 
    "flat_type", 
    "lease_commence_date"]],
    x="lease_commence_date",
    y="resale_price",
    hue="flat_type",
    ax=ax)
  ax.set(title="HDB Flat type prices over the lease commence date")

  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.boxplot(data=data[[
    "resale_price",
    "flat_type"]],
    x="flat_type",
    y="resale_price",
    ax=ax)
  ax.set(title="HDB Flat type vs prices")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.pointplot(data=data[[
    "resale_price", 
    "storey_range", 
    "lease_commence_date"]],
    x="lease_commence_date",
    y="resale_price",
    hue="storey_range",
    ax=ax)
  ax.set(title="HDB Storey range prices in over the lease commence date")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.boxplot(data=data[[
    "resale_price",
    "storey_range"]],
    x="storey_range",
    y="resale_price",
    ax=ax)
  ax.set(title="HDB Storey range vs prices")
  plt.xticks(rotation=70)
  plt.show()

  convert_hdb_to_numeric(data)

  correlations = data[[
    "resale_price",
    "floor_area_sqm",
    "flat_model_label",
    "flat_type_label",
    "lease_commence_date_label",
    "storey_range_label",
    "floor",
    "street_name_label",
    "block_label",
    "latitude",
    "longitude",
    "town_label",
    "postal_code_label",
    "month_label"]].corr() 

  correlations = correlations.nlargest(10000, "resale_price")
  
  mask = np.array(correlations)
  mask[np.tril_indices_from(mask)] = False

  sn.heatmap(correlations, mask=mask, vmax=.8, square=True, annot=True)
  plt.show()

  print(data.dtypes)
  print(data.shape)

# explore Private Housing dataset
def explore_private_housing():
  data = pd.read_csv(private_train_file)
  explore("Private Housing Training Dataset", data)

  set_private_data_types(data)

  prices = data[["price"]]
  price_dist = sn.distplot(prices, label="Skewness : %.2f"%(prices.skew()))
  price_dist = price_dist.legend(loc="best")
  plt.show()

  fig,ax = plt.subplots()
  sn.pointplot(data=data[[
    "price", 
    "property_type", 
    "completion_date"]],
    x="completion_date",
    y="price",
    hue="property_type",
    ax=ax)
  ax.set(title="Private Hosing property type prices over completion date")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.boxplot(data=data[[
    "price",
    "property_type"]],
    x="property_type",
    y="price",
    ax=ax)
  ax.set(title="Private Hosing property type vs prices")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.pointplot(data=data[[
    "price", 
    "floor_num", 
    "completion_date"]],
    x="completion_date",
    y="price",
    hue="floor_num",
    ax=ax)
  ax.set(title="Private Housing floor number vs prices over the completion date")
  plt.xticks(rotation=70)
  plt.show()

  fig,ax = plt.subplots()
  sn.boxplot(data=data[[
    "price",
    "floor_num"]],
    x="floor_num",
    y="price",
    ax=ax)
  ax.set(title="Private Housing floor number vs prices")
  plt.xticks(rotation=70)
  plt.show()

  convert_private_to_numeric(data)

  correlations = data[[
    "price",
    "project_name_label",
    "address_label",
    "floor_area_sqm",
    "type_of_land_label",
    "contract_date_label",
    "property_type_label",
    "tenure_label",
    "completion_date_label",
    "type_of_sale_label",
    "postal_district",
    "postal_sector",
    "postal_code",
    "region_label",
    "area_label",
    "month_label",
    "latitude",
    "longitude",
    "floor_num",
    "unit_num"]].corr() 

  correlations = correlations.nlargest(10000, "price")
  mask = np.array(correlations)
  mask[np.tril_indices_from(mask)] = False

  sn.heatmap(correlations, mask=mask, vmax=.8, square=True, annot=True)
  plt.show()

  print(data.dtypes)
  print(data.shape)

  print("Number of unique values in categories:")

def preprocess_hdb_features(data):

  # very important to reset the index
  # otherwise the FeatureHasher concat will produce wrong concat results
  data = data.reset_index(drop=True)

  set_hdb_data_types(data)
  
  numeric_feature_columns = [
    "floor_area_sqm",
    "latitude",
    "longitude",
    "floor"
  ]
  
  # perform feature hashing
  flat_model_fh = FeatureHasher(n_features=5, input_type="string")
  flat_model_hashed = flat_model_fh.fit_transform(data.flat_model)

  flat_type_fh = FeatureHasher(n_features=3, input_type="string")
  flat_type_hashed = flat_type_fh.fit_transform(data.flat_type)

  lease_commence_date_fh = FeatureHasher(n_features=3, input_type="string")
  lease_commence_date_hashed = lease_commence_date_fh.fit_transform(data.lease_commence_date)

  storey_range_fh = FeatureHasher(n_features=7, input_type="string")
  storey_range_hashed = storey_range_fh.fit_transform(data.storey_range)

  street_name_fh = FeatureHasher(n_features=7, input_type="string")
  street_name_hashed = street_name_fh.fit_transform(data.street_name)

  block_fh = FeatureHasher(n_features=7, input_type="string")
  block_hashed = block_fh.fit_transform(data.block)

  town_fh = FeatureHasher(n_features=7, input_type="string")
  town_hashed = town_fh.fit_transform(data.town)

  postal_code_fh = FeatureHasher(n_features=7, input_type="string")
  postal_code_hashed = postal_code_fh.fit_transform(data.postal_code)

  month_fh = FeatureHasher(n_features=7, input_type="string")
  month_hashed = month_fh.fit_transform(data.month)

  features = pd.concat([
      data[numeric_feature_columns], 
      pd.DataFrame(flat_model_hashed.toarray()),
      pd.DataFrame(flat_type_hashed.toarray()),
      pd.DataFrame(lease_commence_date_hashed.toarray()),
      pd.DataFrame(storey_range_hashed.toarray()),
      pd.DataFrame(street_name_hashed.toarray()),
      pd.DataFrame(block_hashed.toarray()),
      pd.DataFrame(town_hashed.toarray()),
      pd.DataFrame(postal_code_hashed.toarray()),
      pd.DataFrame(month_hashed.toarray())
    ],
    axis=1)
  
  print("Shape of preprocessed data (after feature hashing):")
  print(features.shape)
  return features

def preprocess_private_features(data):
  # very important to reset the index
  # otherwise the FeatureHasher concat will produce wrong concat results
  data = data.reset_index(drop=True)

  set_private_data_types(data)
  
  numeric_feature_columns = [
    "floor_area_sqm",
    "postal_district",
    "postal_code",
    "latitude",
    "longitude"
  ]
  
  # perform feature hashing

  # -- will not use --
  # tenure, 850 = 10 
  # address, too many unique values
  # type of sale, 3 = 2
  # contract date, 7000++ = 13
  # type of land = 2
  # region, 5 = 3
  
  # -- will use --
  # month, 272 = 9
  # property_type = 3
  # area, 40 = 6
  # completion date, 120 = 7
  # project name = 12
  # floor num, 71 = 7

  month_fh = FeatureHasher(n_features=9, input_type="string")
  month_hashed = month_fh.fit_transform(data.month)

  property_type_fh = FeatureHasher(n_features=3, input_type="string")
  property_type_hashed = property_type_fh.fit_transform(data.property_type)

  area_fh = FeatureHasher(n_features=6, input_type="string")
  area_hashed = area_fh.fit_transform(data.area)

  completion_date_fh = FeatureHasher(n_features=7, input_type="string")
  completion_date_hashed = completion_date_fh.fit_transform(data.completion_date)

  project_name_fh = FeatureHasher(n_features=12, input_type="string")
  project_name_hashed = project_name_fh.fit_transform(data.project_name)

  floor_num_fh = FeatureHasher(n_features=7, input_type="string")
  floor_num_hashed = floor_num_fh.fit_transform(data.floor_num)

  unit_num_fh = FeatureHasher(n_features=10, input_type="string")
  unit_num_hashed = unit_num_fh.fit_transform(data.unit_num)

  features = pd.concat([
      data[numeric_feature_columns], 
      pd.DataFrame(month_hashed.toarray()),
      pd.DataFrame(property_type_hashed.toarray()),
      pd.DataFrame(area_hashed.toarray()),
      pd.DataFrame(completion_date_hashed.toarray()),
      pd.DataFrame(project_name_hashed.toarray()),
      pd.DataFrame(floor_num_hashed.toarray()),
      pd.DataFrame(unit_num_hashed.toarray())
    ],
    axis=1)

  features["floor_area_sqm"] = boxcox1p(features["floor_area_sqm"], 0.1)
  
  print("Shape of preprocessed data (after feature hashing):")
  print(features.shape)
  return features

def preprocess_price(data):
  prices = boxcox1p(data, 0.1)

  sn.distplot(prices, fit=st.norm)
  (mu, sigma) = st.norm.fit(prices)
  plt.legend([
      "Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )".format(mu, sigma),
      "Skewness: %.2f" % prices.skew()
    ],
    loc="best")
  
  plt.ylabel("Frequency")
  plt.title("Resale Price distribution")

  fig = plt.figure()
  res = st.probplot(prices, plot=plt)
  plt.show()

  return prices

def evaluate_linear_algorithms(num_folds, seed, scoring, X_train, Y_train):
  models = []
  models.append(("LR", LinearRegression()))
  models.append(("LASSO", Lasso()))
  models.append(("EN", ElasticNet()))
  models.append(("KNN", KNeighborsRegressor()))
  models.append(("CART", DecisionTreeRegressor()))
  models.append(("SVR", SVR()))

  results = []
  names = []
  for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    message = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(message)

  # compare algorithms
  fig = plt.figure()
  fig.suptitle("Algorithm Comparison")
  ax = fig.add_subplot(111)
  plt.boxplot(results)
  ax.set_xticklabels(names)
  plt.show()

def evaluate_ensemble_algorithms(num_folds, seed, scoring, X_train, Y_train):
  ensembles = []
  ensembles.append(("AB", AdaBoostRegressor()))
  ensembles.append(("GBM", GradientBoostingRegressor()))
  ensembles.append(("RF", RandomForestRegressor()))
  ensembles.append(("ET", ExtraTreesRegressor()))
  ensembles.append(("XGB", XGBRegressor()))

  results = []
  names = []
  for name, model in ensembles:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    message = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(message)

  # compare algorithms
  fig = plt.figure()
  fig.suptitle("Ensemble Algorithm Comparison")
  ax = fig.add_subplot(111)
  plt.boxplot(results)
  ax.set_xticklabels(names)
  plt.show()

def tune_gbm(param_grid, seed, scoring, kfold, X_train, Y_train):
  model = GradientBoostingRegressor(random_state=seed)
  grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring, cv=kfold)
  grid_results = grid.fit(X_train, Y_train)

  print("Tuning Results for GBM")
  print("Best Parameters: %f using %s " % (grid_results.best_score_, grid_results.best_params_))
  means = grid_results.cv_results_["mean_test_score"]
  stds = grid_results.cv_results_["std_test_score"]
  params = grid_results.cv_results_["params"]
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" %  (mean, stdev, param))

def tune_xgb(param_grid, scoring, kfold, X_train, Y_train):
  model = XGBRegressor()
  grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring, cv=kfold)
  grid_results = grid.fit(X_train, Y_train)

  print("Tuning Results for XGB")
  print("Best Parameters: %f using %s " % (grid_results.best_score_, grid_results.best_params_))
  means = grid_results.cv_results_["mean_test_score"]
  stds = grid_results.cv_results_["std_test_score"]
  params = grid_results.cv_results_["params"]
  for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" %  (mean, stdev, param))

def predict(model, X_train, Y_train, X_test, Y_test, test_features):

  model.fit(X_train, Y_train)
  P_train = model.predict(X_test)
  print("Mean Absolute Error (training) while in scaled value:")
  print(mean_absolute_error(Y_test, P_train))

  P_train_unscale = inv_boxcox1p(P_train, 0.1)
  print("Predict price (training)")
  print(P_train_unscale)

  Y_test_unscale = inv_boxcox1p(Y_test, 0.1)
  print("Target price (training)")
  print(Y_test_unscale)

  print("Mean Absolute Error (training) manual calculation in UNSCALED value:")
  mae = np.mean(np.abs((Y_test_unscale - P_train_unscale)/Y_test_unscale)) * 100
  print(mae)

  P_test = model.predict(test_features)
  P_test_unscale = inv_boxcox1p(P_test, 0.1)

  print("Target price (test)")
  print(P_test_unscale)

  return P_test_unscale


def predict_hdb():
  all_data = pd.read_csv(hdb_train_file)

  data_2015 = all_data[all_data.month.str.contains('2015')]
  data_2016 = all_data[all_data.month.str.contains('2016')]
  data_2017 = all_data[all_data.month.str.contains('2017')]
  
  data = pd.concat([
    data_2015,
    data_2016,
    data_2017
  ])#.head(50)

  prices = preprocess_price(data["resale_price"])
  features = preprocess_hdb_features(data)
  
  X = features.values
  Y = prices.values

  test_size = 0.20
  seed = 7

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

  # Evaluate Algorithms
  num_folds = 10
  seed = 7
  # scoring = "neg_mean_squared_error"
  scoring = "neg_mean_absolute_error"

  # evaluate_linear_algorithms(num_folds=num_folds, seed=seed, scoring=scoring, X_train=X_train, Y_train=Y_train)
  # evaluate_ensemble_algorithms(num_folds=num_folds, seed=seed, scoring=scoring, X_train=X_train, Y_train=Y_train)

  # tune algorithms
  n_estimators = [100, 500, 1000, 2000, 3000]
  max_depth = [4,6,8,10]
  learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2]
  param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
  kfold = KFold(n_splits=num_folds, random_state=seed)
  
  # tune_gbm(
  #   param_grid=param_grid,
  #   seed=seed,
  #   scoring=scoring,
  #   kfold=kfold,
  #   X_train=X_train,
  #   Y_train=Y_train
  # )
  
  # tune_xgb(
  #   param_grid=param_grid,
  #   kfold=kfold,
  #   scoring=scoring,
  #   X_train=X_train,
  #   Y_train=Y_train
  # )

  # predict with best parameters
  # model = GradientBoostingRegressor(n_estimators=500, max_depth=6) 
  model = XGBRegressor(
    nthreads=-1, 
    n_estimators=1000, 
    max_depth=6,
    learning_rate=0.1) 

  test_data = pd.read_csv(hdb_test_file)
  test_features = preprocess_hdb_features(test_data)
  test_features = test_features.values
  
  P_test_unscale = predict(model, 
    X_train, 
    Y_train, 
    X_test, 
    Y_test, 
    test_features)

  test_data["resale_price"] = P_test_unscale
  test_data["resale_price"] = test_data.resale_price.astype(int)
  print(test_data["resale_price"].head())

  test_data.to_csv('hdb_predicted.csv',index=False)

  # plot_importance(model)
  # plt.show() 


def predict_private():
  all_data = pd.read_csv(private_train_file)

  data_2013 = all_data[all_data.month.str.contains('2013')]
  data_2014 = all_data[all_data.month.str.contains('2014')]
  data_2015 = all_data[all_data.month.str.contains('2015')]
  data_2016 = all_data[all_data.month.str.contains('2016')]
  data_2017 = all_data[all_data.month.str.contains('2017')]
  
  data = pd.concat([
    data_2013,
    data_2014,
    data_2015,
    data_2016,
    data_2017
  ])

  prices = preprocess_price(data["price"])
  features = preprocess_private_features(data)
  
  X = features.values
  Y = prices.values

  test_size = 0.20
  seed = 7

  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

  # Evaluate Algorithms
  num_folds = 10
  seed = 7
  # scoring = "neg_mean_squared_error"
  scoring = "neg_mean_absolute_error"

  # evaluate_linear_algorithms(num_folds=num_folds, seed=seed, scoring=scoring, X_train=X_train, Y_train=Y_train)
  # evaluate_ensemble_algorithms(num_folds=num_folds, seed=seed, scoring=scoring, X_train=X_train, Y_train=Y_train)

  # tune algorithms
  n_estimators = [100, 500, 1000, 2000, 3000]
  max_depth = [4,6,8,10]
  learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2]
  param_grid = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
  kfold = KFold(n_splits=num_folds, random_state=seed)
  
  # tune_gbm(
  #   param_grid=param_grid,
  #   seed=seed,
  #   scoring=scoring,
  #   kfold=kfold,
  #   X_train=X_train,
  #   Y_train=Y_train
  # )
  
  # tune_xgb(
  #   param_grid=param_grid,
  #   scoring=scoring,
  #   kfold=kfold,
  #   X_train=X_train,
  #   Y_train=Y_train
  # )

  # predict with best parameters
  # model = GradientBoostingRegressor(n_estimators=500, max_depth=6) 
  model = XGBRegressor(
    nthreads=-1, 
    n_estimators=4000, 
    max_depth=6,
    learning_rate=0.1
  ) 

  test_data = pd.read_csv(private_test_file)
  test_features = preprocess_private_features(test_data)
  test_features = test_features.values
  
  P_test_unscale = predict(model, 
    X_train, 
    Y_train, 
    X_test, 
    Y_test, 
    test_features)

  test_data["price"] = P_test_unscale
  test_data["price"] = test_data.price.astype(int)
  print(test_data["price"].head())

  test_data.to_csv('private_predicted.csv',index=False)

  # plot_importance(model)
  # plt.show() 


# explore_hdb()
# explore_private_housing()

predict_hdb()
predict_private()

