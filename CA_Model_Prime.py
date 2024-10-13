# Databricks notebook source
# MAGIC %md
# MAGIC #### Libraries and Utilities

# COMMAND ----------

# DBTITLE 1,Reduce Instance Of Deprecation Warnings
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# DBTITLE 1,Install Latest Version Available
#!pip install tensorflow
#!pip install sklearn-pandas

# COMMAND ----------

# DBTITLE 1,Select First Province & Cluster To Create Primary Weights For Other Provinces
# Select province and segment
province = 'Ontario'
segment = 'First'

# COMMAND ----------

# DBTITLE 1,Load All Libraries Required To Check Availability
# Libraries needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn_pandas import DataFrameMapper 
import os as os
import string
from joblib import dump, load
import time
import gc
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow import keras
from tensorflow.keras.losses import MSE
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import mae, mse, mape


# COMMAND ----------

# DBTITLE 1,Implement PySpark Supporting Functions
# MAGIC %run ./supportingfunctions/Supportingfunctions

# COMMAND ----------

# DBTITLE 1,Load Data From Snowflake Storage
## Table Name
CPO_POS_STORE_GROUPING = "CPO_POS_STORE_GROUPING"

# Read in library  
df = readPySparkdf(CPO_POS_STORE_GROUPING).toPandas()

## Removing the percentage sign '%'
df['INFLATION'] = df['INFLATION'].str.replace('%', '').astype(float)


# COMMAND ----------

# DBTITLE 1,Key Input Identifying Variables By Data Type
# Identify Independent Variables & Targets
# Choose Categorical Feature Vars
cat_vars = ['BRAND', 'KA', 'FSA']

# Choose Time Features Vars
time_vars = ['Time_Yr', 'Time_Week',  'Time_Mon']

# # Choose Continuous Feature Vars To Be Scaled
cont_vars = ['TOTAL_POPULATION','INFLATION']

# Choose Continuous Feature Vars To Be Logged
cont_vars_log = ['PRICE_PER_STICK']

# Choose Target Feature
target = ['TOTAL_STICK_SOLD']

# COMMAND ----------

# DBTITLE 1,Function To Select Province
def select_province(data, province):
  data = data[data['PROVINCE']==province]
  data.drop(columns=['PROVINCE'],axis=1,inplace=True)
  data.reset_index(drop=True, inplace=True)
  return data

# COMMAND ----------

# DBTITLE 1,Function To Select Cluster
#function to select clustered FSA segment: 'First', 'Second', 'Third'
def select_segment(data, segment):
  data = data[data['SEGMENT']==segment]
  data.drop(columns=['SEGMENT'],axis=1,inplace=True)
  data.reset_index(drop=True, inplace=True)
  return data

# COMMAND ----------

# DBTITLE 1,Function to convert dates to categorical variables
# Function to convert dates to categorical variables

def add_date_features(data,date,name):
    data[name + 'Yr'] = data[date].dt.year
    #data[name + 'Day'] = data[date].dt.dayofyear
    data[name + 'Week'] = data[date].dt.week
    data[name + 'Mon'] = data[date].dt.month 
    #data[name + 'Qtr'] = data[date].dt.quarter
    data.drop([date], axis = 1, inplace = True)
    data.reset_index(drop=True, inplace=True)
    return data

# COMMAND ----------

# DBTITLE 1,Reduce displayed decimals in continuous variables for readability
def rem_decimals(data, feat):
  data[feat] = data[feat].apply(lambda x: int(x))
  return data

# COMMAND ----------

# DBTITLE 1,Function to remove random sampling noise from any continuous variable
def denoise_x(x,threshold): 
  value=[]
  value.append(round(x[0],4))
  step_start = 0
  for i in range(1,len(x)):
    diff = round(abs((x[i] - x[i-1])),4)
    thresh = round(abs(x[i]*threshold),4)
    if diff < thresh:
      m = np.mean(value[step_start:i])
      value.append(round(m,4))      
    else:
      value.append(round(x[i],4))
      step_start = i
  return value

# COMMAND ----------

# DBTITLE 1,Function to denoise by column grouped by categorical variables
def denoise_column(data, colname, threshold, cat_vars=cat_vars):
  adjname = 'ADJ_' + colname
  values = data.groupby(cat_vars)[colname].apply(list).apply(lambda x: denoise_x(x, threshold)).reset_index(drop=True)
  value = [v for sublist in values for v in sublist]
  data[adjname]=value
  data.drop(colname, axis = 1, inplace = True)
  data = data.rename(columns={adjname:colname})
  return data

# COMMAND ----------

# DBTITLE 1,Reduce number of categorical combinations to a share of quantity sold
def limit_combi_share(data, threshold, yr):
  df_limit = data[data['Time_Yr'] == yr]
  df_limit = df_limit.groupby(cat_vars).agg({'PRICE_PER_STICK':'mean','TOTAL_STICK_SOLD':sum}).reset_index()
  df_limit['SALES'] = df_limit['PRICE_PER_STICK'] * df_limit['TOTAL_STICK_SOLD']
  total_sales = df_limit['SALES'].sum()
  df_limit['SALES_SHARE'] = df_limit['SALES']/total_sales
  df_limit.sort_values(['SALES_SHARE'] ,axis=0,ascending=True,inplace=True)
  df_limit.reset_index(drop=True,inplace=True)
  df_limit['cum_share'] = df_limit['SALES_SHARE'].cumsum()
  df_limit = df_limit[df_limit['cum_share'] >= threshold]
  return df_limit

# COMMAND ----------

# DBTITLE 1,Call Functions To Prepare Data
# Select Province
data = select_province(df,province) #only use df once to preserve original data in notebook

# Select Cluster Segment From 'First','Second','Third'
data = select_segment(data, segment)

# Select Date Features
data = add_date_features(data,'WEEK_END_DATE','Time_')

# Truncate decimals
data = rem_decimals(data,'TOTAL_POPULATION')
data = rem_decimals(data,'TOTAL_STICK_SOLD')

# Append features including target
features = cat_vars + time_vars + cont_vars + cont_vars_log + target

# Remove incomplete last years
data = data[data['Time_Yr'] < 2021]

# Sort For simplicity
data.sort_values(['BRAND','KA', 'FSA', 'Time_Yr', 'Time_Mon', 'Time_Week'] ,axis=0,ascending=True,inplace=True)
data.reset_index(drop=True,inplace=True)

# Create Input Data
data = data[features]
data.reset_index(drop=True,inplace=True)

# Denoise continuous variables
# Denoise price per stick
colname='PRICE_PER_STICK'
data = denoise_column(data, colname, threshold = .01, cat_vars=cat_vars)
# Denoise total stick sold
colname='TOTAL_STICK_SOLD'
data = denoise_column(data, colname, threshold = .01, cat_vars=cat_vars)

## Filter the combination which are less than a threshold
data = data.merge(limit_combi_share(data, 0.05, 2020), left_on = cat_vars, right_on=cat_vars, how = 'right')
data.drop(['PRICE_PER_STICK_y','TOTAL_STICK_SOLD_y','SALES','SALES_SHARE','cum_share'],axis=1,inplace=True)
data.rename(columns={'PRICE_PER_STICK_x':'PRICE_PER_STICK', 'TOTAL_STICK_SOLD_x':'TOTAL_STICK_SOLD'},inplace=True)

# Final sort & index
data.sort_values(['BRAND','KA', 'FSA', 'Time_Yr', 'Time_Mon', 'Time_Week'] ,axis=0,ascending=True,inplace=True)
data.reset_index(drop=True,inplace=True)

# data.head()


# COMMAND ----------

# DBTITLE 1,Take log on continuous features to be logged with +1 to filter negative logs
def log_cont_data(data, cont_vars_log=cont_vars_log):   
  # take logs of each variable to be logged identified
  for feat in cont_vars_log:
    data[feat] = data[feat].apply(lambda x: np.log1p(x).astype(np.float32))
    data = data[cont_vars_log]
  # return data with logged columns
  data.reset_index(drop=True, inplace=True)
  return data

# COMMAND ----------

# DBTITLE 1,Take log of target to compress range
#  Take log on target to reduce range
def log_target(data, target=target):
  # take log of target
  data[target] = data[target].apply(lambda x: np.log1p(x).astype(np.float32))
  # return data 
  return data

# COMMAND ----------

# DBTITLE 1,Function to apply embeddings for categorical features in data
def cat_map_data(data,cat_vars=cat_vars, emax=50, emin=4):
    # compute list of number of unique categories for each categorical variable
    cat_emb = [len(data[c].unique()) for c in cat_vars]
    
    # compute list inserting maximum number of embeddings for each category 
    cat_emb_max = [c if c<= emax else emax for c in cat_emb] #maximum embedded weights is emax (default=50)
    
    # compute list inserting minimum number of embeddings for each category
    cat_emb_max = [c if c>= emin else emin for c in cat_emb_max] #minimum embedded weights is emin (default = 4)
    
    # form dictionary of the categorical variables and the list of embeddings
    cat_vars_dict = dict(zip(cat_vars,cat_emb_max))
    
    # form list of tuples of categorical variables and the label encoder
    cat_map = [(c,LabelEncoder()) for c in cat_vars]
    
    # return the embedding dictionary and the map of label encoders to categorical variables
    return cat_vars_dict,cat_map

# COMMAND ----------

# DBTITLE 1,Function to apply embeddings for time features in data
# Function to apply embeddings for time features in data

def time_map_data(data,time_vars=time_vars, tmax=12, tmin=3):
    
    # compute number of unique values for each time variable
    time_emb = [len(data[t].unique()) for t in time_vars]
    
    # insert maximum embedded coefficients for time variables
    time_emb_max = [t if t <= tmax else tmax for t in time_emb] #maximum embedded weights is tmax (default=12)
    
    # insert minimum embedded coefficients for time variables
    time_emb_max = [t if t >= tmin else tmin for t in time_emb_max]#minimum embedded weights is tmin (default=3)
    time_vars_dict = dict(zip(time_vars,time_emb_max))
    
    # compute list of tuples assigning the Label Encoder to the time variable
    time_map = [(t,LabelEncoder()) for t in time_vars]
    
    # return dictionary of embedded coefficients and list of encoder tuples
    return time_vars_dict,time_map

# COMMAND ----------

# DBTITLE 1,Function to apply scaler on the continuous features in data

# s can be standardscaler,robustscaler or minmaxscaler; default is minmax
# x,y is limit on minmax; default to 0,1
# l,u is percential rank for the robust scaler based on median; default is 10,90

def cont_map_data(cont_vars=cont_vars, s='minmax', x=1, y=2, l=10, u=90): # s can be standardscaler,robustscaler or minmaxscaler
    # select scaler map and form list of tuples for variable and scaler
  if s == 'standard':
      cont_map = [([c],StandardScaler(copy=True,with_mean=True,with_std=True)) for c in cont_vars]

  elif s == 'robust':
      cont_map = [([c],RobustScaler(with_centering=True,with_scaling=True,quantile_range=(t,u))) for c in cont_vars]

  elif s == 'minmax':
      cont_map = [([c],MinMaxScaler(feature_range = (x,y))) for c in cont_vars]

  # return map of scaler and continuous variables tuples
  return cont_map

# COMMAND ----------

# DBTITLE 1,Scale continuous features and fit with DataFrameMapper
# 
def map_cont_data(data):
  
  # map scaler to continuous data;  cont_map_data defaults to cont_vars and minmax scaler with x = 1,y = 2
  cont_map = cont_map_data()
  
  # intialize DataFrameMapper with scalers to be applied
  cont_mapper = DataFrameMapper(cont_map)

  # fit mapper to data; cont_map_fit is required when building input layers
  cont_map_fit = cont_mapper.fit(data)

  # save scaler for prediction algorithm
  filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'cont_scaler'
  dump(cont_map,filename)
  
  # transform and return data
  cont_data = cont_map_fit.transform(data).astype(np.float32)

  return cont_data, cont_map_fit

# COMMAND ----------

# DBTITLE 1,Encode categorical features and fit - transform with DataFrameMapper

def map_cat_data(data):
    
    # map encoder to categorical variables; cat_vars_dict required for input layers
    cat_vars_dict,cat_map = cat_map_data(data)
    
    # save cat map for predictor
    filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'cat_map'
    dump(cat_map,filename)

    # initialize dataframe mapper
    cat_mapper = DataFrameMapper(cat_map)
    
    # fit categorical variables; cat_map_fit required for input layer
    cat_map_fit = cat_mapper.fit(data)
    
    # transform data using dataframe mapper
    cat_data = cat_map_fit.transform(data).astype(np.int64) 
    
    # return cat_vars_dict and cat_map_fit used in embedding input layer to tensorflow
    return  cat_data, cat_vars_dict, cat_map_fit 

# COMMAND ----------

# DBTITLE 1,Encode time features and fit - transform with DataFrameMapper
#     
def map_time_data(data):    
    
  # map encoder to time variables; time_vars_dict used in embedding input layer to tensorflow
  time_vars_dict,time_map = time_map_data(data)

  # save time map for predictor
  filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'time_map'
  dump(time_map,filename)

  # initialize dataframemapper
  time_mapper = DataFrameMapper(time_map)

  # fit time variables to data; time_map_fit req'd for input layers
  time_map_fit = time_mapper.fit(data)

  # transform data
  time_data = time_map_fit.transform(data).astype(np.int64)

  # return encoded time data, time_vars_dict and time_map_fit used in input layer
  return time_data, time_vars_dict, time_map_fit 

# COMMAND ----------

# DBTITLE 1,Scale continuous features and fit with DataFrameMapper
 
def map_cont_data(data):
  
  # map scaler to continuous data;  cont_map_data defaults to cont_vars and minmax scaler with x = 1,y = 2
  cont_map = cont_map_data()
  
  # intialize DataFrameMapper with scalers to be applied
  cont_mapper = DataFrameMapper(cont_map)

  # fit mapper to data; cont_map_fit is required when building input layers
  cont_map_fit = cont_mapper.fit(data)

  # save scaler for prediction algorithm
  filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'cont_scaler'
  dump(cont_map,filename)
  
  # transform and return data
  cont_data = cont_map_fit.transform(data).astype(np.float32)

  return cont_data, cont_map_fit

# COMMAND ----------

# DBTITLE 1,Function to scale and shape continuous variable target tensor
# 
def map_cont_target(data, target, s='minmax', f=1, c=2, l=10, u=90):     

    # set series to single vector
    y = data[target].reshape(-1,1)
    
    # select scaler    
    if s == 'standard':
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

        # fit scaler to column
        scaled = scaler.fit(y)

        # save scaler to quantity_maps subdirectory  
        filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'target_cont_scaler' 
        dump(scaled,filename)

        # transform target
        y_scaled = scaled.transform(y)
    
    elif s == 'robust':
        scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(l,u))

        # fit scaler to target
        scaled = scaler.fit(y)

        # save scaler to quantity_maps subdirectory   
        filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'target_cont_scaler' 
        #filename = os.path.join('quantity_maps','target_cont_scaler')
        dump(scaled,filename)

        # transform target
        y_scaled = scaled.transform(y)

    # note:applying minmax scaler to log removes need for leakyRelu in model
    elif s == 'minmax':
        scaler = MinMaxScaler(feature_range=(f,c))

        # fit scaler to column
        scaled = scaler.fit(y)

        # save scaler to quantity_maps subdirectory
        filename = '/dbfs/CA_Predictor/' + 'prime_maps/' + 'target_cont_scaler'
        dump(scaled,filename)

        # transform target
        y_scaled = scaled.transform(y)
    
    # shape and type y array for tensorflow
    data[feat] = y_scaled.astype(np.float32) 
    
    return data


# COMMAND ----------

# DBTITLE 1,Compute transformed and shaped input data

# scale and encode data 

# # encode categorical variables
data_cat, cat_vars_dict, cat_map_fit = map_cat_data(data)

# encode time variables
data_time, time_vars_dict, time_map_fit = map_time_data(data)

# scale selected continuous variables
data_cont, cont_map_fit = map_cont_data(data)

# log selected continuous variables
data_log = log_cont_data(data)

# # scale or log target
data_target = log_target(data[target]) #log target data
# data_target = map_cont_target(data[target]) #scale target data


# COMMAND ----------

# DBTITLE 1,Compute sample weights to balance wide distributions of quantity
# colname is column name string for column setting weights
# setting hi and lo will adjust relative weights of the sample weighs, i.e., if greater weight on smaller samples required increase lo and hi 

def compute_sample_weights(colname,data=data,features=features,hi=1.2,lo=.8):
  # set raw weights as counts of by colname
  weights = pd.DataFrame(data[colname].value_counts(normalize=True,ascending=False))
  weights.reset_index(inplace=True)
  
  # add column calculating share by label in colname
  weights.columns = [colname,'share']
  
  # scale weights to equalize impact of large volume labels with low volume labels
  # set range limits for sample weights in lo and hi
  scaler = MinMaxScaler(feature_range = (lo,hi))
  weights[['share']] = scaler.fit_transform(weights[['share']])
  
  # convert colname to list for transforming into array for tensorflow
  sample_weight = list(weights['share'])
  
  # reverse weights list to reduce importance of large volume labels and increase importanace of low volume labels
  sample_weight.reverse()
  
  # compute dictionary with label indexed to reversed sample weights
  sample_weight_dict = dict(zip(weights[colname],sample_weight))
  
  # compute array for a sample weight for each sample in data
  weight_arr = np.array([sample_weight_dict[x] for x in data[colname]]).reshape(-1,1)
  
  return weight_arr, sample_weight_dict

# compute sample weights
samp_weights, sample_weight_dict = compute_sample_weights('BRAND')

# concatenate encoded, scaled, logged data and target plus sample weights
data_scaled = np.hstack([data_cat, data_time, data_cont, data_log, samp_weights, data_target])


# COMMAND ----------

# DBTITLE 1,Stratify and perform train validate test split & shape for tensor

# Compute target, data, stratification array and percentile boundaries
y = data_scaled[:,-1].reshape(-1,)
X = data_scaled[:,:-1]

# Compute stratifications
strats, percentiles = pd.qcut(y.reshape(-1,),4,labels=False,retbins=True)
y = np.column_stack((y,strats))

# Perform first split to obtain stratified all and test set 
X_all,X_test,y_all,y_test = train_test_split(X,y,test_size=.05,random_state=75,stratify=strats)

# Redefine strats to be stratification in remaining target vector y_all; remove strats from y_all
strats = y_all[:,1]
y_all = y_all[:,0]

# Remove strats from y_test
y_test = y_test[:,0]

# # Perform second split to obtain stratified train and validation sets 
X_train,X_val,y_train,y_val = train_test_split(X_all,y_all,test_size=.15,random_state=75,stratify=strats)

# # Extract sample weights from X
test_weights = X_test[:,-1] 
train_weights = X_train[:,-1]
val_weights = X_val[:,-1]
all_weights = X_all[:,-1]

# Remove sample weights from data
X_train = X_train[:,:-1]
X_all = X_all[:,:-1]
X_val = X_val[:,:-1]
X_test = X_test[:,:-1]

# Convert train and test input data to list of arrays for tensorflow
X_tr = np.hsplit(X_train, X_train.shape[1])
X_va = np.hsplit(X_val, X_val.shape[1])
X_te = np.hsplit(X_test, X_test.shape[1])
X_al = np.hsplit(X_all, X_all.shape[1])

# Shape targets
y_tr = np.array(y_train).reshape(-1,1)
y_va = np.array(y_val).reshape(-1,1)
y_te =  np.array(y_test).reshape(-1,1)
y_al = np.array(y_all).reshape(-1,1)



# COMMAND ----------

# DBTITLE 1,Form categorical variable input layer with linear first stage embedding

# Apply batch normalization for both regularization and controlling gradients
# Apply dropout to control overfitting and exploding/vanishing gradients

def cat_input(feat,cat_vars_dict,r=.5):
    # compute input vector
    name = feat[0]
    c1 = len(feat[1].classes_)
    c2 = cat_vars_dict[name]

    # create input layer
    inp = Input(shape=(1,),dtype='int64',name=name + '_in')
    cat = Flatten(name=name+'_flt')(Embedding(c1,c2,input_length=1)(inp))
    
    # add dense layers, dropout, and batch normalization
    cat = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cat)
    cat = Dropout(rate=r)(cat)
    cat = BatchNormalization()(cat)
    cat = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cat)
    cat = Dropout(rate=r)(cat)
    cat = BatchNormalization()(cat)
    return inp,cat

# Graph categorical features input
cats = [cat_input(feat,cat_vars_dict) for feat in cat_map_fit.features]

# COMMAND ----------

# DBTITLE 1,Form time variable input layer with linear first stage embedding


def time_input(feat,time_vars_dict,r=.5):
    
    # compute input vector
    name = feat[0]
    c1 = len(feat[1].classes_)
    c2 = time_vars_dict[name]

    # create input layer
    inp = Input(shape=(1,),dtype='int64',name=name + '_in')
    time = Flatten(name=name+'_flt')(Embedding(c1,c2,input_length=1)(inp))
    
    # add dense, dropout and normalization layers
    time = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(time)
    time = Dropout(rate=r)(time)
    time = BatchNormalization()(time)
    time = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(time)
    time = Dropout(rate=r)(time)
    time = BatchNormalization()(time)
    return inp,time

# Graph time features input
times = [time_input(feat,time_vars_dict) for feat in time_map_fit.features]

# COMMAND ----------

# DBTITLE 1,Form continuous variable input layer with linear first stage embedding

def cont_input(feat,r=.5):
    name = feat[0][0]
    
    # create input layer
    inp = Input((1,), name=name+'_in')
    cont = Dense(1, name = name + '_d')(inp)

    # add dense, dropout, batch normalization layers
    cont = Dense(1000, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    cont = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    return inp,cont

# Graph continuous features input
conts = [cont_input(feat) for feat in cont_map_fit.features] 

# COMMAND ----------

# DBTITLE 1,Form continuous logged variable input layer with linear first stage embedding
def log_cont_input(feat,r=.5):
    name = str(feat)
    
    # create input layer
    inp = Input((1,), name=name+'_in')
    cont = Dense(1, name = name + '_d')(inp)

    # add dense, dropout, batch normalization layers
    cont = Dense(1000, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    cont = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    return inp,cont

# Graph continuous features input
log_conts = [log_cont_input(feat) for feat in data_log.columns] 

# COMMAND ----------

# DBTITLE 1,For interconnected layers among and between variables
def build_quantity_model(cats, times, conts, log_conts, r = .5):

    # Build graph for interconnected categorical features
    
    # input concatenated categorical features for interconnected nodes
    c = concatenate([cat for inp,cat in cats], axis=1)
    
    # add dense, dropout and batch normalization layers
    c = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(c)
    c = Dropout(rate=r)(c)
    c = BatchNormalization()(c)
    c = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(c)
    c = Dropout(rate=r)(c)
    c = BatchNormalization()(c)
    
    # Build graph for concatenated time features for interconnected nodes
    
    # concatenate time variables for inteconnected nodes
    t = concatenate([time for inp,time in times], axis = 1)
    
    # add dense, dropout and batch normalization layers 
    t = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(t)
    t = Dropout(rate=r)(t)
    t = BatchNormalization()(t)
    t = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(t)
    t = Dropout(rate=r)(t)
    t = BatchNormalization()(t)

    # Build graph for continuous features fully interconnected by concatenate
    
    # concatenate if 2 or more variables
    f = concatenate([cont for inp,cont in conts], axis=1)
    
    # add dense, dropout and batch normalization layers fully interconnected
    f = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(f)
    f = Dropout(rate=r)(f)
    f = BatchNormalization()(f)
    f = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(f)
    f = Dropout(rate=r)(f)
    f = BatchNormalization()(f)
    
     # Build graph for logged continuous features fully interconnected by concatenate
    
    # concatenate if 2 or more variables
    l = concatenate([cont for inp,cont in log_conts], axis = 1)

    # add dense, dropout and batch normalization layers fully interconnected
    l = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(l)
    l = Dropout(rate=r)(l)
    l = BatchNormalization()(l)
    l = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(l)
    l = Dropout(rate=r)(l)
    l = BatchNormalization()(l)
    
    # Concatenate categorical, time, continuous and continuous logged features
    x = concatenate([c,t,f,l], axis=1)

    # add fully interconnected dense, dropout and normalization layers
    x = Dense(500,activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x)
    x = Dropout(rate=r)(x)
    x = BatchNormalization()(x)
    x = Dense(500,activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x)
    x = Dropout(rate=r)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x) 
    
    # add input layer for the model
    model = Model([inp for inp,cat in cats] + [inp for inp,time in times] + [inp for inp,cont in conts] + [inp for inp,cont in log_conts], x)
    
    # set learning rate
    #adam = keras.optimizers.Adam(learning_rate=.001)
    
    #  compile with optimizer, loss and metrics
    model.compile(optimizer='Adam', loss = MSE, metrics = ['mae','mse','mape'])
    
    return model

# COMMAND ----------

# DBTITLE 1,Call formation of the basic quantity model
quantity_model = build_quantity_model(cats, times, conts, log_conts)

# COMMAND ----------

# DBTITLE 1,Initialize callbacks for validation
es = EarlyStopping(monitor='loss', patience=5, verbose=0,
    mode='min', min_delta=.015, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='loss', factor=0.9,
                              mode = 'min', min_delta=.01, patience=3, min_lr=0.0005)

filepath = '/dbfs/CA_Predictor/' + 'prime_val_checkpoint/' + 'val_checkpoint'
mckp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch')

# COMMAND ----------

# #load weights from the prior model
# # load weights from previous model
# #filepath = '/dbfs/CA_Predictor/' + 'prime_val_model/' + 'val_weights'
# filepath = '/dbfs/Saksham/CA_Predictor/prime_val_model/val_weights'
# quantity_model.load_weights(filepath)

# COMMAND ----------

# DBTITLE 1,Perform validation model
# Train the model using early stopping and reducing learning rates to reduce overfitting and 'memorization'.

quantity_model.fit(X_tr,y_tr,
            batch_size=512,
            epochs=20,
            verbose=True, sample_weight=train_weights,
            validation_data = (X_va, y_va),callbacks=[es,rlr,mckp])

# COMMAND ----------

# DBTITLE 1,Call epoch history in validation model
# call history
dh = pd.DataFrame(data=quantity_model.history.history)

# COMMAND ----------

# DBTITLE 1,Interpret initial epoch progress
dh.head()

# COMMAND ----------

# DBTITLE 1,Interpret last epoch performance
dh.tail()

# COMMAND ----------

# DBTITLE 1,Compare training and validation losses
dh[['loss','val_loss']].plot()

# COMMAND ----------

# DBTITLE 1,Compare absolute error between training and validation set
dh[['mae','val_mae']].plot()

# COMMAND ----------

# DBTITLE 1,Compare squared error between training and validation epochs
dh[['mse','val_mse']].plot()

# COMMAND ----------


# Confirm performance using test data

# make prediction in scaled log of quantity 
y_pred_test = quantity_model.predict(X_te)

# compute array of differences between predictions and actuals
y_diff_test = abs(y_te - y_pred_test)

# compute percentage errors on y_test
y_percent_error_test = 100 * y_diff_test/y_te


# COMMAND ----------

# compute mean of percent error
y_percent_error_test.mean()

# COMMAND ----------

# compute deviation of percent error
y_percent_error_test.std()

# COMMAND ----------

# show scatter plot of error
plt.scatter(y_te,y_pred_test)
plt.plot(y_te,y_te,'r')

# COMMAND ----------

# ## Saving the validation model

# from tensorflow.keras.applications import InceptionV3
# quantity_model.save('/tmp/validation_quantity_model.h5')

# ## copy data to DBFS as dbfs:/tmp/model-full.h5 and check it:
# dbutils.fs.cp("file:/tmp/validation_quantity_model.h5", "dbfs:/CA_Predictor/prime_val_model/val_model/quantity_model.h5")
# #display(dbutils.fs.ls("file:/tmp/quantity_model.h5"))

# COMMAND ----------

# DBTITLE 1,Set callbacks for full model training
# Train the predictor using all of the data set except the reserved test set for final evaluation

# initiate callbacks

es = EarlyStopping(monitor='loss', patience=3, verbose=0,
    mode='min', min_delta=.001, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='loss', factor=0.95,
                              mode = 'min', min_delta=.0005, patience=2, min_lr=0.0005)

filepath = '/dbfs/CA_Predictor/' + 'prime_all_checkpoint/' + 'all_checkpoint'
mckp = ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch')

# COMMAND ----------

# DBTITLE 1,Load validation weights for transfer learning
#load weights from the prior model
# load weights from previous model
#filepath = '/dbfs/CA_Predictor/' + 'prime_val_model/' + 'val_weights'
filepath = '/dbfs/CA_Predictor/prime_all_model/all_weights'
quantity_model.load_weights(filepath)

# COMMAND ----------

# DBTITLE 1,Form training epochs on final fully weighted data
# Additional training on merged train and validation set with epochs limited to the min loss epoch on validation set
# when order_model is fit the second time, even with changed data, it loads the final weights from the first fit automatically
# clear_session does not reset the weights in the model and the learning rate starts at the last rlr learning rate

quantity_model.fit(X_al,y_al,
            batch_size=512,
            epochs=10,
            verbose=True,
            validation_data = None, 
            sample_weight=all_weights,
            callbacks=[es,rlr,mckp])


# COMMAND ----------

# DBTITLE 1,Save coefficient weights for predictor model
# save coefficient weights
filepath = '/dbfs/CA_Predictor/' + 'prime_all_model/' + 'all_weights'
quantity_model.save_weights(filepath, overwrite=True, save_format=None, options=None)

# COMMAND ----------

# DBTITLE 1,Reload epoch history to dataframe

da = pd.DataFrame(data=quantity_model.history.history)


# COMMAND ----------

# DBTITLE 1,Review first five epochs
da.head()

# COMMAND ----------

# DBTITLE 1,Review last five epochs
da.tail()

# COMMAND ----------

# DBTITLE 1,Confirm performance using test data

# make prediction in scaled log of quantity 
y_pred_test = quantity_model.predict(X_te)

# compute array of differences between predictions and actuals
y_diff_test = y_te - y_pred_test

# compute percentage errors on y_test
y_percent_error_test = 100 * y_diff_test/y_te

# COMMAND ----------

# DBTITLE 1,Compute mean of test percent error
# 
y_percent_error_test.mean()

# COMMAND ----------

# DBTITLE 1,Compute deviation of percent error
 
y_percent_error_test.std()

# COMMAND ----------

# DBTITLE 1,Show scatter plot of error

plt.scatter(y_te,y_pred_test)
plt.plot(y_te,y_te,'r')
