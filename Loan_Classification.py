#!/usr/bin/env python
# coding: utf-8

# ## Purpose: Predict the probability of a loan classification.  
# ### The output is a normalized probability for each class.  

# ### This model is designed to work with sparse data and imbalanced classes.  This code requires a predictor for production.  

# ### Libraries and Utilities

# In[6]:


#%reload_ext autoreload
#%autoreload


# In[7]:


#!mkdir loan_checkpoint
#!mkdir loan_all_checkpoint
#!mkdir loan_all_model
#!mkdir loan_maps
#!mkdir loan_val_model


# In[8]:


import warnings
warnings.filterwarnings('ignore')


# In[9]:


#allow cell to perform multiple computations

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


# Python libraries needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn_pandas import DataFrameMapper 
import os as os
import string
from joblib import dump, load
import time
import gc


# In[ ]:


# Import Tensorflow & libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.metrics import Precision, Recall, binary_accuracy, AUC
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# ### Load Data

# In[ ]:


# Import Data
dx = load()


# ## Build Loan Classificaton Predictor
# #### Encode, scale and shape data for tensorflow
# #### Perform train test split

# In[ ]:


# Choose categorical feature vars
cat_vars = ['','']

# Choose time features vars
time_vars = ['','']

# Choose continuous feature vars
cont_vars = ['','']

# Choose target features .. can be multiple
target = ['']

# Identify data and target dataframes
features = cat_vars + time_vars + cont_vars + target
data = dx[features]


# ## Apply encoders, embeddings, max/min embeddings and scalers to feature data

# In[ ]:


# Function to convert dates to categorical variables
def add_date_features(data,date,name):   # date is column name .. name is prefix for time feature column name e.g., "Time_"
    data[name + 'Yr'] = data[date].dt.year
    #data[name + 'Day'] = data[date].dt.dayofyear
    #data[name + 'Week'] = data[date].dt.week
    #data[name + 'Mon'] = data[date].dt.month 
    data[name + 'Qtr'] = data[date].dt.quarter
    data.drop([date], axis = 1, inplace = True)
    data.reset_index(drop=True, inplace=True)
    return data


# ### Compute lists of embedding sizes as tuples of label encoders and scale continuous features

# In[ ]:


# Function to apply embeddings for categorical features in data

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


# In[ ]:


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


# In[ ]:


# Function to apply scaler on the continuous features in data

# s can be standardscaler,robustscaler or minmaxscaler; default is minmax
# x,y is limit on minmax; default to 1,3
# l,u is percential rank for the robust scaler based on median; default is 10,90
# robust scaler removes outliers before applying standard scaler on median value USE WITH CAUTION

def cont_map_data(cont_vars=cont_vars, s='minmax', x=1, y=3, l=10, u=90): # s can be standardscaler,robustscaler or minmaxscaler

    # select scaler map and form list of tuples vor variable and scaler
    if s == 'standard':
        cont_map = [([c],StandardScaler(copy=True,with_mean=True,with_std=True)) for c in cont_vars]

    elif s == 'robust':
        cont_map = [([c],RobustScaler(with_centering=True,with_scaling=True,quantile_range=(t,u))) for c in cont_vars]

    elif s == 'minmax':
        cont_map = [([c],MinMaxScaler(feature_range = (x,y))) for c in cont_vars]

    # return map of scaler and continuous variables tuples
    return cont_map


# ### Transform feature variables and convert to array using DataFrameMapper

# In[ ]:


# Encode categorical features amd fit - transform with DataFrameMapper
def map_cat_data(data):

    # map encoder to categorical variables; cat_vars_dict required for input layers
    cat_vars_dict,cat_map = cat_map_data(data)

    # initialize dataframe mapper
    cat_mapper = DataFrameMapper(cat_map)

    # fit categorical variables; cat_map_fit required for input layer
    cat_map_fit = cat_mapper.fit(data)

    # transform data using dataframe mapper
    cat_data = cat_map_fit.transform(data).astype(np.int64) 

    # return cat_vars_dict and cat_map_fit used in embedding input layer to tensorflow
    return  cat_data, cat_vars_dict, cat_map_fit 


# In[ ]:


# Encode time features and fit - transform with DataFrameMapper    
def map_time_data(data):    

    # map encoder to time variables; time_vars_dict used in embedding input layer to tensorflow
    time_vars_dict,time_map = time_map_data(data)

    # initialize dataframemapper
    time_mapper = DataFrameMapper(time_map)

    # fit time variables to data; time_map_fit req'd for input layers
    time_map_fit = time_mapper.fit(data)

    # transform data
    time_data = time_map_fit.transform(data).astype(np.int64)

    # return encoded time data, time_vars_dict and time_map_fit used in input layer
    return time_data, time_vars_dict, time_map_fit 


# In[ ]:


# Scale continuous features and fit with DataFrameMapper
def map_cont_data(data):

    # map scaler to continuous data;  cont_map_data defaults to data, cont_vars and minmax scaler, x = 1,y = 3
    cont_map = cont_map_data() 

    # intialize DataFrameMapper with scalers to be applied
    cont_mapper = DataFrameMapper(cont_map)

    # fit mapper to data; cont_map_fit is required when building input layers
    cont_map_fit = cont_mapper.fit(data)

    # transform and return data
    cont_data = cont_map_fit.transform(data).astype(np.float32)

    return cont_data, cont_map_fit


# #### Encode categorical target
# ##### Note that LabelEncoder will first order ascending numeric followed by ascending alphabetical.  Consequently, the encoder will assign 0 to the lowest number and then sequence 1,2,3.. to the next higher numbers and into alphabetical.  Examine class names for consistent ordering, i.e., all should begin alpha or num

# In[ ]:


# Function to encode and shape categorical target tensor
def map_cat_target(target):   
    #target is dataframe series of shape(-1,)      
    # map label array
    y = np.array(target).reshape(-1,1)
    encoder = LabelEncoder()
    y_mapped = encoder.fit_transform(y)

    # shape y array
    y_mapped = np.array(y_mapped).reshape(-1,1).astype(np.int64)  #tensorflow requires int64 in array
    return y_mapped


# ### Compute transformed and shaped input data

# In[ ]:


# Compute input layer functions
# scale and encode data
data_cat, cat_vars_dict, cat_map_fit = map_cat_data(data)
data_time, time_vars_dict, time_map_fit = map_time_data(data)
data_cont, cont_map_fit = map_cont_data(data)

# concatenate scaled and encoded data
data_scaled = np.hstack([data_cat,data_time,data_cont])

# scale or encode target data
y_scaled = map_cat_target(target)


# ### Split data into train, validate and test sets.  Shape into list of arrays for tensorflow.

# In[ ]:


# Apply stratified split twice on target for validation and test to balance split sizes by class counts
# Note: splitting in tensorflow.keras.model.fit is not able to stratify by labels
# Test sample is 5% of total sample and validation sample is 20% of remainder 

# first stratify and split for test data on stratify target 'y_scaled'
data_,data_test,y_,y_test = train_test_split(data_scaled,y_scaled,test_size=.05,random_state=75,stratify=y_scaled)

# resize target vector for second stratification for validation data
temp = y_.reshape(-1,1)

# second stratify and split for validation data on resized target data
data_train,data_val,y_train,y_val = train_test_split(data_,y_,test_size=.20,random_state=75,stratify=temp)

# Convert train and test input data to list of arrays for tensorflow
X_train = np.hsplit(data_train, data_train.shape[1])
X_val = np.hsplit(data_val, data_val.shape[1])
X_test = np.hsplit(data_test, data_test.shape[1])
X_all = np.hsplit(data_scaled, data_scaled.shape[1])

# Convert y_train, y_val, y_test and y_all to reshaped vector arrays

# win targets
y_train = np.array(y_train).reshape(-1,1)
y_val = np.array(y_val).reshape(-1,1)
y_test =  np.array(y_test).reshape(-1,1)
y_all = y_scaled.reshape(-1,1)


# In[ ]:


# Compute class weights for tensorflow fitting method
def category_class_weight(y):   #target is encoded category labels
    #compute normal weight for each category
    weights = pd.DataFrame(y.value_counts(normalize=True,ascending=False))
    weights.reset_index(inplace=True)
    weights.columns = ['class','frequency']

    #compute frequency list
    frequency_list = weights['frequency'].tolist()
    frequency_list_reversed = frequency_list.reverse()

    #compute class weight dictionary
    class_weight_dict = dict(zip(weights['class'],frequency_list_reversed))

    return class_weight_dict

train_weight = np.array([category_class_weight_dict[x] for x in y_train]).reshape(-1,1)
all_weight = np.array([category_class_weight_dict[x] for x in y_all]).reshape(-1,1)


# ## Build Loan Classification Model
# #### Plot model (if supported)

# In[ ]:


# Import required libraries

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, Input
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.constraints import max_norm, unit_norm
from tensorflow.keras.metrics import Precision, Recall, accuracy, AUC
from tensorflow.keras.losses import SparseCategoricalCrossentropy  #allows label encoded classes


# In[ ]:


# Graph categorical features input layers
# Initialize with Xavier by calling 'glorot_normal' to minimize vanishing/exploding gradients
# Apply batch normalization for both regularization and controlling gradients
# Apply dropout to control overfitting

# this builds input layer and subsequent layers in linear (not interconnected) configuration
def cat_input(feat,cat_vars_dict,r=.5):
    # compute input vector
    name = feat[0]
    c1 = len(feat[1].classes_)
    c2 = cat_vars_dict[name]    
    # create input layer
    inp = Input(shape=(1,),dtype='int64',name=name + '_in')
    cat = Flatten(name=name+'_flt')(Embedding(c1,c2,input_length=1)(inp))
    cat = Dense(1000, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cat)
    cat = Dropout(rate=r)(cat)
    cat = BatchNormalization()(cat)
    cat = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cat)
    cat = Dropout(rate=r)(cat)
    cat = BatchNormalization()(cat)
    return inp,cat

# Graph categorical features input
cats = [cat_input(feat,cat_vars_dict) for feat in cat_map_fit.features]


# In[ ]:


# Graph time input layers
# Initialize with Xavier by calling 'glorot_normal' to minimize vanishing/exploding gradients
# Apply batch normalization for both regularization and controlling gradients
# Apply dropout to control overfitting

# this builds time input layers followed by linear layers (not interconnected) for the time variable (not interconnected)
def time_input(feat,time_vars_dict,r=.5):
    # compute input vector
    name = feat[0]
    c1 = len(feat[1].classes_)
    c2 = time_vars_dict[name]

    # create input layer
    inp = Input(shape=(1,),dtype='int64',name=name + '_in')
    time = Flatten(name=name+'_flt')(Embedding(c1,c2,input_length=1)(inp))
    time = Dense(1000, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(time)
    time = Dropout(rate=r)(time)
    time = BatchNormalization()(time)
    time = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(time)
    time = Dropout(rate=r)(time)
    time = BatchNormalization()(time)
    return inp,time

# Graph time features input
times = [time_input(feat,time_vars_dict) for feat in time_map_fit.features]


# In[ ]:


# Graph input layers for continuous features
# Initialize with Xavier by calling 'glorot_normal' to minimize vanishing/exploding gradients
# Apply batch normalization for both regularization and controlling gradients
# Apply dropout to control overfitting

# this builds input layer for continuous features and subsequent layers in linear configuration (not interconnected)
def cont_input(feat,r=.5):
    name = feat[0][0]
    inp = Input((1,), name=name+'_in')
    cont = Dense(1, name = name + '_d')(inp)
    cont = Dense(1000, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    cont = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(cont)
    cont = Dropout(rate=r)(cont)
    cont = BatchNormalization()(cont)
    return inp,cont

# Graph continuous features input
conts = [cont_input(feat) for feat in cont_map_fit.features]


# In[ ]:


def build_loan_model(cats,times,conts,r=.5):

    # Build graph for categorical features such that all categorical features are interconnected
    c = concatenate([cat for inp,cat in cats])

    # add linear layers
    c = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(c)
    c = Dropout(rate=r)(c)
    c = BatchNormalization()(c)
    c = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(c)
    c = Dropout(rate=r)(c)
    c = BatchNormalization()(c)

    # Build graph for time features such that all time features are interconnected
    t = concatenate([time for inp,time in times])

    # add linear layers
    t = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(t)
    t = Dropout(rate=r)(t)
    t = BatchNormalization()(t)
    t = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(t)
    t = Dropout(rate=r)(t)
    t = BatchNormalization()(t)


    # Build graph for continuous features such that all continuous features are interconnected
    f = concatenate([cont for inp,cont in conts])

    # add linear layers
    f = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(f)
    f = Dropout(rate=r)(f)
    f = BatchNormalization()(f)
    f = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(f)
    f = Dropout(rate=r)(f)
    f = BatchNormalization()(f)


    # Concatenate categorical, time and continuous features such that all features are interconnected
    x = concatenate([c,t,f])

    # add linear layers
    x = Dense(500,activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x)
    x = Dropout(rate=r)(x)
    x = BatchNormalization()(x)
    x = Dense(500,activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x)
    x = Dropout(rate=r)(x)
    x = BatchNormalization()(x)
    x = Dense(500,activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(x)    
    x = Dropout(rate=r)(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='sigmoid',kernel_initializer='glorot_normal',bias_initializer='normal')(x) 

    # set input layer and compile
    model = Model([inp for inp,cat in cats] + [inp for inp,time in times] + [inp for inp,cont in conts], x)
    model.compile(optimizer='Adam',loss=SparseCrossEntropy(),metrics=['accuracy','Precision','Recall','AUC'])
    return model



# In[ ]:


loan_model = build_loan_model(cats,times,conts)


# In[ ]:


# Initialize callbacks
# both earlystopping and modelcheckpoint save the best model to be stored

es = EarlyStopping(monitor='val_loss', patience=5, verbose=0,
    mode='auto', min_delta=.005, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                              mode = 'min', min_delta=.005, patience=3, min_lr=0.0001)

filepath = 'loan_checkpoint'
mckp = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True,
    save_weights_only=True, mode='auto', save_freq='epoch')


# In[ ]:


# Train the model using early stopping and reducing learning rates to reduce overfitting and 'memorization'.

loan_model.fit(X_train,y_train,
            batch_size=64,
            epochs=20,
            verbose=True,
            validation_data = (X_val, y_val), 
            class_weight=train_weight,
            callbacks=[es,rlr,mckp])


# #### Display convergence of metrics

# In[ ]:


dh = pd.DataFrame(data=loan_model.history.history)
dh
dh[['loss','val_loss']].plot()
dh[['Precision','val_Precision']].plot()
dh[['Recall','val_Recall']].plot()
dh[['binary_accuracy', 'val_binary_accuracy']].plot()
dh[['AUC','val_AUC']].plot()


# #### Display classification and confusion matrices

# In[ ]:


# Performance on the validation set

predicted = loan_model.predict(X_val)
predictions = np.rint(predicted)
print('Training Set Confusion Matrix')
print(confusion_matrix(y_val,predictions))
print('Training Set Classification Report')
print(classification_report(y_val,predictions))


# In[ ]:


# Performance on the Test set

predicted = loan_model.predict(X_test)
predictions = np.rint(predicted)
print('Test Set Confusion Matrix')
print(confusion_matrix(y_test,predictions))
print('Test Set Classification Report')
print(classification_report(y_test,predictions))


# ### The important comparison is between the training set and the test set.  Both the confusion matrix and the classification report are nearly identical for both data sets meaning that the model will generalize to new data well.

# In[ ]:


from tensorflow.keras.backend import clear_session
clear_session()


# In[ ]:


# Train the predictor using all of the data set except the reserved test set for final evaluation

# set the number of epochs from validation set model using early stopping

e = dh.val_loss.min()
e = dh.index[dh['val_loss']== e]
epochs = e[0]

# initiate callbacks

es = EarlyStopping(monitor='loss', patience=5, verbose=0,
    mode='auto', min_delta=.005, restore_best_weights=True)

rlr = ReduceLROnPlateau(monitor='loss', factor=0.9,
                              mode = 'min', min_delta=.005, patience=2, min_lr=0.0001)

mckp = ModelCheckpoint(filepath='loan_all_checkpoint', monitor='loss', verbose=0, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch')


# In[ ]:


# Train model on merged train and validation set with epochs limited to the min loss epoch on validation set
# when order_model is fit the second time, even with changed data, it loads the final weights from the first fit automatically
# clear_session does not reset the weights in the model and the learning rate starts at the last rlr learning rate
# 

loan_model.fit(X_all,y_all,
            batch_size=64,
            epochs=epochs,
            verbose=True,
            validation_data = None, class_weight=all_weight,
            callbacks=[es,rlr])


# In[ ]:


# Save model for prediction

#filepath = 'win_all_model'
#save_model(loan_model, filepath, overwrite=True, include_optimizer=True, save_format=None,
#    signatures=None, options=None)


# #### Display convergence

# In[ ]:


da = pd.DataFrame(data=loan_model.history.history)
da
da[['binary_accuracy']].plot()
da[['Precision']].plot()
da[['Recall']].plot()
da[['AUC']].plot()
da[['loss']].plot()


# #### Display confusion and classificaton matrices

# In[ ]:


# Display classification report and confusion matrix for fit on all data
predicted = loan_model.predict(X_all)
predictions = np.rint(predicted)
print('Win All Confusion Matrix')
print(confusion_matrix(y_all,predictions))
print('Win All Classification Report')
print(classification_report(y_all,predictions))


# In[ ]:


# Display classification report and confusion matrix from fit on all data for test data
predicted = loan_model.predict(X_test)
predictions = np.rint(predicted)
print('Loan Test Confusion Matrix')
print(confusion_matrix(y_test,predictions))
print('Loan Test Classification Report')
print(classification_report(y_test,predictions))


# In[ ]:


from tensorflow.keras.backend import clear_session
clear_session()
del loan_model
gc.collect()


# In[ ]:




