def log_cont_data(data, cont_vars_log=cont_vars_log):   
  # take logs of each variable to be logged identified
  for feat in cont_vars_log:
    data[feat] = data[feat].apply(lambda x: np.log1p(x).astype(np.float32))
    data = data[cont_vars_log]
  # return data with logged columns
  data.reset_index(drop=True, inplace=True)
  return data

# log selected continuous variables
data_log = log_cont_data(data)

# concatenate encoded, scaled, logged data and target
data_scaled = np.hstack([data_cat, data_time, data_cont, data_log, data_target])

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

 # concatenate if 2 or more variables
    l = ([cont for inp,cont in log_conts], axis = 1)

    # add dense, dropout and batch normalization layers fully interconnected
    l = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(l)
    l = Dropout(rate=r)(l)
    l = BatchNormalization()(l)
    l = Dense(500, activation='relu',kernel_initializer='glorot_normal',bias_initializer='normal')(l)
    l = Dropout(rate=r)(l)
    l = BatchNormalization()(l)
    
    # Concatenate categorical, time, continuous and continuous logged features
    x = concatenate([c,t,f,l], axis=1)


# add input layer for the model
    model = Model([inp for inp,cat in cats] + [inp for inp,time in times] + [inp for inp,cont in conts] + [inp for inp,cont in log_conts], x)
    
    # set learning rate