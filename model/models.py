# %%
from keras.layers import Input
from keras.models import Model
import pandas as pd
import os.path
import _pickle as cpickle
import numpy as np
import keras.utils
import time
from keras.callbacks import TensorBoard, CSVLogger
from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout,Bidirectional, Conv2D, GRU
from keras.layers.embeddings import Embedding
from keras import optimizers
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.metrics import categorical_accuracy

def model(idx_model,X_train_meta,lstm_size,hidden_size,num_steps,num_epochs,batch_size,kernel_sizes,filter_size,vocab_length,EMBEDDING_DIM,embedding_matrix,label_reverse_arr,binaresed):
    if binaresed == True:
      dim_out = 2
    else:
      dim_out = 6


    if idx_model ==1:
      #Keras LSTM Model (defining sequential model simple)
      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(LSTM(hidden_size))
      model.add(Dense(dim_out, activation='softmax'))
      return model
    
    if idx_model == 2:
      #Defining a complex model. Adding meta data features to the model (LSTM Based)
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = LSTM(lstm_size,dropout=0.2)(x)
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)
      x = keras.layers.concatenate([lstm_in, x_meta])
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==3:
      # seck tinsting ok
      #Defining a complex model. Adding meta data features to the model (CNN Based)
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      #x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      #x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_meta = Dense(64, activation='relu')(meta_input)
      #x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      #x = Dense(64, activation='relu')(x) #Add some density
      #x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==4:
      #Defining a complex model. Adding meta data features to the model (CNN Based) + LSTM
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

      lstm_in = LSTM(lstm_size,dropout=0.2)(x) # seck

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      #x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_meta = Dense(64, activation='relu')(meta_input)
      #x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta,lstm_in])
      x = Dense(128, activation='relu')(x) #Add some density
      #x = Dense(64, activation='relu')(x) #Add some density
      #x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==5:
      #Defining a complex model. Adding meta data features to the model (CNN Based)+LSTM
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings

      lstm_in = LSTM(lstm_size,dropout=0.2)(x) # seck

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      #x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_meta = Dense(64, activation='relu')(meta_input)
      #x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta,lstm_in])
      x = Dense(128, activation='relu')(x) #Add some density
      #x = Dense(64, activation='relu')(x) #Add some density
      #x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==6:
      #Defining a complex model. Adding meta data features to the model (CNN Based) + LSTM
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

      lstm_in = LSTM(lstm_size,dropout=0.2)(x) # seck

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_em = LSTM(lstm_size,dropout=0.2)(x_em) # seck
      x_meta = Dense(64, activation='relu')(x_em)
      x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta,lstm_in])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==7:
      #Defining a complex model. Adding meta data features to the model (CNN Based)+LSTM
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings

      lstm_in = LSTM(lstm_size,dropout=0.2)(x) # seck

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_em = LSTM(lstm_size,dropout=0.2)(x_em) # seck
      x_meta = Dense(64, activation='relu')(x_em)
      x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta,lstm_in])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==8:
      #Defining a complex model. Adding meta data features to the model (CNN Based) + LSTM
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

      lstm_in_1 = LSTM(lstm_size,dropout=0.2)(x) # seck
      lstm_in_2 = LSTM(lstm_size,dropout=0.2)(x) # seck
      lstm_in_3 = LSTM(lstm_size,dropout=0.2)(x) # seck


      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)

      #Meta input
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_em = Embedding(output_dim=hidden_size, input_dim=2, input_length=X_train_meta.shape[1])(meta_input)
      x_em = LSTM(lstm_size,dropout=0.2)(x_em) # seck
      x_meta = Dense(64, activation='relu')(x_em)
      x_meta = Dense(56, activation='relu')(x_meta)
      x = keras.layers.concatenate([conv_in, x_meta,lstm_in_1,lstm_in_2,lstm_in_3])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model


    # Defining a complex model. Adding meta data features to the model (LSTM Based)
    if idx_model==9:
      lstm_size=100
      
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = LSTM(lstm_size,dropout=0.2, return_sequences=False)(x)

      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)

      x = keras.layers.concatenate([lstm_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model
    
    if idx_model==10:
      lstm_size=100

      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = GRU(lstm_size,dropout=0.2)(x)

      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)

      x = keras.layers.concatenate([lstm_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==11:
      lstm_size=100

      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = Bidirectional(LSTM(lstm_size, return_sequences=False))(x)

      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)

      x = keras.layers.concatenate([lstm_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==12:
      lstm_size=100
      
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = LSTM(lstm_size,dropout=0.2, return_sequences=True)(x)
      lstm_in = LSTM(lstm_size,dropout=0.2, return_sequences=False)(lstm_in)
      
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)

      x = keras.layers.concatenate([lstm_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    if idx_model==13:
      lstm_size=100
      
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      lstm_in = Bidirectional(LSTM(lstm_size,dropout=0.2, return_sequences=True))(x)
      lstm_in = Bidirectional(LSTM(lstm_size,dropout=0.2, return_sequences=False))(lstm_in)
      
      meta_input = Input(shape=(X_train_meta.shape[1],), name='aux_input')
      x_meta = Dense(64, activation='relu')(meta_input)

      x = keras.layers.concatenate([lstm_in, x_meta])
      x = Dense(128, activation='relu')(x) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density
      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input, meta_input], outputs=[main_output])
      return model

    # Defining a simple model. No meta data features to the model (LSTM Based)
    if idx_model==14:
      lstm_size=100

      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(LSTM(lstm_size))
      model.add(Dense(dim_out, activation='softmax'))
      return model

    if idx_model==15:
      lstm_size=100

      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(GRU(lstm_size))
      model.add(Dense(dim_out, activation='softmax'))
      return model 

    if idx_model==16:
      lstm_size=100

      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(Bidirectional(LSTM(lstm_size)))
      model.add(Dense(dim_out, activation='softmax'))
      return model 
    

    if idx_model==17:
      lstm_size=100

      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(LSTM(lstm_size,return_sequences=True))
      model.add(LSTM(lstm_size,return_sequences=False))
      model.add(Dense(dim_out, activation='softmax'))
      return model

    
    if idx_model==18:
      lstm_size=100

      model = Sequential()
      model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
      model.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
      model.add(Bidirectional(LSTM(lstm_size, return_sequences=False)))
      model.add(Dense(dim_out, activation='softmax'))
      return model

    if idx_model==19:
      # seck tinsting ok
      #Defining a complex model. No meta (CNN Based)
      kernel_arr = []
      statement_input = Input(shape=(num_steps,), dtype='int32', name='main_input')
      x = Embedding(vocab_length+1,EMBEDDING_DIM,weights=[embedding_matrix],input_length=num_steps,trainable=False)(statement_input) #Preloaded glove embeddings
      #x = Embedding(output_dim=hidden_size, input_dim=vocab_length+1, input_length=num_steps)(statement_input) #Train embeddings from scratch

      for kernel in kernel_sizes:
          x_1 = Conv1D(filters=filter_size,kernel_size=kernel)(x)
          x_1 = GlobalMaxPool1D()(x_1)
          kernel_arr.append(x_1)
      conv_in = keras.layers.concatenate(kernel_arr)
      conv_in = Dropout(0.6)(conv_in)
      conv_in = Dense(128, activation='relu')(conv_in)
      
      x = Dense(128, activation='relu')(conv_in) #Add some density
      x = Dense(64, activation='relu')(x) #Add some density
      x = Dense(32, activation='relu')(x) #Add some density

      main_output = Dense(dim_out, activation='softmax', name='main_output')(x)
      model = Model(inputs=[statement_input], outputs=[main_output])
      return model

# %%
