# -*- coding: utf-8 -*-

import pandas as pd
import os.path
#import _pickle as cpickle
import numpy as np
import keras.utils
#import time
#from keras.callbacks import TensorBoard, CSVLogger
#from keras.models import Sequential
#from keras.layers import Dense,Flatten,LSTM,Conv1D,GlobalMaxPool1D,Dropout
#from keras.layers.embeddings import Embedding
from keras import optimizers
#import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.metrics import categorical_accuracy
from sklearn.metrics import confusion_matrix
import seaborn as sn 
from keras.models import load_model
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from keras.utils.vis_utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from model import loaders
from model import models

'''
HYPER PARAMETERS
'''
lstm_size = 100
hidden_size = 128
num_steps = 25
num_epochs = 32
batch_size = 32
kernel_sizes = [2,5,8]
filter_size = 128
idx_model = 19
binaresed = True

'''
LOADER DATA 
'''
(X_train_meta,X_val_meta,X_test_meta),(X_train,Y_train),(X_val,Y_val),(X_test,Y_test),vocab_length,EMBEDDING_DIM,embedding_matrix,label_reverse_arr= loaders.read(num_steps,binaresed=binaresed)

#CREATE AND STATE AT THE SAVE DIR
path = os.getcwd()

save_dir = path + "/results/" + str(idx_model) + "/" + str(num_epochs) + "/" 
if not os.path.exists(save_dir):
        os.makedirs(save_dir)
os.chdir(save_dir)

#Load the model
model = models.model(idx_model,X_train_meta,lstm_size,hidden_size,num_steps,num_epochs,batch_size,kernel_sizes,filter_size,vocab_length,EMBEDDING_DIM,embedding_matrix,label_reverse_arr,binaresed=binaresed)
#Compile model and print summary
#Define specific optimizer to counter over-fitting
sgd = optimizers.SGD(lr=0.025, clipvalue=0.3, nesterov=True)
adam = optimizers.Adam(lr=0.000075, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

print(model.summary())

with open(save_dir + 'modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

#SAVE CHECK POINT 
csv_logger = keras.callbacks.CSVLogger(save_dir + 'training.csv')
filepath= save_dir + "weights.best.hdf5"
#Or use val_loss depending on whatever the heck you want
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')

#SAVE ARCHITECTURE 
#Visualize model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
SVG(model_to_dot(model,show_shapes=True).create(prog='dot', format='svg'))

'''
TRAINING
'''
if idx_model == 1 or idx_model==19:
  #Start training the sequential model here
  history = model.fit(X_train,Y_train,batch_size=batch_size,epochs=num_epochs,verbose=1,validation_data=(X_val,Y_val),callbacks=[csv_logger,checkpoint])

else:
  history = model.fit({'main_input': X_train, 'aux_input': X_train_meta},
            {'main_output': Y_train},epochs=num_epochs, batch_size=batch_size,
            validation_data=({'main_input': X_val, 'aux_input': X_val_meta},{'main_output': Y_val}),
          callbacks=[csv_logger,checkpoint])

'''
VISUALISATION
'''
fig = plt.figure(figsize=(5, 5))
# Plot training & validation accuracy values
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
fig.savefig('acc.png')
plt.close()

fig = plt.figure(figsize=(5, 5))
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'validation'], loc='upper left')
plt.show()
fig.savefig('loss.png')
plt.close()
  
'''
LOAD THE BEST MODEL AND PRED
'''
#Load a pre-trained model if any
model1 = load_model('weights.best.hdf5')

if idx_model == 1 or idx_model==19:
  #Make predictions on test set
  preds = model1.predict([X_test], batch_size=batch_size, verbose=1)
else:
  preds = model1.predict([X_test,X_test_meta], batch_size=batch_size, verbose=1)

#Write data to array
Y_pred = np.zeros((len(preds),6))
counter = 0
for pred in preds:  
    idx = np.argmax(pred)
    Y_pred[counter][idx] = 1
    counter += 1



c=categorical_accuracy(Y_test, Y_pred)
sess = K.get_session()
array = sess.run(c)

score = np.sum(array)/len(preds)
print("score : ", score)

with open('resultats.txt', 'w') as f:
    f.writelines("num_steps : "+str(num_steps)+"\n")
    f.writelines("num_epochs : "+str(num_epochs)+"\n")
    f.writelines("batch_size : "+str(batch_size)+"\n")

    f.writelines("num_kernel_sizesepochs : "+str(kernel_sizes)+"\n")
    f.writelines("filter_size : "+str(filter_size)+"\n")
    
    f.writelines("idx_model : "+str(idx_model)+"\n")
    f.writelines("score : "+str(score)+"\n")

'''
Confusion matrix to understand what the model learnt
'''
YY_test = []
YY_pred = []
for i in range(len(Y_test)):
  YY_test.append(np.argmax(Y_test[i]))
  YY_pred.append(np.argmax(Y_pred[i]))

YY_test = np.array(YY_test)
YY_pred = np.array(YY_pred)

a = confusion_matrix(YY_test, YY_pred)
if binaresed == True: 
  df = pd.DataFrame(a, index=['false','true'], 
                    columns=['false','true'])
else: 
  df = pd.DataFrame(a, index=['pants-fire','false','barely-true','half-true','mostly-true','true'], 
                  columns=['pants-fire','false','barely-true','half-true','mostly-true','true'])

plt.figure()
plt.title('confusion matrix : model_'+str(idx_model))
sn.heatmap(df, annot=True)
plt.show()