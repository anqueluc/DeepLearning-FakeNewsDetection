import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Bidirectional,LSTM,Conv1D,GlobalMaxPool1D,Dropout, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing import sequence
import os
import _pickle as cpickle


path = os.getcwd()

root_file = path + "/home/lucas/Bureau/M2_S.I.D/S1/Deep_Learning/Projet/demo/"
embedding_file = path + "/../data/vocab.p"
best_model_weights = path + "/../results/categorical_models/16/blstm_1layer_70cells/weights.best.hdf5"


#%%
#read file
df_obama_true = pd.read_csv("./real_obama.txt", header=None, sep='\n')
words = list(df_obama_true.iloc[0])
real_obama = ''.join(words)
df_real_obama = pd.DataFrame({'statement':[real_obama]})

df_trump_true = pd.read_csv("./real_trump.txt", header=None, sep='\n')
words = list(df_trump_true.iloc[0])
real_trump = ''.join(words)
df_real_trump = pd.DataFrame({'statement':[real_trump]})

df_obama_fake = pd.read_csv("./fake_obama.txt", header=None, sep='\n')
words = list(df_obama_fake.iloc[0])
fake_obama = ''.join(words)
df_fake_obama = pd.DataFrame({'statement':[fake_obama]})

df_trump_fake = pd.read_csv("./fake_trump.txt", header=None, sep='\n')
words = list(df_trump_fake.iloc[0])
fake_trump = ''.join(words)
df_fake_trump = pd.DataFrame({'statement':[fake_trump]})
print('FILES READ\n')

#%%
#Preprocessing of the news
vocab_dict = {}
vocab_dict = cpickle.load(open(embedding_file, "rb" ))

def pre_process_statement(statement):
    text = text_to_word_sequence(statement)
    val = [0] * 10
    val = [vocab_dict[t] for t in text if t in vocab_dict] #Replace unk words with 0 index
    return val

df_real_obama['word_ids'] = df_real_obama['statement'].apply(pre_process_statement)
real_obama_test = df_real_obama['word_ids']
df_fake_obama['word_ids'] = df_fake_obama['statement'].apply(pre_process_statement)
fake_obama_test = df_fake_obama['word_ids']
df_real_trump['word_ids'] = df_real_trump['statement'].apply(pre_process_statement)
real_trump_test = df_real_trump['word_ids']
df_fake_trump['word_ids'] = df_fake_trump['statement'].apply(pre_process_statement)
fake_trump_test = df_fake_trump['word_ids']
print('PREPROCESSING DONE\n')

#%%
#load model
vocab_length = len(vocab_dict.keys())
hidden_size = 70
num_steps = 25
real_obama_test = sequence.pad_sequences(real_obama_test, maxlen=num_steps, padding='post',truncating='post')
fake_obama_test = sequence.pad_sequences(fake_obama_test, maxlen=num_steps, padding='post',truncating='post')
real_trump_test = sequence.pad_sequences(real_trump_test, maxlen=num_steps, padding='post',truncating='post')
fake_trump_test = sequence.pad_sequences(fake_trump_test, maxlen=num_steps, padding='post',truncating='post')

model = Sequential()
model.add(Embedding(vocab_length+1, hidden_size, input_length=num_steps))
model.add(Bidirectional(LSTM(hidden_size)))
model.add(Dense(6, activation='softmax'))
model.load_weights(best_model_weights)

print('MODEL LOADED\n' )
#%%
#prediction
y_label_dict = {0:'pants-fire',1:'false',2:'barely-true',3:'half-true',4:'mostly-true',5:'true'}

y_real_obama = model.predict(real_obama_test)
idx_y_hat = np.argmax(y_real_obama)
y_hat_real_obama = y_label_dict[idx_y_hat]
print("Prediction for the real tweet of Obama :")
print(y_hat_real_obama + '\n')

y_fake_obama = model.predict(fake_obama_test)
idx_y_hat = np.argmax(y_fake_obama)
y_hat_fake_obama = y_label_dict[idx_y_hat]
print("Prediction for the fake tweet of Obama :")
print(y_hat_fake_obama + '\n')

y_real_trump = model.predict(real_trump_test)
idx_y_hat = np.argmax(y_real_trump)
y_hat_real_trump = y_label_dict[idx_y_hat]
print("Prediction for the real tweet of Trump :")
print(y_hat_real_trump + '\n')

y_fake_trump = model.predict(fake_trump_test)
idx_y_hat = np.argmax(y_fake_trump)
y_hat_fake_trump = y_label_dict[idx_y_hat]
print("Prediction for the fake tweet of Trump :")
print(y_hat_fake_trump + '\n')




#result