#!/usr/bin/env python
# coding: utf-8

# In[69]:


# Methods to unpack json file and import as pandas data frame
import json
import pandas as pd
import numpy as np
import gzip

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.utils import to_categorical

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


df = getDF('Software_5.json.gz')


# In[70]:


# Extracting predictor and target variables
X, y = df['reviewText'].values.astype('U'),df['overall']

max_features = 25000 # number of words to consider as features
maxlen = 1000 # cuts off the text after this number of words (among the most common words in max_features)

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[71]:


# keras preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

sequences = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(sequences, maxlen=maxlen)

sequences = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(sequences, maxlen=maxlen)

y_train = to_categorical(y_train) # Cannot set number of classes to 5. For some reason it's 6. TODO
y_test = to_categorical(y_test) # Cannot set number of classes to 5. For some reason it's 6. TODO


# In[72]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# In[73]:


def BuildModel():
    
    model = Sequential()
    model.add(Embedding(25000, 64, input_length=maxlen)) # max_words, embedding_dim, input_length
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    
    return model


# In[ ]:


model = BuildModel()

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['acc'])
model.fit(x=X_train, y=y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1)


# In[ ]:


model.evaluate(x=X_test, y=y_test, batch_size=64, verbose=1)


# In[ ]:




