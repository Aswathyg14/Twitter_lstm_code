#!/usr/bin/env python
# coding: utf-8

# In[35]:


#importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
import nltk.corpus
import re
from numpy import asarray
from numpy import zeros
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[36]:


d= pd.read_csv(r"C:\Users\aswab\Desktop\dataset\train.csv")


# In[37]:


d


# In[38]:


d.info


# In[39]:


d.isnull().sum()


# In[40]:


d=d.dropna()


# In[41]:


d


# In[42]:


d.isnull().sum()


# In[43]:


d.describe


# # EDA

# In[44]:


sns.countplot(x=d['label'])


# In[45]:


def preprocess(text):
    # remove mentions and hashtags
    text = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', text)
    # remove urls
    text = re.sub(r'http\S+', '', text)
    # tokenize using TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(text)
    # remove stopwords
    stopword_list = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopword_list]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # join the tokens back into a string
    text = ' '.join(tokens)
    return text


# In[47]:


import nltk
nltk.download('stopwords')

d['label'] = d['label'].apply(preprocess)


# In[49]:


d


# In[50]:


d.tail


# In[51]:


d.head


# # Label encoding

# In[52]:


#label encoding

from sklearn import preprocessing   
label = preprocessing.LabelEncoder()
d[ 'label' ]= label.fit_transform(d[ 'label' ]) 


# In[53]:


d


# # Vectorization and data splitting

# In[54]:


#Vectorization and data splitting

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)


# In[56]:


X=cv.fit_transform(d["Tweets"]).toarray()


# In[57]:


cv.vocabulary_


# In[58]:


y=d["label"]


# In[59]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# In[60]:


print(y_train)


# # now preprocessing steps applied to test data

# In[114]:


df=pd.read_csv(r"C:\Users\aswab\Desktop\dataset\test.csv", names= ["tweets","sentiment"])


# In[115]:


df


# In[116]:


df.info


# In[117]:


df


# In[118]:


df.isnull().sum()


# In[119]:


df.columns


# # EDA

# In[120]:


sns.countplot(x=df['sentiment'])


# In[121]:


def preprocessing(text):
    # remove mentions and hashtags
    text = re.sub(r'@[A-Za-z0-9_]+|#[A-Za-z0-9_]+', '', text)
    # remove urls
    text = re.sub(r'http\S+', '', text)
    # tokenize using TweetTokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(text)
    # remove stopwords
    stopword_list = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopword_list]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # join the tokens back into a string
    text = ' '.join(tokens)
    return text


# In[122]:


import nltk
nltk.download('stopwords')

df['sentiment'] = df['sentiment'].apply(preprocessing)


# # Label Enconding for test data

# In[123]:


from sklearn import preprocessing   
label = preprocessing.LabelEncoder()
df[ 'sentiment' ]= label.fit_transform(df[ 'sentiment' ]) 


# In[124]:


df


# # Vectorization and splitting the datsset

# In[129]:


from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer(max_features=1000)
X = vector.fit_transform(df["tweets"]).toarray()


# In[131]:


X=vector.fit_transform(df["tweets"]).toarray()


# In[104]:


y=df["sentiment"]


# In[134]:


from sklearn.model_selection import train_test_split

# Assuming X and y are already defined
# If not, replace them with your actual features and labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[135]:


from sklearn.model_selection import train_test_split

# Assuming X and y are already defined
# If not, replace them with your actual features and labels

# Ensure y is a 1D array (or a pandas Series)
y = y.ravel()

# Check the shapes of X and y
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[132]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# # Now apply LSTM

# In[108]:


#import libraries

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence


# In[ ]:


X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2)


# In[111]:


# fix random seed for reproducibility
tf.random.set_seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = d.load_data(num_words=top_words)

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

tf.random.set_seed(7)
(X_train, y_train) = load_custom_train_data()
top_words = 5000
max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)

# Your LSTM model training code would go here

# Example model compilation (replace with your actual model)
model = tf.keras.Sequential([
    # Your LSTM layers and other layers go here
    # Example: tf.keras.layers.Embedding(top_words, 50, input_length=max_review_length),
    #         tf.keras.layers.LSTM(100),
    #         tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Example model training (replace with your actual training code)
model.fit(X_train, y_train, epochs=5, batch_size=64)


# In[112]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed for reproducibility
tf.random.set_seed(10)

# Define the LSTM model
embedding_dim = 50 
vocab_size = 1000  
max_length = X.shape[1]  

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
print(model.summary())

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred = model.predict_classes(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Display additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[113]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the LSTM model
embedding_dim = 50  # Adjust this based on your requirements
vocab_size = 1000  # Max features in CountVectorizer
max_length = X.shape[1]  # Max sequence length

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display the model summary
print(model.summary())

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)  # Thresholding for binary classification

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Display additional metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:




