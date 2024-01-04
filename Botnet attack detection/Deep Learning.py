import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, SimpleRNN
import pandas as pd
import numpy as np
import time

dataset = pd.read_csv('C:/Users/GrBel/spyder environ/n-BaloT/n_BaIoT_concentrated.csv')

x = dataset.iloc[:,1:116]
y = dataset.iloc[:,117]

del dataset

y=y.astype('int')

x=pd.DataFrame(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


x1=x

### F REGRESSION SELECTION ###
from sklearn.feature_selection import f_regression, SelectKBest
x_select = SelectKBest(score_func=f_regression, k=20).fit_transform(x, y)
x=x_select
x= pd.DataFrame(x)
# x.to_csv('x-selected-best-results',index=False)
#####

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)
#####

### STANDARDIZATION ###
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#####

from keras.utils import to_categorical
from sklearn.preprocessing import Normalizer
from keras.utils.np_utils import to_categorical

y_train1 = np.array(y_train)
y_test1 = np.array(y_test)

### CATEGORIZATION ###
y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)
#####

########### FOR COMPILING RNN MODELS #############
# x_train = np.reshape(x_train, (x_train.shape[0],1, x_train.shape[1]))
# x_test = np.reshape(x_test, (x_test.shape[0],1,x_test.shape[1]))

########### FOR COMPILING OTHER MODELS #############
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))




########## MODEL TO BE CHOSEN ##########

##############    LSTM SINGLE     ##################
model = Sequential()
model.add(LSTM(32, activation='sigmoid', return_sequences=False))
model.add(Flatten())
model.add(Dense(11, activation='softmax'))
##############     LSTM STACKED    #################
# model = Sequential()
# model.add(LSTM(32, activation='sigmoid', return_sequences=True))
# model.add(LSTM(32, activation='sigmoid', return_sequences=False))
# model.add(Flatten())
# model.add(Dense(11, activation='softmax'))
##############     ANN SINGLE      #################
# model = Sequential()
# model.add(Dense(64, activation='sigmoid'))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
##############     ANN STACKED     #################
# model = Sequential()
# model.add(Dense(64, activation='sigmoid'))
# model.add(Dense(64, activation='sigmoid'))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
# ############     RNN SINGLE      ###################
# model = Sequential()
# model.add(SimpleRNN(128,activation='relu',return_sequences=False))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
# #############     RNN STACKED     ##################
# model = Sequential()
# model.add(SimpleRNN(128,activation='relu',return_sequences=True))
# model.add(SimpleRNN(128,activation='relu',return_sequences=False))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
# ##############     CNN SINGLE      #################
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
# ##############      CNN STACKED   #################
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid',padding='same'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid',padding='same'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(Flatten()) 
# model.add(Dense(11,activation='softmax'))
#############   CNN-LSTM SINGLE   ##################
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(LSTM(32, activation='sigmoid'))
# model.add(Flatten())
# model.add(Dense(11, activation='softmax'))
# #############   CNN-LSTM STACKED  ##################
# model = Sequential()
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid',padding='same'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid',padding='same'))
# model.add(MaxPooling1D(pool_size=5, strides=2))
# model.add(LSTM(32, activation='sigmoid', return_sequences=True))
# model.add(LSTM(32, activation='sigmoid', return_sequences=False))
# model.add(Flatten())
# model.add(Dense(11, activation='softmax'))

start=time.time()
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=25)
end=time.time()
time_spent=round(end - start)


y_pred=np.argmax(model.predict(x_test), axis=-1)
y_test2=np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score,f1_score
cm = confusion_matrix(y_test2, y_pred)
acc = accuracy_score(y_test2, y_pred)
precision = precision_score(y_test2, y_pred, average='weighted',zero_division=1)
recall = recall_score(y_test2, y_pred, average='weighted',zero_division=1)
f1score = f1_score(y_test2, y_pred, average='weighted',zero_division=1)
