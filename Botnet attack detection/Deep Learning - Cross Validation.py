import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, SimpleRNN
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('C:/Users/GrBel/Spyder Environ/n-BaloT/n_BaIoT_concentrated.csv')
df=dataset
del dataset

x = df.iloc[:,1:116]
y = df.iloc[:,117]
y=y.astype('int')

### F REGRESSION SELECTION ###
from sklearn.feature_selection import f_regression, SelectKBest
x_select = SelectKBest(score_func=f_regression, k=20).fit_transform(x, y)
x=x_select
#####


from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import Normalizer
from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
# from keras.layers import Dense, Dropout, Activation, Input, Dense, Flatten, Bidirectional
# from keras.layers import SimpleRNN, LSTM
import tensorflow.keras.optimizers
import time

accuracyscores = []
timescores = []
prescores = []
recscores = []
f1scores = []
cmlist = []
i=0

kfold = KFold(n_splits=3, random_state=0, shuffle=True)
for train, test in kfold.split(x, y):
    print(i)
    i=i+1
    y_train = y[train]
    y_test = y[test]
    x_train = x.iloc[train]
    x_test = x.iloc[test]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)     

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    y_train= to_categorical(y_train)
    y_test= to_categorical(y_test)
    
    
    ##### FOR COMPILING RNN #####
    # x_train = np.reshape(x_train, (x_train.shape[0],1, x_train.shape[1]))
    # x_test = np.reshape(x_test, (x_test.shape[0],1, x_test.shape[1]))
    
    ##### FOR COMPILING OTHER MODELS #####
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    
    ##############    LSTM SINGLE     ##################
    model = Sequential()
    model.add(LSTM(32, activation='sigmoid', return_sequences=True))
    model.add(Flatten())
    model.add(Dense(11, activation='softmax'))
    # ##############     LSTM STACKED    #################
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
    # model.add(SimpleRNN(128, input_dim=20,activation='relu',return_sequences=False))
    # model.add(Flatten()) 
    # model.add(Dense(11,activation='softmax'))
    # #############     RNN STACKED     ##################
    # model = Sequential()
    # model.add(SimpleRNN(128, input_dim=20,activation='relu',return_sequences=True))
    # model.add(SimpleRNN(128,activation='relu',return_sequences=False))
    # model.add(Flatten()) 
    # model.add(Dense(11,activation='softmax'))
    # ##############     CNN SINGLE      #################
    # model = Sequential()
    # model.add(Conv1D(filters=64, kernel_size=5, strides=1, activation='sigmoid', padding='same'))
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
    
    
    optimizer = tensorflow.keras.optimizers.Adam(lr=0.001)
    batch_size=32
    
    start = time.time()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    results = model.fit(x_train, y_train, batch_size=batch_size,epochs=6)
    score, acc = model.evaluate(x_test, y_test,batch_size=32)
    
   
    end = time.time()
    y_pred=np.argmax(model.predict(x_test), axis=-1)
    y_test= np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    cmlist.append(cm)

    accuracyscores.append(round(acc, 4) * 100)
    timescores.append(round(end - start, 4))
    precision = precision_score(y_test, y_pred, average='macro',zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro',zero_division=1)
    f1score = f1_score(y_test, y_pred, average='macro',zero_division=1)
    prescores.append(round(precision, 4) * 100)
    recscores.append(round(recall, 4) * 100)
    f1scores.append(round(f1score, 4) * 100)


print('Test accuracy:', np.mean(accuracyscores))
print("Completed in", np.mean(timescores), "seconds")
print("Classification report: ")
print("Precision: ", np.mean(prescores))
print("Recall : ", np.mean(recscores))
print("F1-Score: ", np.mean(f1scores))

print(len(cmlist))
zeros_row = [0,0,0,0]
zeros_column = [[0],[0],[0],[0]]

for i in range(3):
        if (len(cmlist[i])==3):
                cmlist[i] = cmlist[i].tolist()
                cmlist[i] = [x + [0] for x in cmlist[i]]
                cmlist[i].append(zeros_row)
                cmlist[i] = np.asarray(cmlist[i])
    



