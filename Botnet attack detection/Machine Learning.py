import numpy as np
import pandas as pd
import time

dataset = pd.read_csv('C:/Users/User/Spyder Warehouse/n-BaIoT/n_BaIoT_concentrated3.csv')

x = dataset.iloc[:,1:116]
y = dataset.iloc[:,117]
del dataset
y=y.astype('int')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

##### Random Forest Classifier #####
start=time.time()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
end=time.time()
time_spent=round(end - start)
###################################

##### K-nearest Neighbours Classifier #####
start=time.time()
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knnClassifier.fit(x_train, y_train)
y_pred = knnClassifier.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
end=time.time()
time_spent=round(end - start)
###################################

##### Naive Bayes Classifier ######
start=time.time()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
end=time.time()
time_spent=round(end - start)
###################################

##### Decision Tree ###############
start=time.time()
from sklearn import tree
clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

end=time.time()
time_spent=round(end - start)
###################################


