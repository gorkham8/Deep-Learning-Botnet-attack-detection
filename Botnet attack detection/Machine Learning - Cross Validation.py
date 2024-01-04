import numpy as np
import pandas as pd
import time

dataset = pd.read_csv('C:/Users/User/Spyder Warehouse/n-BaIoT/n_BaIoT_concentrated.csv')

x = dataset.iloc[:,1:116]
y = dataset.iloc[:,117]
del dataset
y=y.astype('int')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)
from sklearn.preprocessing import Normalizer
transformer = Normalizer().fit(x)


##### Random Forest Classifier #####
start=time.time()
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
from sklearn.model_selection import cross_val_score
scores=cross_val_score(classifier,x,y,cv=3)
mean=np.mean(scores)
end=time.time()
time_spent=round(end - start)
###################################


##### K-nearest Neighbours Classifier #####
start=time.time()
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(knnClassifier,x,y,cv=3)
mean=np.mean(scores)
end=time.time()
time_spent=round(end - start)
###################################

##### Naive Bayes Classifier ######
start=time.time()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
from sklearn.model_selection import cross_val_score
scores=cross_val_score(gnb,x,y,cv=3)
mean=np.mean(scores)
end=time.time()
time_spent=round(end - start)
###################################

##### Decision Tree ###############
start=time.time()
from sklearn import tree
clf = tree.DecisionTreeClassifier()
from sklearn.model_selection import cross_val_score
scores=cross_val_score(clf,x,y,cv=3)
mean=np.mean(scores)
end=time.time()
time_spent=round(end - start)
###################################


