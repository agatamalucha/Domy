# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:00:03 2020

@author: Agatka
"""
import pandas as pd

import numpy as np
import matplotlib.pyplot  as plt  # imports necessery functions for charts
import seaborn as sb
from pandas import DataFrame



def cut_string(string):
    """helper function that returns first two elements of the string"""
    if string[0]=='>':
        new_string=string[1:3]
    else:
        new_string=string[:2]
    return new_string

def extract_third_element(string):
    """helper function that extracts third element of the list if present"""
    lst=string.split(', ')
    if len(lst)==3:
        element=lst[2]
    else:
        element='none'
    return element


dataset=pd.read_csv('domy_wszystko.csv',encoding="utf-8")
# Rooms

dataset['number_rooms']=dataset['number_rooms'].apply(lambda x: int(cut_string(x)))




# Deleting items Dzialka and incorrectly loaded items

dataset=dataset[~dataset.price_per_meter.str.contains("działka")]
dataset=dataset[~dataset.area.str.contains("1 125.86")]  # entire building,not apartament
dataset=dataset[~dataset.area.str.contains("6 081")]     #incorrect square footage area

# Area

dataset['area']=dataset['area'].str.replace(',','.')
dataset['area']=dataset['area'].str.replace('m²','')
dataset['area']=dataset['area'].str.strip()
#correction of one item where area incorrectly calculated on website
dataset['area']=dataset['area'].str.replace('7 797','65')
dataset['area']=dataset['area'].astype(float)

#price_per_meter

dataset['price_per_meter']=dataset['price_per_meter'].str.replace('zł/m²','')
dataset['price_per_meter']=dataset['price_per_meter'].str.replace(' ','')
dataset['price_per_meter']=dataset['price_per_meter'].str.strip()
dataset['price_per_meter']=dataset['price_per_meter'].astype(int)


# apartment price

dataset['price']=(dataset['price_per_meter'])*(dataset['area'])


#Location

dataset=dataset[~dataset.location.str.contains("dolnośląskie")]

dataset['city']=dataset['location'].apply(lambda x: x.split(', ')[0])
dataset['district']=dataset['location'].apply(lambda x: x.split(', ')[1])
dataset['neighbourhood']=dataset['location'].apply(lambda x: extract_third_element(x))


# deleting column

del dataset['location']

dataset=dataset[['area','number_rooms','district','price_per_meter',]]


### Shuffle data
from sklearn.utils import shuffle
dataset = shuffle(dataset)



### Separate label y from features X
X=dataset.iloc[:, :-1].values             #X (independant var) value capital letter,  ':'- means all values
y=dataset.iloc[:,3].values              # y (dependant var) -small letter


from sklearn.preprocessing import LabelEncoder, OneHotEncoder  # import library to change categorical variable into numerical
# Encoding categorical data
labelencoder_X=LabelEncoder()
X[:,2]=labelencoder_X.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features= [2])
X=onehotencoder.fit_transform(X).toarray()


### Standardize features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split  # old versionfrom sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

################################### MODELS ###############################################################

### Logisitic Regression
from sklearn.linear_model import LogisticRegression
classifier_log = LogisticRegression()
classifier_log.fit(X_train,y_train)

### SVM
from sklearn.svm import SVC
classifier_svml = SVC()
classifier_svml.fit(X_train,y_train)

### Kernel SVM
from sklearn.svm import SVC
classifier_svmg = SVC()
classifier_svmg.fit(X_train,y_train)

### Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier() 
classifier_rf.fit(X_train, y_train)

### AdaBoost
from sklearn.ensemble import AdaBoostClassifier
classifier_ada = AdaBoostClassifier()
classifier_ada.fit(X_train, y_train)


#### Naive Bayes - Gaussian
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)


### Gradient Tree Boosting
from sklearn.ensemble import GradientBoostingClassifier
classifier_gtb =  GradientBoostingClassifier()
classifier_gtb.fit(X_train, y_train) 

#  Applying K-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies_log = cross_val_score(estimator = classifier_log, X = X_train, y = y_train, cv = 10, n_jobs = -1)  
accuracies_svml = cross_val_score(estimator = classifier_svml, X = X_train, y = y_train, cv = 10, n_jobs = -1) 
accuracies_svmg = cross_val_score(estimator = classifier_svmg, X = X_train, y = y_train, cv = 10, n_jobs = -1)  
accuracies_ada = cross_val_score(estimator = classifier_ada, X = X_train, y = y_train, cv = 10, n_jobs = -1)
accuracies_nb = cross_val_score(estimator = classifier_nb, X = X_train, y = y_train, cv = 10, n_jobs = -1)
accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train, y = y_train, cv = 10, n_jobs = -1)  
accuracies_gtb = cross_val_score(estimator = classifier_gtb, X = X_train, y = y_train, cv = 10, n_jobs = -1)


print(accuracies_log.mean(), accuracies_log.std())
print(accuracies_svml.mean(), accuracies_svml.std())
print(accuracies_svmg.mean(), accuracies_svmg.std())
print(accuracies_ada.mean(), accuracies_ada.std())
print(accuracies_nb.mean(), accuracies_nb.std())
print(accuracies_rf.mean(), accuracies_rf.std())
print(accuracies_gtb.mean(), accuracies_gtb.std())
print(accuracies_xgb.mean(), accuracies_xgb.std())
print(accuracies_vot.mean(), accuracies_vot.std())






#Predicting the Test results
y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1) ),1))


# saving dataset in xlsx file
 
writer=pd.ExcelWriter('apartaments.xlsx')
dataset.to_excel(writer)
writer.save()






from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


