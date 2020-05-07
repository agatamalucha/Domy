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
dataset=dataset[~(dataset['number_rooms'] >=6)]



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
dataset=dataset[~(dataset['area'] <=25)]
dataset=dataset[~(dataset['area'] >=110)]


#price_per_meter

dataset['price_per_meter']=dataset['price_per_meter'].str.replace('zł/m²','')
dataset['price_per_meter']=dataset['price_per_meter'].str.replace(' ','')
dataset['price_per_meter']=dataset['price_per_meter'].str.strip()
dataset['price_per_meter']=dataset['price_per_meter'].astype(int)
dataset=dataset[~(dataset['price_per_meter'] <=2500)]
dataset=dataset[~(dataset['price_per_meter'] >=10001)]

# apartment price

#dataset['price']=(dataset['price_per_meter'])*(dataset['area'])


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
onehotencoder=OneHotEncoder(categorical_features= [1])
X=onehotencoder.fit_transform(X).toarray()
X = X[:, 2:]     # dummy variable trap exclusion


### Standardize features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


from sklearn.model_selection import train_test_split  # old versionfrom sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)

################################### MODELS ###############################################################

################################### MODELS ###############################################################

### SGDRegressor
from sklearn.linear_model import SGDRegressor
regressor_sgd = SGDRegressor()
regressor_sgd.fit(X_train,y_train)

### BayesianRidge
from sklearn.linear_model import BayesianRidge
regressor_br = BayesianRidge()
regressor_br.fit(X_train,y_train)

### LassoLars
from sklearn.linear_model import LassoLars
regressor_ll = LassoLars()
regressor_ll.fit(X_train,y_train)

from sklearn.linear_model import XGBRegressor  
from xgboost import XGBRegressor
regressor_xgb = XGBRegressor()
regressor_xgb.fit(X_train,y_train)


#  Applying K-fold cross validation

from sklearn.model_selection import cross_val_score

accuracies_sgd = cross_val_score(estimator = regressor_sgd, X = X_train, y = y_train, cv = 10, n_jobs = -1)  
accuracies_br = cross_val_score(estimator = regressor_br, X = X_train, y = y_train, cv = 10, n_jobs = -1) 
accuracies_ll = cross_val_score(estimator = regressor_ll, X = X_train, y = y_train, cv = 5, n_jobs = -1)  
accuracies_xgb = cross_val_score(estimator = regressor_xgb, X = X_train, y = y_train, cv = 10, n_jobs = -1)


print(accuracies_sgd.mean(), accuracies_sgd.std())
print(accuracies_br.mean(), accuracies_br.std())
print(accuracies_ll.mean(), accuracies_ll.std())
print(accuracies_xgb.mean(), accuracies_xgb.std())






#Predicting the Test results
y_pred=regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1) ),1))


# saving dataset in xlsx file
 
writer=pd.ExcelWriter('apartaments.xlsx')
dataset.to_excel(writer)
writer.save()

####################    STATISTICS #############################
dataset=dataset[60< (dataset['area']<80 )]
dataset1=dataset[(dataset['number_rooms'] ==2)]
dataset3=dataset[(dataset['number_rooms'] ==3)]
dataset4=dataset[(dataset['number_rooms'] ==4)]


describe=dataset.describe()
describe_2bedroom=dataset1.describe()
describe_3bedroom=dataset3.describe()
describe_4bedroom=dataset4.describe()