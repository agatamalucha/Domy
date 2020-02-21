# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:52:12 2020

@author: Agatka
"""

import pandas as pd


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

dataset=pd.read_csv('domy.csv',encoding="utf-8")


# Area


dataset['area']=dataset['area'].str.replace(',','.')
dataset['area']=dataset['area'].str.replace('m²','')
dataset['area']=dataset['area'].str.strip()
dataset['area']=dataset['area'].astype(float)

#location

dataset=dataset[~dataset.location.str.contains("dolnośląskie")]
dataset['city']=dataset['location'].apply(lambda x: x.split(', ')[0])
dataset['district']=dataset['location'].apply(lambda x: x.split(', ')[1])
dataset['neighbourhood']=dataset['location'].apply(lambda x: extract_third_element(x))

# Rooms

dataset['number_rooms']=dataset['number_rooms'].apply(lambda x: int(cut_string(x)))


#price_per_meter

dataset=dataset[~dataset.price_per_meter.str.contains("działka")]

dataset['price_per_meter']=dataset['price_per_meter'].str.replace('zł/m²','')
dataset['price_per_meter']=dataset['price_per_meter'].str.replace(' ','')
dataset['price_per_meter']=dataset['price_per_meter'].str.strip()
dataset['price_per_meter']=dataset['price_per_meter'].astype(int)

# deleting column

del dataset['location']


