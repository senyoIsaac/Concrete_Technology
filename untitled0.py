# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 12:39:37 2023

@author: Winslow
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
df = pd.read_table("C:\\Users\Winslow\Desktop\Concrete Technology\concrete_data.txt")

feature = ['cement','blast_furnace_slag','fly_ash','water','superplasticizer','coarse_aggregate','age','concrete_compressive_strength','age']


X = df[feature]
y = df['concrete_compressive_strength']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=(0), train_size=0.6)
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)

#Object of models
decisionModel = DecisionTreeRegressor()
decisionModel.fit(X, y)

print(decisionModel.score(X_train, y_test))