#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from geneticalgorithm import geneticalgorithm as ga
from pyitlib import discrete_random_variable as drv

from leap_ec.simple import ea_solve


dataset = pd.read_csv("data/adult.data2.csv")

# Check for Null Data
dataset.isnull().sum()

# Replace All Null Data in NaN
dataset = dataset.fillna(np.nan)

# Peek at data
dataset.head(2)

# Reformat Column We Are Predicting
dataset['income']=dataset['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})
dataset.head(4)

# Identify Numeric features
numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

# Identify Categorical features
cat_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native']

# Fill Missing Category Entries
dataset["workclass"] = dataset["workclass"].fillna("X")
dataset["occupation"] = dataset["occupation"].fillna("X")
dataset["native.country"] = dataset["native.country"].fillna("United-States")

# Confirm All Missing Data is Handled
dataset.isnull().sum()

# Convert Sex value to 0 and 1
dataset["sex"] = dataset["sex"].map({"Male":0, "Female":1})

# Convert Race value (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black) to numbers
dataset["race"] = dataset["race"].map({"White":0, "Asian-Pac-Islander":1, "Amer-Indian-Eskimo":2, "Other":3, "Black":4})

# Create Married Column - Binary Yes(1) or No(0)
dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
#dataset["marital.status"] = dataset["marital.status"].astype(int)
print(dataset.head())

# Drop the data we don't want to use
dataset.drop(labels=["workclass","education.num","race","marital.status","capital.loss","education","occupation","relationship","native.country","fnlwgt"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(dataset.head())

feature_names=["age","sex","capital.gain","hours.per.week","income"]

# Split-out Validation Dataset and Create Test Variables
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)

results = []
names = []

kfold = KFold(n_splits=10)#, random_state=seed)
#cv_results = cross_val_score(LinearRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
#cv_results = cross_val_score(RandomForestRegressor(), X_train, Y_train, cv=kfold, scoring='accuracy')
cv_results = cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % ("LR", cv_results.mean(), cv_results.std())
print(msg)

# Finalize Model

#Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, Y_train)
predictions = logistic_regression.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

# Prediction
new_record = [[35, 1, 5000, 40]]
print(logistic_regression.predict(new_record)[0])
