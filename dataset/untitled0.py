# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:50:31 2023

@author: muthu
"""

# Import the necessary libraries 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the dataset 
data = pd.read_csv('heart')
# Split the dataset into features and labels
X	= data.iloc[:, :-1]
Y	= data.iloc[:, -1]
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.2, random_state=42)
# Create a Random Forest model model 
model=RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on the training set 
model.fit(X_train, Y_train)
# Make predictions on the testing set 
Y_pred = model.predict(X_test)
# Evaluate the performance of the model 
accuracy = accuracy_score(Y_test, Y_pred) 
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Accuracy:", accuracy) 
print("Confusion matrix:", conf_matrix)
 

