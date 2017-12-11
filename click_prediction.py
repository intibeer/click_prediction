#!/usr/bin/env python
#Example of code of adverting click predictions with Naive Bayes and Logistic Regression, author: Inti Beer. 
#For data contact: intibeer@googlemail.com
import re
import os
import csv
import sys
import argparse
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing 
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
#Prepare dictionary vectorizer
v = DictVectorizer(sparse=True)

def prepare_data():
	#Load Data and split 
	X_train = pd.read_csv('train.csv')
	X_val = pd.read_csv('validation.csv')
	y_train = X_train['click']
	y_val = X_val['click']
	print("Finshed importing training data.")
	return X_train, X_val, y_train, y_val

def process_train(df):
	#This function drops unused data and vectorizes remaining data in the dataframe.
    drop = ['logtype', 'bidid', 'url', 'payprice', 'bidprice', 'keypage', 'urlid', 'click']
    df.drop(drop,inplace=True, axis=1)
    df = df[['weekday', 'hour', 'region', 'adexchange','advertiser','slotvisibility']]
    df = df.to_dict(orient='records')
    X = v.fit_transform
    return X

def scores(y_true, y_pred):
	#Prints two binary classifaction evaluation scores:-
	#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
	#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

    print("AUC: " + str(roc_auc_score(y_true, y_pred)))
    print("Cross Entropy" + str(log_loss(y_true, y_pred)))
    print('Completed.')

def RunNaiveBayes(y_val, X_train, X_val, y_train):
	
	#Vectorize and train
	print("Started encoding x training data...")
	X = process_train(X_train)
	print("Done encoding x training data...")
	
	#Train model
	print("Training model...")
	clf = GaussianNB()
	clf.fit(X, y_train)

	#Generate predictions
	expected = y_val
	x_val = process(X_val)
	predicted = predict_proba(x_val)
	prediction = pd.DataFrame(predicted)
	prediction.to_csv('NaiveBayes_Predictions.csv')

	#Read predictions from file
	y_pred = pd.read_csv('NaiveBayes_1.csv') 
	y_pred = y_pred[' probability_click']
	y_val = y_val.astype('float')
	y_pred =y_pred.astype('float')

	# Generate Confusion matrix: metrics.confusion_matrix(expected, predicted)

	#Output model scores
	return scores(y_val, y_pred)

def RunLogisticRegression(y_val, X_train, X_val, y_train):
	#Vectorize and train
	print("Started encoding x training data...")
	X = process_train(X_train)
	print("Done encoding x training data...")

	#Train model
	print("Training model...")
	model = LogisticRegression()
	model.fit(X, y_train)

	# Make Predictions
	print("making predictions...")
	expected = y_val
	X_v = process(X_val)
	predicted = model.predict(X_v)
	prediction.to_csv('LogisticRegression_Predictions.csv')

	#Read predictions from file
	y_pred = pd.read_csv('LogisticRegression_Predictions.csv') 
	y_pred = y_pred[' probability_click']
	y_val = y_val.astype('float')
	y_pred =y_pred.astype('float')

	# Generate Confusion matrix: metrics.confusion_matrix(expected, predicted)

	#Output model scores
	return scores(y_val, y_pred)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("model", help="Logistic or NaiveBayes")
	args = parser.parse_args()

	if args.model == "Logistic":
		y_val, X_train, X_val, y_train = prepare_data()
		RunLogisticRegression(y_val, X_train, X_val, y_train)

	elif args.model == "NaiveBayes":
		y_val, X_train, X_val, y_train = prepare_data()
		RunNaiveBayes(y_val, X_train, X_val, y_train)
	else:
		return "Invalid argument, please specify Logistic or NaiveBayes"





