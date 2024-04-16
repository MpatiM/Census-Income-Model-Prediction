# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 20:03:05 2023

@author: Kanna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import sklearn.preprocessing as preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import plot_roc_curve, auc


## Load Data Census Income
path = "C:/Users/Kanna/Documents/Rutgers Grad/Spring 2023/BINF5125 MLT/Project Census Income/"

# Column names to be added
column_names=["age", "workclass", "fnlwgt", "education", "education_num", 
              "marital_status", "occupation", "relationship", "race",
              "sex", "capital_gain", "capital_loss", "hours_per_week", 
              "native_country", "income_grp"]

# Add column names while reading a CSV file
ci = pd.read_csv(path+"census_income_data.csv", names = column_names)
                                                                                 
print(ci)
print(ci.head(10))

## Data Cleaning: Pre-processing and Feature Selection

# Feature Selection 
# Redundant education information
ci['education_num'].unique()
ci['education'].unique()

# Drop 'education' categorical column

# Will drop column 'fnlwgt' 
# since this column is the number of people census believes entry represents.
# This does not influence the outcome prediction.

ci = ci.drop(columns=['education', 'fnlwgt'])

print(ci)

# Replace missing value '?' with sentinel value NaN

# String columns only
str_col = ["workclass", "marital_status", "occupation", "relationship", "race",
            "sex", "native_country"]

# # Replace the values with NaN
# ci[str_col] = ci[str_col].replace(r'(\?)', np.nan, regex=True) 
# ci[str_col].head(20)
# ci.head(20)

# # Check to see where null exists
# ci.isnull().any()
# ci.isnull().sum()

#### OR ####

# Replace the values with 'Other'
ci[str_col] = ci[str_col].replace(r'(\?)', "Other", regex=True) 
ci[str_col].head(20)
ci.head(20)

# Look at unique values in all columns 
for col in ci:
    print("Column", col, ":\n", ci[col].unique()) 

# Data Pre-processing
ci.columns

cont_cols=list(ci[['age', 'capital_gain', 'capital_loss', 'hours_per_week']])
cat_cols = list(set(ci.columns) - set(cont_cols))

print("Continuous Features:\n", cont_cols)
print("Categorical Features:\n", cat_cols)

# Ordinal encoder of Categorical Vars
ordinal_encoder = preprocessing.OrdinalEncoder()

# Perform standardization of categorical columns to convert to numeric values
ci[cat_cols] = ordinal_encoder.fit_transform(ci[cat_cols])

ci[cat_cols]

# Standard Scalar of Continuous Vars
scaler = preprocessing.StandardScaler()

# performs standardization on the numeric_cols of df to return the new array X_numeric_scaled. 
ci[cont_cols] = scaler.fit_transform(ci[cont_cols])

ci[cont_cols]

# Print dataset after standardization
print(ci.head(20))

# Separate features and outcome 'income_grp'
fts = ci.loc[:, ci.columns != 'income_grp']
fts

tgt = ci['income_grp']
tgt

# Check unique values of target outcome to ensure it is correct
tgt.unique()

# Do the same for features
for col in fts:
    print("Column", col, ":\n", fts[col].unique()) 
    

## Splitting the data into train and test sets (80/20)
x_train, x_test, y_train, y_test = train_test_split(fts, tgt, 
                                                    test_size=0.20, 
                                                    random_state=42)

## Perform classification using models LogisticRegression and SVM kernels using all features

# Logistic Regression
clf = make_pipeline(preprocessing.StandardScaler(),
                     LogisticRegressionCV(random_state=0, 
                                          solver='lbfgs', 
                                          multi_class='multinomial'))

clf = clf.fit(x_train, y_train)
print(clf) 

# Evaluate the model
print("Logistic Regression Model with Cross Validation predicted probability score: {}"
      .format(clf.score(x_test, y_test)))

# ROC Curve
print("Logistic Regression")
plot_roc_curve(clf, x_test, y_test)

# Define SVM for all three kernel (using C=1 and gamma=1/2)
## Linear
svc_linear = svm.SVC(kernel='linear', C=1, gamma=1/2)
svc_linear.fit(x_train, y_train)

## Polynomial
#svc_poly = svm.SVC(kernel='poly', C=1,gamma= 1/2)
#svc_poly.fit(x_train, y_train)

## RBF
svc_rbf = svm.SVC(kernel='rbf', C=1,gamma= 1/2)
svc_rbf.fit(x_train, y_train)

# Predict the score on train data for all three kernel
svc_linear.score(x_train, y_train)                                                                        
#svc_poly.score(x_train, y_train)                                                                        
svc_rbf.score(x_train, y_train)  

# Predict the score on test data for all three kernel
svc_linear.score(x_test, y_test)                                                                        
#svc_poly.score(x_test, y_test)                                                                        
svc_rbf.score(x_test, y_test)   
    
# ROC Curve
print("SVM Linear Kernel")
plot_roc_curve(svc_linear, x_test, y_test)

# print("SVM Polynomial Kernel")
# plot_roc_curve(svc_poly, x_test, y_test)

print("SVM RBF Kernel")
plot_roc_curve(svc_rbf, x_test, y_test)

# Prediction Score Results for all 4 classification models
results1 = pd.DataFrame({'PredScore_80_20': [clf.score(x_test, y_test),
                                             svc_linear.score(x_test, y_test),
                                             # svc_poly.score(x_test, y_test),
                                             svc_rbf.score(x_test, y_test)]}, 
                        index=['LogisticRegression', 
                               'SVC_Linear', 
                                #'SVC_Polynomial', 
                               'SVC_RBF'])
results1

## Splitting the data into train and test sets (70/30)
x_train, x_test, y_train, y_test = train_test_split(fts, tgt, 
                                                    test_size=0.30, 
                                                    random_state=42)

## Perform classification using models LogisticRegression and SVM kernels using all features

# Logistic Regression
clf = make_pipeline(preprocessing.StandardScaler(),
                     LogisticRegressionCV(random_state=0, 
                                          solver='lbfgs', 
                                          multi_class='multinomial'))

clf1 = clf.fit(x_train, y_train)
print(clf) 

# Evaluate the model
print("Logistic Regression Model with Cross Validation predicted probability score: {}"
      .format(clf1.score(x_test, y_test)))

clf_score1 = clf1.score(x_test, y_test)

# ROC Curve
print("Logistic Regression")
plot_roc_curve(clf1, x_test, y_test)

# Define SVM for all three kernel (using C=1 and gamma=1/2)
## Linear
svc_linear1 = svm.SVC(kernel='linear', C=1, gamma=1/2)
svc_linear1.fit(x_train, y_train)

## Polynomial
#svc_poly = svm.SVC(kernel='poly', C=1,gamma= 1/2)
#svc_poly.fit(x_train, y_train)

## RBF
svc_rbf1 = svm.SVC(kernel='rbf', C=1,gamma= 1/2)
svc_rbf1.fit(x_train, y_train)

# Predict the score on train data for all three kernel
svc_linear1.score(x_train, y_train)                                                                        
#svc_poly.score(x_train, y_train)                                                                        
svc_rbf1.score(x_train, y_train)  

# Predict the score on test data for all three kernel
linear_score1 = svc_linear1.score(x_test, y_test)                                                                        
#svc_poly.score(x_test, y_test)                                                                        
rbf_score1 = svc_rbf1.score(x_test, y_test)   
    
# ROC Curve
print("SVM Linear Kernel")
plot_roc_curve(svc_linear1, x_test, y_test)

# print("SVM Polynomial Kernel")
# plot_roc_curve(svc_poly, x_test, y_test)

print("SVM RBF Kernel")
plot_roc_curve(svc_rbf1, x_test, y_test)

# Adding Prediction Scores to results table for comparison between 80/20 split and 70/30 pslit
results1['PredScore_70_30'] = pd.Series([clf_score1,
                                         linear_score1,
                                         # svc_poly.score(x_test, y_test),
                                         rbf_score1],
                                        index=['LogisticRegression', 
                                               'SVC_Linear', 
                                                #'SVC_Polynomial', 
                                               'SVC_RBF'])

results1

## Since the prediction scores do not change or show a difference between the two splitted train/test datasets
## We select the best features

## Lasso-Regression model method

# Perform GridSearchCV to tune best-fit LR model
#param = {'C': [10**-2,10**-1,10**0,10**1,10**2]}
#param = {'C': [10**-3,10**-2,10**0,10**2,10**3]}
param = {'C': [10**-4,10**-3,10**0,10**3,10**4]}

lr_model = LogisticRegression(penalty='l1', solver='liblinear')
gs_model = GridSearchCV(estimator=lr_model, param_grid=param)
gs_model.fit(x_train, y_train)

# Train a LR model with best parameters
model = LogisticRegression(**gs_model.best_params_, penalty='l1', solver='liblinear')
model.fit(x_train, y_train)

# Compute coefficients of Logistic Regression Model
coef = model.coef_[0]
coef

## Recursive Feature Elimination (RFE)

from sklearn.feature_selection import RFE

# Features and Target
fts
tgt

# Create a logistic regression model
model = LogisticRegression()

# Use RFE to select the top 10 features
rfe = RFE(model, n_features_to_select=8)
rfe.fit(fts, tgt)

# Print the selected features
print(rfe.support_)
    # [ True False  True  True False  True False  True  True  True  True False]
fts.columns
    # Based on RFE selection, the following are the top 8 selected features:
    # age, edeucation_num, marital_status, relationship, sex, capital_gain, cpaital_loss, hours_per_week
    
# Using the selected features, train the model
imp_features = pd.Series(fts.columns)[list(rfe.support_)]
imp_features    

sel_fts = fts[imp_features]
sel_fts

# Split the data with selected 8 features (70/30)
x_train, x_test, y_train, y_test = train_test_split(sel_fts, tgt, 
                                                    test_size=0.30, 
                                                    random_state=42)

## Perform classification using models LogisticRegression and SVM kernels using all features

# Logistic Regression
clf = make_pipeline(preprocessing.StandardScaler(),
                     LogisticRegressionCV(random_state=0, 
                                          solver='lbfgs', 
                                          multi_class='multinomial'))

clf2 = clf.fit(x_train, y_train)
print(clf) 

# Evaluate the model
print("Logistic Regression Model with Cross Validation predicted probability score: {}"
      .format(clf2.score(x_test, y_test)))

clf_score2 = clf2.score(x_test, y_test)

# ROC Curve
print("Logistic Regression")
plot_roc_curve(clf2, x_test, y_test)

# Define SVM for all three kernel (using C=1 and gamma=1/2)
## Linear
svc_linear2 = svm.SVC(kernel='linear', C=1, gamma=1/2)
svc_linear2.fit(x_train, y_train)

## Polynomial
#svc_poly = svm.SVC(kernel='poly', C=1,gamma= 1/2)
#svc_poly.fit(x_train, y_train)

## RBF
svc_rbf2 = svm.SVC(kernel='rbf', C=1,gamma= 1/2)
svc_rbf2.fit(x_train, y_train)

# Predict the score on train data for all three kernel
svc_linear2.score(x_train, y_train)                                                                        
#svc_poly.score(x_train, y_train)                                                                        
svc_rbf2.score(x_train, y_train)  

# Predict the score on test data for all three kernel
linear_score2 = svc_linear2.score(x_test, y_test)                                                                        
#svc_poly.score(x_test, y_test)                                                                        
rbf_score2 = svc_rbf2.score(x_test, y_test)   
    
# ROC Curve
print("SVM Linear Kernel")
plot_roc_curve(svc_linear2, x_test, y_test)

# print("SVM Polynomial Kernel")
# plot_roc_curve(svc_poly, x_test, y_test)

print("SVM RBF Kernel")
plot_roc_curve(svc_rbf2, x_test, y_test)

results1['RFE_Sel_70_30'] = pd.Series([clf_score2,
                                         linear_score2,
                                         # svc_poly.score(x_test, y_test),
                                         rbf_score2],
                                        index=['LogisticRegression', 
                                               'SVC_Linear', 
                                                #'SVC_Polynomial', 
                                               'SVC_RBF'])

results1

## Create a bar chart for visual comparison of prediction scores across all three models
#list(results1.columns.values)
methods = results1.columns.values.tolist()
x = np.arange(3)

#list(results1[0:1].values)

results1[0:1].values.tolist()

# lr = list(results1[0:1].values)
# svc_ln = list(results1[1:2].values)
# svc_r = list(results1[2:].values)

lr = results1[0:1].values.tolist()
svc_ln = results1[1:2].values.tolist()
svc_r = results1[2:].values.tolist()


width = 0.20

a = plt.figure()
ax = a.add_axes([0,0,1,1])
ax.set_title("Comparison of Prediction Scores")
ax.bar(x+0.1, lr, width)
ax.bar(x, svc_ln, width)
ax.bar(x-0.1, svc_r, width)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.legend(['LinearRegression', 'SVC_Linear', 'SVC_RBF'])
          # loc = 'lower center', 
          # bbox_to_anchor =(0.5,-0.15), ncol=2)

plt.yticks(x, methods)
plt.show()

#####################

methods = ['PredScore_80_20', 'PredScore_70_30', 'RFE_Sel_70_30']
x = np.arange(len(methods))

lr = [0.8258866881621373, 0.8263895997543249, 0.826696693622684]
svc_ln = [0.8165208045447566, 0.8209642747466476, 0.8200429931415703]
svc_r = [0.8384768923691079, 0.8334527587265841, 0.8469648889343843]


width = 0.20

a = plt.figure()
ax = a.add_axes([0,0,1,1])
ax.set_title("Comparison of Prediction Scores")
ax.bar(x-0.2, lr, width)
ax.bar(x, svc_ln, width)
ax.bar(x+0.2, svc_r, width)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
ax.legend(['LinearRegression', 'SVC_Linear', 'SVC_RBF'],
          loc = 'lower center', 
          bbox_to_anchor =(1.0,1.0))
ax.set_ylabel("Prediction Scores")
ax.set_xlabel("Methods")

plt.xticks(x, methods)
plt.show()

    # Based on the table and graph, SVC_RBF performs the best in predicting the outcome variable.
    # Out of the three methods performed, when splitting the train/test by 70-30, selecting 8 features using
    # Recursive Feature Elimination (RFE) feature selection technique, this had performed better in comparison
    # to the other fitted models.