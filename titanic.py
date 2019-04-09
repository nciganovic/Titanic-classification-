"""
Created on Fri Apr  5 17:58:30 2019

@author: Nikola Ciganovic
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

#LOAD DATA 
csv_train = r'''C:\Users\pc_user\titanic\train.csv'''
csv_test = r'''C:\Users\pc_user\titanic\test.csv'''
csv_gender = r'''C:\Users\pc_user\titanic\gender_submission.csv'''

train_data = pd.read_csv(csv_train)
test_data = pd.read_csv(csv_test)

#Creating y data (true or false if human survived) 
survived = (train_data["Survived"] == 1)

#REMOVING DATA
train_data = train_data.drop("Name", 1)
train_data = train_data.drop("Ticket", 1)
train_data = train_data.drop("Cabin", 1)
train_data = train_data.drop("PassengerId", 1)
train_data = train_data.drop("Survived", 1)
#train_data = train_data.drop("Fare", 1)
#train_data = train_data.drop("Embarked", 1)

#REPLACING DATA
train_data["Sex"] = train_data["Sex"].replace('male', "1", regex=True)
train_data["Sex"] = train_data["Sex"].replace('fe1', "0", regex=True)

train_data["Embarked"] = train_data["Embarked"].replace('Q', "0", regex=True)
train_data["Embarked"] = train_data["Embarked"].replace('S', "1", regex=True)
train_data["Embarked"] = train_data["Embarked"].replace('C', "2", regex=True)

#info on data
data_info = train_data.info()

#REPLACING NULL WIHT MEDIAN 
from sklearn.impute import SimpleImputer

sample_incomplete_rows = train_data[train_data.isnull().any(axis=1)].head()

imputer = SimpleImputer(strategy="median")

imputer.fit(train_data)
X = imputer.transform(train_data)
train_data_full = pd.DataFrame(X, columns=train_data.columns)

#BINARY CLASSIFIER

from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(train_data_full, survived)

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, train_data_full, survived, cv=10)
print(svm_scores.mean())

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, train_data_full, survived, cv=10)
print(forest_scores)
print(forest_scores.mean())

plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()
