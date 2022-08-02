'''logistic regression'''

import numpy as np
import pandas as pd
import joblib
from sklearn import metrics
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_train)
print(classifier.predict([[30,87000]]))
joblib.dump(classifier,'classifier_joblib.sav')


metrics.accuracy_score(y_train, y_pred)
acc=(len(y_pred)/len(y_train))
print(acc)