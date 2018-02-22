import pandas as pd 
import numpy as np  
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
ffj2017 = "/Users/christine/Desktop/python/jan2017.csv"
# f2017 = "/Users/christine/Desktop/python/dffeb2017.csv"

db = pd.read_csv(j2017, low_memory=False)
# de = pd.read_csv(f2017, low_memory=False)

a = db['ARR_DELAY_NEW']
# a = a.clip(upper=2)
# a = a.clip(lower=1) 
# a = a.fillna(1).astype(int)


a = a.fillna(0).astype(int)
idx = 0
for val in np.nditer(a):
	if val <= 15 and val>0:
		a[idx] = 15
	if val<=30 and val>15:
		a[idx] = 30
	if val<=45 and val>30:
		a[idx] = 45
	if val>45:
		a[idx] = 46
	idx = idx+1


keep_col = ['MONTH', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DISTANCE']
d = db[keep_col]
ab = pd.concat([d,a], axis=1)
ab['DISTANCE'] = ab['DISTANCE'].astype(int)
ab['ARR_DELAY_NEW'] = ab['ARR_DELAY_NEW'].astype(int)

array = ab.values     
X = array[:,0:5]
Y = array[:,5] 
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
results = []
names = ['MONTH', 'AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DISTANCE', 'ARR_DELAY_NEW']
seed = 7
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



lr = LogisticRegression() 
lr.fit(X_train, Y_train) 
predictions = lr.predict(X_validation)



# LR: 0.635191 (0.002719)
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# /Users/christine/anaconda3/lib/python3.6/site-packages/sklearn/discriminant_analysis.py:388: UserWarning: Variables are collinear.
#   warnings.warn("Variables are collinear.")
# LDA: 0.635191 (0.002719)
# KNN: 0.596370 (0.002657)
# CART: 0.642621 (0.003097)
# NB: 0.634338 (0.002740)



# e = de[keep_col]
# a = d.head(50)
# b = e.head(50)
# b = b.reset_index(drop=True)
# t = pd.concat([a,b], axis = 0, ignore_index=True)
# scatter_matrix(t)
# plt.show()


