# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 18:54:30 2018

@author: jamespatten
"""

# =============================================================================
# Import Libs
# =============================================================================

import pandas as pd
import numpy as np
from numpy import array
from sklearn.externals import joblib

#Viz
import matplotlib.pyplot as plt
import seaborn as sns

#preprocessing
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from numpy import sort
from sklearn.preprocessing import Imputer

#Modelling
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

# =============================================================================
# Preprocessing
# =============================================================================
df = pd.read_csv('events.csv')
df = df.loc[df['event_team'] == 'Arsenal']
df = df.loc[df['event_type'] == 1]
df = df.drop(['id_odsp','id_event','sort_order','text','event_type','event_type2','event_team','player_in','player_out','shot_outcome','fast_break','player2'],axis = 1)
df = pd.get_dummies(df, columns=["player"])
df = pd.get_dummies(df, columns=["opponent"])

imp=Imputer(missing_values="NaN", strategy="mean" )
df["shot_place"]=imp.fit_transform(df[["shot_place"]]).ravel()

X = df.drop('is_goal', axis=1)
y = df["is_goal"]


# =============================================================================
# Feature Selection: 
# =============================================================================

#Fit untuned model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
xg = XGBClassifier()
xg.fit(X_train,y_train)


thresholds = sort(xg.feature_importances_) 
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(xg, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))   
    #best to use only top 6 features

#identify what the top 6 features are 
importances = xg.feature_importances_
indices = np.argsort(importances)[::-1]

d = {key: value for (key, value) in enumerate(X.columns.values)}
#name_indices = array([index for index, value in enumerate(X.columns.values)])
feature_names_ranked = np.vectorize(d.get)(indices)

for f in range(X.shape[1]):
    print("Feature: " +  feature_names_ranked[f] + " (" +  str(importances[indices[f]]) + ")")

#Manually reducing features
X = df[['time','shot_place','location','assist_method','bodypart','situation']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# =============================================================================
# Hyperparameter Tuning: Bayesian Optimisation (TPE) - Gini Coeff
# =============================================================================

def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)

def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)

def objective(params):
    params = {
        'max_depth': int(params['max_depth']),
        'gamma': "{:.3f}".format(params['gamma']),
        'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
    }
    
    xg = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        n_jobs=4,
        **params
    )
    
    score = cross_val_score(xg, X, y, scoring=gini_scorer, cv=StratifiedKFold()).mean()
    print("Gini {:.3f} params {}".format(score, params))
    return score

space = {
    'max_depth': hp.quniform('max_depth', 2, 8, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
    'gamma': hp.uniform('gamma', 0.0, 0.5),
}

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50)

print("Hyperopt estimated optimum {}".format(best))

for key, value in best.items():
    best['max_depth'] = int(best['max_depth'])

XGBgoal = XGBClassifier(**best)
XGBgoal.fit(X_train,y_train)

# =============================================================================
# Serialise Tuned Model
# =============================================================================

joblib.dump(XGBgoal, 'XGBgoal.pkl')
print("Model dumped!")

# Saving the data columns from training
XGBgoal_columns = list(X.columns)
joblib.dump(XGBgoal_columns, 'XGBgoal_columns.pkl')