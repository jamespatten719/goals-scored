# -*- coding: utf-8 -*-
"""

@author: jamespatten
EDA inspiration from angps95
Hyperopt implementation from eikedehling

"""
# =============================================================================
# Import Libs
# =============================================================================
import pandas as pd
import numpy as np
from numpy  import array
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from numpy import sort
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from scipy import statsa
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import Imputer


#Bayesian Optimisation
from hyperopt import hp, tpe
from hyperopt.fmin import fmin

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
df = pd.read_csv('events.csv')
df=df[df["is_goal"]==1]
df = df.loc[df['event_team'] == 'Arsenal']

#Standard Correlation Heatmap
corrmat = df.corr() #correlation matrix heatmap
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


#Goals in each minute
fig=plt.figure(figsize=(10,8))
plt.hist(df.time,width=1,bins=100,color="blue")   #100 so 1 bar per minute
plt.xlabel("Min")
plt.ylabel("Goal Count")

#Home/Away df
fig=plt.figure(figsize=(10,8))
plt.hist(df[df["side"]==1]["time"],width=1,bins=100,color="cyan",label="home df")   
plt.hist(df[df["side"]==2]["time"],width=1,bins=100,color="grey",label="away df") 
plt.xlabel("Min")
plt.ylabel("Goal Count")
plt.legend()

#Goals by shot location
diff_angle_df=sum(df["location"]==6)+sum(df["location"]==7)+sum(df["location"]==8)
long_range_df=sum(df["location"]==16)+sum(df["location"]==17)+sum(df["location"]==18)
box_df=sum(df["location"]==3)+sum(df["location"]==9)+sum(df["location"]==11)+sum(df["location"]==15)
close_range_df=sum(df["location"]==10)+sum(df["location"]==12)+sum(df["location"]==13)
penalties=sum(df["location"]==14)
not_recorded=sum(df["location"]==19)

labels=["Long Range Goals","Difficult angle Goals","Goals from around the box","Close range Goals","Penalties","Not recorded"]
sizes=[long_range_df,diff_angle_df,box_df,close_range_df,penalties,not_recorded]
colors=["gray","yellow","aqua","coral","red","violet"]
plt.pie(sizes,colors=colors,autopct='%1.1f%%',startangle=60,pctdistance=0.8,radius=3)
plt.axis('equal')
plt.title("Percentage of each location for df",fontname="Times New Roman Bold",fontsize=18,fontweight="bold")
plt.legend(labels)

# =============================================================================
# Modelling for Arsenal Goals Scored 
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

#initial XGB test using scikit learn tools
xg = XGBClassifier()
xg.fit(X_train,y_train)
xg.predict(X_test)
xg_score = xg.score(X_test,y_test)
print(xg_score) #0.9183673469387755
# fit model no training data

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_score = rf.score(X_test, y_test) #0.8877551020408163
print(rf_score)

# =============================================================================
# Random Forrest Model Tuning 
# ============================================================================

#Ranking feature importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

d = {key: value for (key, value) in enumerate(X.columns.values)}
#name_indices = array([index for index, value in enumerate(X.columns.values)])
feature_names_ranked = np.vectorize(d.get)(indices)

for f in range(X.shape[1]):
    print("Feature: " +  feature_names_ranked[f] + " (" +  str(importances[indices[f]]) + ")")

# Plot the feature importances of the CV_rf
plt.figure(figsize=(10,8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

#Re-running model without least import features - manual feature selection
df = df.drop(['player_jack wilshere', 'opponent_Cardiff',
       'opponent_Manchester Utd', 'opponent_Fulham',
       'player_laurent koscielny', 'player_tomas rosicky',
       'player_alex iwobi', 'player_jack cork', 'player_nicklas bendtner',
       'player_hector bellerin', 'player_carl jenkinson',
       'player_gabriel paulista', 'player_kieran gibbs',
       'player_kyle naughton', 'player_per mertesacker',
       'player_bacary sagna', 'player_mathieu flamini',
       'player_joel campbell', 'player_kevin wimmer',
       'player_granit xhaka', 'player_francis coquelin',
       'player_chuba akpom', 'player_nacho monreal',
       'player_shkodran mustafi', 'player_thomas vermaelen',
       'player_mikel arteta', 'opponent_Middlesbrough',
       'player_mathieu debuchy', 'player_serge gnabry',
       'player_calum chambers', 'player_yaya sanogo',
       'player_kim kallstrom', 'player_mohamed elneny',
       'player_lucas perez', 'player_aly cissokho',
       'player_tyler blackett', 'player_damien delaney',
       'player_fabricio coloccini', 'player_gabriel', 'player_mark bunn'],axis=1)


#Grid Search for Param Tuninag
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10],
    'criterion' :['gini', 'entropy']
}

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)
CV_rf.best_params_
y_pred = CV_rf.predict(X_test)
CV_rf_score = rf.score(X_test, y_test)
print(CV_rf_score) #not much increase in R2 score

# =============================================================================
# Random Forrest Model Evaluation
# =============================================================================

#k fold accuracy
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
def accuracy_score(model):
    return np.mean(cross_val_score(model,X_train,y_train,cv=k_fold,scoring="accuracy"))


#confusion matrix to assess preciesion/recall
def confusion_matrix_model(model):
    cm=confusion_matrix(y_train,model.predict(X_train))
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Goals", "Predicted Misses"]
    cm.index=["Actual Goals", "Actual Misses"]
    return cm

confusion_matrix_model(rf)
confusion_matrix_model(CV_rf)
accuracy_score(CV_rf)

#Kappa Score
kappa_score = cohen_kappa_score(y_test, y_pred, labels=None, weights=None)
print(kappa_score) #0.0782178217821784 - very low Kappa despite high R2

#Receiver Operating Characteristic
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr) #0.6422309883848346 - quite low

#Plot of AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#F1 score
f1_score(y_test, y_pred, average='weighted') #0.8708086355973681 - strong 

# =============================================================================
# XGB Tuning using test error mean and AUC
# =============================================================================
#reload df
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

#Initialise Parameters
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

#Tune max_depth and min_child_weight
param_xgb1 = {
 'max_depth':[4,5,6],
 'min_child_weight':[4,5,6]
}

CV_xgb1 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_xgb1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

CV_xgb1.fit(X_train, y_train)
CV_xgb1.grid_scores_, CV_xgb1.best_params_, CV_xgb1.best_score_

#Tuning Gamma
param_xgb2= {
 'gamma':[i/10.0 for i in range(0,5)]
}
CV_xgb2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=140, max_depth=4,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_xgb2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

CV_xgb2.fit(X_train, y_train)
CV_xgb2.grid_scores_, CV_xgb2.best_params_, CV_xgb2.best_score_

#Tuning Regularisation Parameters
param_xgb3 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
CV_xgb3= GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=177, max_depth=4,
 min_child_weight=6, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_xgb3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

CV_xgb3.fit(X_train, y_train)
CV_xgb3.grid_scores_, CV_xgb3.best_params_, CV_xgb3.best_score_
# AUC 0.9337659622950121

# =============================================================================
# Bayesian Optimisation with hyperopt (using Tree-structured Parzen Estimator)
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
            max_evals=10)

print("Hyperopt estimated optimum {}".format(best))

# =============================================================================
# XGB Feature Selection
# =============================================================================
CV_xgb3 = CV_xgb3.best_estimator_

thresholds = sort(CV_xgb3.feature_importances_) #
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(CV_xgb3, threshold=thresh, prefit=True)
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
    
#mapping the most important features to feature names
importances = CV_xgb3.feature_importances_
indices = np.argsort(importances)[::-1]

d = {key: value for (key, value) in enumerate(X.columns.values)}
#name_indices = array([index for index, value in enumerate(X.columns.values)])
feature_names_ranked = np.vectorize(d.get)(indices)

for f in range(X.shape[1]):
    print("Feature: " +  feature_names_ranked[f] + " (" +  str(importances[indices[f]]) + ")")

# Plot the feature importances of the CV_xgb3
plt.figure(figsize=(10,8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

#check if same 6 features are selected through ANOVA F and Chi2
#X_chi2 = SelectKBest(chi2, k=6).fit_transform(X, y)
#X_f = SelectKBest(f_classif, k=6).fit_transform(X, y)

##F-test of significance between full and reduced model 
#X_reduced = df[['time','shot_place','location','assist_method','bodypart','situation']]
#X_trainR, X_testR, y_trainR, y_testR = train_test_split(X_reduced, y, test_size = 0.1)
#xgbR = CV_xgb3.fit(X_trainR,y_trainR)
#
#y_pred = CV_xgb3.predict(X_test)
#y_pred = y_pred.tolist()
#y_predR = (CV_xgb3.predict(X_testR)).tolist()
#
#F = np.var(y_predR) / np.var(y_pred)
#df1 = len(y_predR) - 1
#df2 = len(y_pred) - 1
##alpha = 0.05
#p_value = stats.f.cdf(F, df1, df2) #  0.967849254975412
#Insufficient evidence to reject null hypothesis that reduced model is significantly different to full model 

# =============================================================================
# Vizualising Model 
# =============================================================================

#Convert booster into an XGBModel instance

#Initial accuracy check
xg_matrix= xgb.DMatrix(data=X, label=y)
params = {"objective":"binary:logistic", "max_depth":4,"learning_rate":0.1,"n_estimators":177,
          "min_child_weight":6,"gamma":0.2, "subsample":0.8, "colsample_bytree":0.8,
          "nthread":4, "scale_pos_weight":1} 
#train model
CV_xgb4 = xgb.train(params=params, dtrain=xg_matrix, num_boost_round=10)

#cross val
cv_results = xgb.cv(dtrain=xg_matrix, params=params, nfold=3, num_boost_round=5, metrics="error", as_pandas=True, seed=123)
# Accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1])) #0.8993413333333333

#Visualise XGBoost trees - requires GraphViz executables
# Plot the first tree

# Plot the last tree sideways
xgb.plot_tree(CV_xgb4, num_trees=9, rankdir='LR')
plt.show()

