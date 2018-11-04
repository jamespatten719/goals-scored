# -*- coding: utf-8 -*-
"""
@author: jamespatten
EDA inspiration from angps95

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
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score

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
df = pd.get_dummies(df, columns=["opponent"])
df = pd.get_dummies(df, columns=["player"])
df = df.dropna()

X = df.drop('is_goal', axis=1)
y = df["is_goal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf.predict(X_test)
rf_score = rf.score(X_test, y_test) #0.8877551020408163
print(rf_score)

# =============================================================================
# Model Evalutation / Parameter Tuning 
# =============================================================================

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

#Re-running model without least import features

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
       'player_fabricio coloccini', 'player_gabriel', 'player_mark bunn'])

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

#Grid Search for Param Tuning
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
print(CV_rf_score)


#Rank current features in the model


confusion_matrix_model(rf)
confusion_matrix_model(CV_rf)
accuracy_score(CV_rf)

#Kappa Score following from 
kappa_score = cohen_kappa_score(y_test, y_pred, labels=None, weights=None)
print(kappa_score) #0.0782178217821784 - very low Kappa despite high R2


