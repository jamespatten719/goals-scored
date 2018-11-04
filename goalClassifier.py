# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 22:25:17 2018

@author: jamespatten
EDA inspiration from angps95

"""
# =============================================================================
# Import Libs
# =============================================================================
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
df = pd.read_csv('events.csv')
df=df[df["is_goal"]==1]
df = df.loc[df['event_team'] == 'Arsenal']

#df in each minute
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

#df = df.loc[df['event_team'] == 'Arsenal']
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

# =============================================================================
# Model Evalutation / Parameter Tuning 
# =============================================================================

#Grid Search for Param Tuning
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10],
    'criterion' :['mse', 'mae']
}

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)
CV_rf.best_params_


def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_train,XC.predict(X_train))
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Goal","Predicted Miss"]
    cm.index=["Actual Goal", "Actual Miss"]
    return cm

confusion_matrix_model(rf)


