# -*- coding: utf-8 -*-
"""
Created on October 15th 22:25:17 2018

@author: jamespatten
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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


#predictor is shot outcome
