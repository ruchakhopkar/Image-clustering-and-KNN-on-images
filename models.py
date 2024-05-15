#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:57:00 2023

@author: ruchak
"""
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def dimension_reduction(test_feature_space):
    tsne = TSNE(n_components=3, random_state=42)
    x = tsne.fit_transform(test_feature_space)
    return x

def get_dbscan(df, test_feature_space):
    dbscan = DBSCAN(eps=1.1, min_samples=6, p = 1)  # Adjust parameters based on your data
    df['Labels'] = dbscan.fit_predict(test_feature_space)
    return df

def get_knn_results(df, test_feature_space):
    train_feature_space = np.load('/home/ruchak/Desktop/RDF/scripts/train/features.npy', allow_pickle = False)
    X_train = dimension_reduction(train_feature_space)
    y_train = pd.read_csv('/home/ruchak/Desktop/RDF/scripts/train/all_images.csv')
    y_train = y_train[y_train['Labels']!= -1]
    X_train = X_train[y_train.index.to_list()]
    y_train = y_train['Labels'].values.reshape(-1,1)
    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    y = knn.predict(test_feature_space)
    df['Classes'] = y
    return df
    
    
    
def main(df, test_feature_space, train = False):
    test_feature_space = dimension_reduction(test_feature_space)
    
    if train:
        df = get_dbscan(df, test_feature_space)
    
    else:
        df = get_knn_results(df, test_feature_space)
    return df
        