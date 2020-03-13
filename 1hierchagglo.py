# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 13:08:24 2019

@author: Simon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import normalize
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv("Iris.csv")
data = data.drop(columns="class")
data=pd.DataFrame(data, columns=data.columns)
#data1 = data[data.columns.difference(["class"])]

#dend = sch.dendrogram(sch.linkage(data, method="ward", metric="euclidean"))
cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
cluster.fit_predict(data)

plt.figure(figsize=(10,7))
plt.scatter(data['sepal-length'], data['petal-length'], c=cluster.labels_) 