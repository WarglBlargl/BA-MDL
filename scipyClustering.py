# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:26:53 2019

@author: Simon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster, cophenet
from scipy.spatial.distance import pdist


#for continous data
#np.set_printoptions(precision=5, suppress=True)

plt.figure(27,figsize=(15,15))

data = pd.read_csv("C:/Users/Simon/Dropbox/Uni/BA MDL Hierarchical Clustering/MDL based example/data/numerical/breast-w.csv")
data.columns =['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

data = data[data.bare_nuclei != '?']

for i in range(0, len(data.columns)):
    data.iloc[:,i] = pd.to_numeric(data.iloc[:,i], errors='ignore')
# errors='ignore' lets strings remain as 'non-null objects'#dimensions sliced
    
variables = data.iloc[:100, : 9].values
class_variable = data.iloc[:100, (9)].values

plt.title("dendrogram")
plt.xlabel("clusters")
plt.ylabel("distance")


#dendrogram
linked = linkage(variables, 'ward')
dend=dendrogram(linked, truncate_mode='lastp',p=12, leaf_rotation=45., leaf_font_size=15.,show_contracted=True)
