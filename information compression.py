# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 14:17:56 2019

@author: Simon
"""

import math
import scipy.special
import numpy as np

array = ([[3, 5, 2, 4],
         [7, 6, 8, 8],
         [1, 6, 7, 7]])

np.random.seed(0)
x1 = np.random.randint(4, size=(3,10))

def prob (attribute, outcome):
   return np.count_nonzero(attribute==outcome)/len(attribute)

def hlength (data):
    for i in range(len(data)):
       # print (str(i) + ". attribute")
        for j in range(len(data[i])):
           # print(data[i][j])
           
           inflength = -(math.log(prob(data[i],data[i][j]),2))
           print (inflength)