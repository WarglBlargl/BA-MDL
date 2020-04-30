# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:42:09 2020

@author: Simon
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
  #return a * np.exp(-b * x) + c
  return a * np.log(b * x) + c

def plot():
    
    x = np.linspace(1,5,10)   # changed boundary conditions to avoid division by 0
    y = func(x, 2.5, 1.3, 0.5)
    np.random.seed(5)
    yn = y + 0.5*np.random.normal(size=len(x))
    popt, pcov = curve_fit(func, x, yn)
    
    plt.figure(dpi=100)
    plt.plot(x, yn, 'ko', label="Original Data")
    #plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
    #plt.plot(x,[i+0.75 for i in x] , 'b-',label="Simple Model")
    coefficients = np.polyfit(x, yn, len(y)-1)
    poly = np.poly1d(coefficients)
    new_x = np.linspace(x[0], x[-1])
    new_y = poly(new_x)
    #plt.plot(x, y, "o", c='black')
    plt.plot(new_x, new_y, c='purple', label="Overfitting Model" )
    plt.xlim([x[0]-1, x[-1]+1])
    plt.ylim([0,max(y)+1])
    plt.legend()
    plt.savefig("OverfittingModel.jpg")
    plt.show()