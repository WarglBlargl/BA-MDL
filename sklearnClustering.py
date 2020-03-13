# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:49:44 2020

@author: Simon
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

import scipy
import scipy.stats

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm
        #download directly sklearn datasets package| generate datasets myself
data = pd.read_csv("C:/Users/Simon/Dropbox/Uni/BA MDL Hierarchical Clustering/MDL based example/data/numerical/breast-w.csv")
data.columns =['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']


data = data[data.bare_nuclei != '?']

for i in range(0, len(data.columns)):
    data.iloc[:,i] = pd.to_numeric(data.iloc[:,i], errors='ignore')
# errors='ignore' lets strings remain as 'non-null objects'#dimensions sliced
    
#take sample: first 100 obvervations
variables = data.iloc[:500, : 9].values
class_variable = data.iloc[:500, (9)].values

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage= 'ward')
fitted = cluster.fit_predict(variables)
print(fitted)


#takes cluster affiliation matrix and dataframe
#returns distance matrix: datapoints-mean of cluster || last column = cluster affiliation
def mapping (clusterAffiliation, data_array):
    
    cluster_arr = np.array(clusterAffiliation)
    d_arr = np.array(data_array)
    
    #create an array as a combination of the data values
    #and the cluster indicies
    dic_arr = np.zeros(shape=(d_arr.shape[0],d_arr.shape[1]+1))
    dic_arr[:,:-1] = d_arr
    dic_arr[:,-1] = cluster_arr

    #create a set consisting of all unique cluster indicies for later iteartion
    clusters = set(clusterAffiliation)
    numberofc = len(clusters)
    print('number of clusters: ', numberofc)
    
    #initialize the mean and distance matrix
    means=np.zeros(shape=(numberofc,data_array.shape[1]+1))
    d_distance = np.zeros(shape=(d_arr.shape[0],d_arr.shape[1]+1))
    
    #iterate through all clusters
    for value in clusters:      
        #calcuation of the mean and distances to the mean values of the current cluster
        means[value] = np.mean(dic_arr[dic_arr[:,-1]==value,:],axis=0)
        d_temp = np.sqrt((dic_arr[dic_arr[:,-1]==value,:]-means[value])**2)
        d_temp[:,-1]=value
        
        #combine the distances of all clusters into one matrix
        if value == 0:
            d_distance = d_temp
        else:
            d_distance = np.concatenate((d_distance,d_temp))
            
            
    return d_distance

#takes distancematrix, last attribute is cluster affiliation
def binning (distanceMatrix):

        
    cluster=set(distanceMatrix[:,-1])
    
    #takes means of all attributes for every element
    means=np.zeros(shape=(distanceMatrix.shape[0],2))
    attributes=np.copy(distanceMatrix[:,:-1])
    
    for value in range(len(attributes)):
        means[value][0]=np.mean(attributes[value])
        means[value][1]=distanceMatrix[value][-1]
            
    #binned=np.digitize(means[:,:-1],bins)
    
    clusterList=list()
    for value in cluster:
        
        tempcluster = means[means[:,-1]==value,:-1]
        clusterList.append(tempcluster)
        print(value, ' ', len(tempcluster))
        
    #plot individual binned cluster
    bins = np.arange(0,10.1,0.1)
    fig=plt.figure(figsize=(len(cluster)*7,7))

    for index in range(len(cluster)):
        
        currentCluster = clusterList[index]
        ax=fig.add_subplot(1,len(cluster),index+1)
        weights = np.ones_like(currentCluster)/len(currentCluster)
        print(weights)
        ax.hist(clusterList[index], bins=bins, histtype='bar', alpha=1, log=False, weights=weights)
        
#takes distancematrix with last column=clusteraffiliation
#returns list of means of elements to its clustermean
def toClusterList(distanceMatrix):
            
    cluster=set(distanceMatrix[:,-1])
    
    #takes means of all attributes for every element
    means=np.zeros(shape=(distanceMatrix.shape[0],2))
    attributes=np.copy(distanceMatrix[:,:-1])
    
    for value in range(len(attributes)):
        means[value][0]=np.mean(attributes[value])
        means[value][1]=distanceMatrix[value][-1]
            
    #binned=np.digitize(means[:,:-1],bins)
    
    clusterList=[]
    for value in cluster:
        
        tempcluster = means[means[:,-1]==value,:-1]
        clusterList.append(tempcluster)
        print(value, ' ', len(tempcluster))
    
    return (clusterList)

#takes distancematrix, last attribute is cluster affiliation
#returns best fitting distribution (String) and its best Parameters
def fitDistribution(clusterList, ax=None):
    
    #fit best distribution
    
    y, x = np.histogram(clusterList, bins=50, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    
    dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
    # Best holders
    best_distribution = 'norm'
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for dist_name in dist_names:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                dist = getattr(scipy.stats, dist_name)
                params = dist.fit(clusterList)

                # Separate parts of parameters
                #loc=mean, scale = standard deviation
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                
                pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))
                
                print('SSE: ', sse, ' ', dist_name)
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                except Exception:
                    pass
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = dist_name
                    best_params = params
                    best_sse = sse
                
        except Exception:
            pass

    return (best_distribution, best_params)     

def make_pdf(dist_name, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist_name.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist_name.ppf(0.01, loc=loc, scale=scale)
    end = dist_name.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist_name.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist_name.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def plotpdf(distancematrix):
    
    clusterList = toClusterList(distancematrix)
    for cluster in range(len(clusterList)):
        
        data = pd.Series(np.ravel(clusterList[cluster]))                  #!!!Change to every cluster individualy
        
        # Plot for comparison
        plt.figure(figsize=(12,8))
        ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5)
        # Save plot limits
        dataYLim = ax.get_ylim()
        
        # Find best fit distribution
        best_fit_name, best_fit_params = fitDistribution(data, ax=ax)
        best_dist = getattr(scipy.stats, best_fit_name)
        
        # Update plots
        ax.set_ylim(dataYLim)
        ax.set_title(u'Mean distance to cluster centre.\n All Fitted Distributions')
        ax.set_xlabel(u'Distance')
        ax.set_ylabel('Frequency')
        
        # Make PDF with best params 
        pdf = make_pdf(best_dist, best_fit_params)
        
        # Display
        plt.figure(figsize=(12,8))
        ax = pdf.plot(lw=2, label='PDF', legend=True)
        data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)
        
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
        dist_str = '{}({})'.format(best_fit_name, param_str)
        
        ax.set_title(u'Mean distance to cluster centre with best fit distribution \n' + dist_str)
        ax.set_xlabel(u'Distance to cluster mean')
        ax.set_ylabel('Frequency')

"""
def computeMDL (distribution, params, data):
    
    #k:number of cluster means
    #kFP: number of free parameters needed for given pdf
    #n: number of observations over all clusters
    #sum: sum of -log2 of distances to its cluster means over all clusters
    
    k = len(data)
    kFP = len(params)
    n=0
    sum=0
    for element in range(len(clusterList)):
        n += len(clusterList[element])
        sum += np.sum(-np.log2(clusterList[element]))
        
    HeuCost = (k + (kFP) * math.log2(n)

    #Data|Heuristic cost is the log2 likelihood of the distances w.r.t. that PDF
    
    DHCost = sum
    
    MDLCost= HeuCost + DHCost
   
    return MDLCost
"""