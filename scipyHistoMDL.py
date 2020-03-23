# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:04:32 2020

@author: Simon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration

import sklearn.datasets as sk
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cut_tree
from scipy.cluster.hierarchy import cophenet
from scipy.cluster.hierarchy import inconsistent
import scipy.cluster.hierarchy as sc


from scipy.spatial.distance import pdist


'Loading and ploting Datasets'


def loadDF(name= sk.load_iris()):
    data = name
    df = pd.DataFrame(data= np.c_[data['data'], data['target']], columns=np.append(data['feature_names'], 'target'))
    return df

#generates DF with n_blobs cluster + 1 moon
def generateDF(n_blobs=2, n_samples_blobs=100, n_samples_circ=0, random_state=2, std=1, center_box=(-30,30)):

    n_features=2
    X1, y1 = make_blobs(n_samples_blobs, centers=n_blobs, n_features=2, random_state=random_state, center_box=center_box, cluster_std=std)
    X2, _ = make_moons(n_samples_circ*2, noise=.05, random_state=random_state, shuffle= False)
    X2=X2*8
    circ_labels=np.full(n_samples_circ, fill_value=n_blobs)
    
    #get dataframe with 2 features + target
    d = {'x':np.append(X1[:,-n_features], X2[:n_samples_circ,-n_features]),
         'y':np.append(X1[:,-n_features+1], X2[:n_samples_circ,-n_features+1]),
         'target':np.append(y1,circ_labels) }
    df=pd.DataFrame(data=d)
    return df

#Plot 2d Dataframe with 4 clusters
def plotDF(df):
    fig, ax = plt.subplots()
    grouped = df.groupby('target')
    centers = df['target'].unique()
    colors = cm.rainbow(np.linspace(0,1, len(centers)))

    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x = df.columns.values[0], y= df.columns.values[1], 
                   label = int(key) ,
                   color=colors[int(key)].reshape(1,4), alpha=0.8)
        
def fancy_dendrogram(*args, **kwargs):
    plt.figure(num=0, figsize=(20,20))
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):

        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    ddata
        

'Scaling of data and dimensionality reduction through PCA or Feature Agglomeration'



#returns linkage as [idx1, idx2, dist, sample_count]
def getDist(df, method='single', metric='euclidean', optimal_ordering=False):
    
    data = df.drop(['target'], axis=1)

    link = linkage(data, method, metric, optimal_ordering)
    
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
    link_df = pd.DataFrame(link)
    #cutree= cut_tree(link_df, n_clusters=link_df.shape[0])
    return link


def getleafdict(node):
    """
    Perform pre-order traversal without recursive function calls.

    When a leaf node is first encountered, ``func`` is called with
    the leaf node as its argument, and its result is appended to
    the list.

    For example, the statement::

       ids = root.pre_order(lambda x: x.id)

    returns a list of the node ids corresponding to the leaf nodes
    of the tree as they appear from left to right.

    Parameters
    ----------
    func : function
        Applied to each leaf ClusterNode object in the pre-order traversal.
        Given the ``i``-th leaf node in the pre-order traversal ``n[i]``,
        the result of ``func(n[i])`` is stored in ``L[i]``. If not
        provided, the index of the original observation to which the node
        corresponds is used.

    Returns
    -------
    L : list
        The pre-order traversal.

    """

    n = node.count
    
    Dict={}
    
    curNode = [None] * (2 * n)
    lvisited = set()
    rvisited = set()
    curNode[0] = node
    k = 0
    dists = [node.dist]
    while k >= 0:
        nd = curNode[k]
        ndid = nd.id
        if nd.is_leaf():
        #preorder.append(func(nd))
            k = k - 1
            Dict[nd.id]=dists[:k+1]
            del dists[-1]
        else:
            if ndid not in lvisited:
                curNode[k + 1] = nd.left
                dists.append(nd.left.dist)
                lvisited.add(ndid)
                k = k + 1
            elif ndid not in rvisited:
                curNode[k + 1] = nd.right
                dists.append(nd.right.dist)
                rvisited.add(ndid)
                k = k + 1
            # If we've visited the left and right of this non-leaf
            # node already, go up in the tree.
            else:
                k = k - 1
                
    return Dict

def binning(Dict):
    
    n_bins=[]
    distribution=[]
    bins = 'sqrt'
    #fig=plt.figure(figsize=(len(Dict.keys())*7,7))
    #n_leaves= np.zeros(len(Dict))
    #counter =0

    for cluster, c_content in Dict.items():
        nonzero=[]
        
        dists=np.zeros(len(c_content.keys()))
        index=0
        
        "takes sum of dist over all steps of each individual leaf"
        
        for leaf in c_content:
            
            dists[index]=np.sum(c_content[leaf])
            index+=1
            
        #n_leaves[counter-1]=index
        #counter += 1
        #ax=fig.add_subplot(1,len(Dict.keys()), counter)
        hist, bin_edges = np.histogram(dists, bins= bins, range=(0,dists.max()))

        n_bins.append(len(bin_edges))
        #n, b, patches= ax.hist(dists, bins=bins, histtype='bar', alpha=1, log=False, density=False)
        for binheight in hist:
            if binheight>0:
                nonzero.append(binheight)
        distribution.append(nonzero)
        
    return MDL(distribution, n_bins)

def MDL(distribution, n_bins):
    
    n_datapoints=0
    n_clusters = len(distribution)
    for c in distribution:
        n_datapoints+=sum(c)

    
    heucost= 0
    print("numberofclusters in clustering: ", n_clusters)
    repcost=0
    #k= number of clusters (choosing one clusterlabel per cluster)
    #P= number of parameters needed to describe the cluster (equal to number of probabilities needed to describe hist)
    
    #L(H)=sumOverClusters(log2(n_probs needed to describe hist)+log2(k = len(n_leaves)))
    #L(D|H)=sumOverClusters(-log2(probs))
    index=0
    for c in distribution:
        
        n_leaves= sum(c)
        probs = [i * (1/n_leaves) for i in c]
        
        heucostc = np.log2(n_clusters)+(-np.log2(len(probs)/n_bins[index]))*n_datapoints
        repcostc = np.sum(np.multiply(c,-np.log2(probs)))
        
        heucost += heucostc
        repcost += repcostc
        index +=1
        print("n_leaves in cluster: ", n_leaves, " heucost: " ,heucostc, " repcost: ", repcostc)
    print("total heucost: " ,heucost, " total repcost: ", repcost, " total cost: ", heucost+repcost)
    return heucost+repcost

def optMDL (df):
    
    Z=getDist(df)
    tree = sc.to_tree(Z, rd=True)[1]
    minMDL=1000000
    optK=0
    desLength=0
    
    for n_cluster in range(1,18,1): #range(df.shape[0]+1)
        N=fcluster(Z, n_cluster, criterion='maxclust')
        L, M = sc.leaders(Z, N)
        leaders = list(L)
        print(leaders)
        leafDict={}
    
        for node in tree:
            if node.get_id() in leaders:
                key = node.get_id()
                                
                if node.get_count() > 1:
                    
                    dist= getleafdict(node)
                else:
                    dist = {key: 0 }
                    
                leafDict[key] = dist
                
        desLength=binning(leafDict)
        
        
        #if desLength
        if desLength<minMDL:
            minMDL=desLength
            optK=n_cluster
    return optK, minMDL
        
def binningC (Dict, n_datapoints):
    
    distribution=[]
    bins = 'sqrt'
    dists=np.zeros(len(Dict))
    index=0
    
    for leaf, l_content in Dict.items():
        dists[index]=np.sum(l_content)
        index+=1
        
    hist, bin_edges = np.histogram(dists, bins= bins, range=(0,dists.max()))
    
    for binheight in hist:
                if binheight>0:
                    distribution.append(binheight)
    
    #k= number of clusters (choosing one clusterlabel per cluster)
    #P= number of parameters needed to describe the cluster (equal to number of probabilities needed to describe hist)
    
    #L(H)= number of bits required to represent the model
    #L(D|H)= number of bits required to represent the predictions of the model
    
    probs = [i * (1/sum(hist)) for i in distribution]
    repcostc = np.sum(np.multiply(distribution,[-np.log2(i) for i in probs]))
    
    heucostc =(-np.log2(len(probs)/len(hist)))*n_datapoints
    print("probs: ", len(probs),"hist: ", len(hist))
    print("heucostc: ",heucostc,"probs: ", probs)
    return repcostc+heucostc

def optMDLC (df, cutoffvalue):
    
    Z=getDist(df)   
    n_datapoints=df.shape[0]
    n_cluster= 1
    root = sc.to_tree(Z, rd=False)
    rootDL = binningC(getleafdict(root),n_datapoints)
    #monocrit=np.zeros((Z.shape[0],))
    print("ROOTDL", rootDL)
    nodelist = [root]
    dllist = [rootDL]

    
    while max(dllist) > cutoffvalue:
        nodeindex = dllist.index(max(dllist))
        node = nodelist[nodeindex]
        if node.is_leaf():
            break
        
        nodeDL = binningC(getleafdict(node),n_datapoints)

        '''clustercost of MDL(H) = np.log2(n_cluster)*n_cluster'''
        
        leftDL = binningC(getleafdict(node.get_left()), n_datapoints)
        rightDL = binningC(getleafdict(node.get_right()),n_datapoints)
        
        print( " \n LEFTDL:", leftDL, "RIGHTDL: ", rightDL, "NODEDL: ", nodeDL)
        if nodeDL + np.log2(n_cluster)*n_cluster > leftDL + rightDL + np.log2(n_cluster+1)*2:
        
            nodelist.append(node.get_left())
            dllist.append(leftDL)
            nodelist.append(node.get_right())
            dllist.append(rightDL)
            nodelist.remove(node)
            del dllist[nodeindex]
            n_cluster += 1
        else:
            break
        print("dllist:  ", dllist, "minDL:  ", sum(dllist))
    minDL= sum(dllist) + np.log2(n_cluster)*n_cluster
    
    return n_cluster, minDL

def greedysearch(node, cutoffvalue, n_datapoints, cutofflist, DLlist):
    
    
    cutofflist=cutofflist
    DLlist=DLlist
    DLC=binningC(getleafdict(node),n_datapoints)
    
   
    
    if DLC < cutoffvalue:
        
        cutofflist.append(node)
        DLlist.append(DLC)
        
    if DLC >= cutoffvalue:
        if not node.get_left().is_leaf():
            greedysearch(node.get_left(),cutoffvalue, n_datapoints, cutofflist, DLlist)
        if not node.get_right().is_leaf():
            greedysearch(node.get_right(),cutoffvalue, n_datapoints, cutofflist, DLlist)
    
    return cutofflist, DLlist
    

