#!/usr/bin/env python
# coding: utf-8

# # ------read data--------

# In[1]:


import pandas as pd
import numpy as np
import STAGATE
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import h5py
from itertools import compress
from termcolor import colored


# In[4]:


def iter_feature(adata,marker_num,k):
    ii=adata.uns['rank_genes_groups']['names']
    a=np.array(ii[0:marker_num])
    b=[]
    for i in range(len(a)):
        b.append(list(a[i]))

    p=[]
    for i in range(len(b)):
        for j in range(k):
            p.append(b[i][j])

    pp=sorted(set(p), key = p.index)
    l=len(pp)
    return pp,l


# # 2D data automic peak picking

# In[23]:


def automatic_2D_peak_picking(adata,ol_mass,num_marker=1300,r=1.4,alpha=0.2,
                     n_epochs=1000,pre_resolutation=0.2,hidden_dims=[256, 10],rank_method='wilcoxon',
                    cluster_method='louvain',k_class_first=5,k_class=10,k_resolutation=0.2):
    #---adata:input  data
    #----ol_mass:the number of unpicked features
    #---r: the neighbour of SNN
    #k_class_first:if it is te first iteration,then this is the k_class of mclust
    #k_class:the number of class when choose mclust to cluste
    #rank_method:include for [wilcoxon,t-test,t-test-variance,log]
    #which  order the contribution of each cluster
    #clust_method:[mclust,louvain,leiden]
    #k_resolutation: the resolutation of louvain and leiden 
    #num_marker:the number of marker choose
    #k_resolution:if cluster_method=[louvain,leiden],then is the resolution of cluster
    #pre_resolutation:pre-cluster resolution
    #n_epochs:number of epoch
    ifpicking=int(input("ifpicking= "))
    
    feature_number=adata.n_vars
    
    #-----whether need automatic peak picking---------
    if ifpicking==0:
        
        adata_2=adata.copy()
        print('No peak picking, just run the graph-attention autoencoder')
        
        sc.pp.highly_variable_genes(adata_2,flavor="cell_ranger",n_top_genes=feature_number)
        
        STAGATE.Cal_Spatial_Net(adata_2, rad_cutoff=r)
        STAGATE.Stats_Spatial_Net(adata_2)
        adata_2= STAGATE.train_STAGATE(adata_2,alpha=alpha,n_epochs=n_epochs,
                                      pre_resolution=pre_resolutation,hidden_dims=hidden_dims)
        
        sc.pp.neighbors(adata_2, use_rep='STAGATE')
        #-----judge the method for clusting------
        if cluster_method=='louvain':
            sc.tl.louvain(adata_2, resolution=k_resolutation)
        elif cluster_method=='mclust':
            adata_2 = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
        elif cluster_method=='leiden':
            sc.tl.leiden(adata_2, resolution=k_resolutation)
            
        #sc.tl.rank_genes_groups(adata_2,cluster_method,method=rank_method)
        
        #sc.tl.rank_genes_groups(adata,cluster_method,method=rank_method)
            
    #---to judge whether is the first iteration---
    
    elif ifpicking==1:
        
        
        if feature_number==ol_mass:
            print("---Begin to automatic peak picking!-----")
            print('First iter begin!')
            
            sc.pp.highly_variable_genes(adata,flavor="cell_ranger",n_top_genes=feature_number)

            STAGATE.Cal_Spatial_Net(adata, rad_cutoff=r)
            STAGATE.Stats_Spatial_Net(adata)
            
            adata = STAGATE.train_STAGATE(adata,alpha=alpha,n_epochs=n_epochs,
                                 pre_resolution=pre_resolutation,hidden_dims=hidden_dims)

            sc.pp.neighbors(adata, use_rep='STAGATE')

            #-----judge the method for clusting------
            if cluster_method=='louvain':
                sc.tl.louvain(adata, resolution=k_resolutation)
                k=len(np.unique(adata.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=k_class_first)
                k=k_class_first
            elif cluster_method=='leiden':
                sc.tl.leiden(adata, resolution=k_resolutation)
                k=len(np.unique(adata.obs['leiden']))

            #-----rank of peak -------    
            sc.tl.rank_genes_groups(adata,cluster_method,method=rank_method)

            #----choose top rank peak feature-----
            pp,l=iter_feature(adata,num_marker,k)
            ll=[]
            for i in range(len(pp)):
                listA=adata.var_names==pp[i] 
                res = list(compress(range(len(listA)), listA))
                ll.append(res[0])

            #----After first iter and decide the top feature we need to run graph-attention again to peak picking    
            adata_2=adata.T[ll].copy().T
            feature_number_2=adata_2.n_vars
            sc.pp.highly_variable_genes(adata_2,flavor="cell_ranger",n_top_genes=feature_number_2)

            STAGATE.Cal_Spatial_Net(adata_2, rad_cutoff=r)
            STAGATE.Stats_Spatial_Net(adata_2)
            adata_2 = STAGATE.train_STAGATE(adata_2, alpha=alpha,n_epochs=1000,pre_resolution=pre_resolutation,hidden_dims=[256, 10])
            
            sc.pp.neighbors(adata_2, use_rep='STAGATE')

            if cluster_method=='louvain':
                sc.tl.louvain(adata_2, resolution=k_resolutation)
                #k=len(np.unique(adata_2.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
                #k=k_class_first
            elif cluster_method=='leiden':
                sc.tl.leiden(adata_2, resolution=k_resolutation)
                #k=len(np.unique(adata_2.obs['leiden']))
                
            sc.tl.rank_genes_groups(adata_2, cluster_method, method=rank_method)
        else:
            #----if not the first iterion-----
            print(colored('------begin to automic peak picking-----', attrs=['bold']))  
            
            i=['louvain','leiden','mclust']
            if i in adata.obs.columns.values:
                k=len(np.unique(adata.obs[i]))
                
            #----choose the top rank features
            pp,l=iter_feature(adata,num_marker,k)
            ll=[]
            for i in range(len(pp)):
                listA=adata.var_names==pp[i] 
                res = list(compress(range(len(listA)), listA))
                ll.append(res[0])

            adata_2=adata.T[ll].copy().T

            feature_number_2=adata_2.n_vars
            sc.pp.highly_variable_genes(adata_2,flavor="cell_ranger",n_top_genes=feature_number_2)

            STAGATE.Cal_Spatial_Net(adata_2, rad_cutoff=r)
            STAGATE.Stats_Spatial_Net(adata_2)
            adata_2 = STAGATE.train_STAGATE(adata_2, alpha=alpha,n_epochs=1000,pre_resolution=pre_resolutation,hidden_dims=[256, 10])

            #-----judge the algorithm to clust-------
            if cluster_method=='louvain':
                sc.tl.louvain(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
            elif cluster_method=='leiden':
                sc.tl.leiden(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['leiden']))

            sc.tl.rank_genes_groups(adata_2, cluster_method, method=rank_method)
    else:
            print("Please ifpicking enter 0/1: 0 means no picking and 1 means peak picking")
            adata_2=adata.copy()
        
    return adata_2






