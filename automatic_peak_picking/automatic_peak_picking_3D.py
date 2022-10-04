#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib
import h5py
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys

import STAGATE
from itertools import compress
from termcolor import colored


# In[ ]:


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


# In[ ]:


def automatic_3D_peak_picking(adata,ol_mass,section_order,num_marker=1300,
                        rad_cutoff_2D=1.4,rad_cutoff_Zaxis=1.4,key_section='section_id',
                     alpha=0.2,n_epochs=1000,pre_resolutation=0.2,hidden_dims=[256, 10],
                       cluster_method='louvain',rank_method='wilcoxon',
                            k_resolutation=0.2,k_class_first=6,k_class=10):
     #---adata:input  data
    #----ol_mass:the number of unpicked features
    #num_marker:chosse the number of peak feature
    #k_class:the number of class when choose mclust to cluste
    #--section_order: secticn order
    #--rad_cutoff_2D: A neighbor of the same slice in an SNN
    #---rad_cutoff_Zaxis：Number of neighbors in adjacent slices
    #rank_method:include for [wilcoxon,t-test,t-test-variance,log] which  order the contribution of each cluster
    #clust_method:[mclust,louvain,leiden]
    #k_resolutation: the resolutation of louvain and leiden 
    feature_number=adata.n_vars
    
    ifpicking=int(input("ifpicking= "))
    
    if ifpicking==0:
        
        adata_2=adata.copy()
        print('No peak picking, just run the graph-attention autoencoder')
        
        sc.pp.highly_variable_genes(adata_2,flavor="cell_ranger",n_top_genes=feature_number)
        
        STAGATE.Cal_Spatial_Net_3D(adata_2, rad_cutoff_2D=rad_cutoff_2D, rad_cutoff_Zaxis=rad_cutoff_Zaxis,
                                   key_section=key_section,section_order = section_order, verbose=True)
        
        #STAGATE.Cal_Spatial_Net(adata, rad_cutoff=r)
        #STAGATE.Stats_Spatial_Net(adata)
        #adata = STAGATE.train_STAGATE(adata, alpha=0)
        adata_2 = STAGATE.train_STAGATE(adata_2,alpha=alpha,n_epochs=n_epochs,pre_resolution=pre_resolutation,hidden_dims=hidden_dims)
        
        sc.pp.neighbors(adata_2, use_rep='STAGATE')
        
        if cluster_method=='louvain':
            sc.tl.louvain(adata_2, resolution=k_resolutation)
            #k=len(np.unique(adata_2.obs['louvain']))
        elif cluster_method=='mclust':
            adata_2 = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
            #k=k_class
        elif cluster_method=='leiden':
            sc.tl.leiden(adata_2, resolution=k_resolutation)
            #k=len(np.unique(adata_2.obs['leiden']))
        #adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=k_class)
        #sc.tl.rank_genes_groups(adata,cluster_method,method=rank_method)
        
    elif ifpicking==1:
        
        if feature_number==ol_mass:
            #---first iteration---
            print("First iter begin!")
            #pigspa=adata.obsm['spatial']
            sc.pp.highly_variable_genes(adata,flavor="seurat_v3",n_top_genes=feature_number)

            STAGATE.Cal_Spatial_Net_3D(adata, rad_cutoff_2D=rad_cutoff_2D, rad_cutoff_Zaxis=rad_cutoff_Zaxis,
                                       key_section=key_section,section_order = section_order, verbose=True)

            #STAGATE.Cal_Spatial_Net(adata, rad_cutoff=r)
            #STAGATE.Stats_Spatial_Net(adata)
            #adata = STAGATE.train_STAGATE(adata, alpha=0)
            adata = STAGATE.train_STAGATE(adata,alpha=alpha,n_epochs=n_epochs,pre_resolution=pre_resolutation,hidden_dims=hidden_dims)

            sc.pp.neighbors(adata, use_rep='STAGATE')

            if cluster_method=='louvain':
                sc.tl.louvain(adata, resolution=k_resolutation)
                k=len(np.unique(adata.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=k_class_first)
                k=k_class_first
            elif cluster_method=='leiden':
                sc.tl.leiden(adata, resolution=k_resolutation)
                k=len(np.unique(adata.obs['leiden']))
            #adata = STAGATE.mclust_R(adata, used_obsm='STAGATE', num_cluster=k_class)
            sc.tl.rank_genes_groups(adata,cluster_method,method=rank_method)
            #k=len(np.unique(adata.obs['louvain']))

            pp,l=iter_feature(adata,num_marker,k)
            ll=[]
            for i in range(len(pp)):
        #if pp[i] in (adata.var_names):
                listA=adata.var_names==pp[i] 
                res = list(compress(range(len(listA)), listA))
                ll.append(res[0])

            adata_2=adata.T[ll].copy().T
            feature_number_2=adata_2.n_vars
            sc.pp.highly_variable_genes(adata_2,flavor="seurat_v3",n_top_genes=feature_number_2)

            #STAGATE.Cal_Spatial_Net(adata_2, rad_cutoff=r)
            #STAGATE.Stats_Spatial_Net(adata_2)
            STAGATE.Cal_Spatial_Net_3D(adata, rad_cutoff_2D=rad_cutoff_2D, rad_cutoff_Zaxis=rad_cutoff_Zaxis,
                                       key_section=key_section,section_order = section_order, verbose=True)
            adata_2 = STAGATE.train_STAGATE(adata_2, alpha=alpha,n_epochs=1000,
                                            pre_resolution=pre_resolutation,hidden_dims=[256, 10])
            
            sc.pp.neighbors(adata_2, use_rep='STAGATE')

            if cluster_method=='louvain':
                sc.tl.louvain(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
                k=k_class
            elif cluster_method=='leiden':
                sc.tl.leiden(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['leiden']))
            sc.tl.rank_genes_groups(adata_2, cluster_method, method=rank_method)
        else:

            print("------begin to automatic peak picking-----")  

            pp,l=iter_feature(adata,num_marker,k)
            ll=[]
            for i in range(len(pp)):
                listA=adata.var_names==pp[i] 
                res = list(compress(range(len(listA)), listA))
                ll.append(res[0])

            adata_2=adata.T[ll].copy().T
            feature_number_2=adata_2.n_vars
            sc.pp.highly_variable_genes(adata_2,flavor="seurat_v3",n_top_genes=feature_number_2)

            #STAGATE.Cal_Spatial_Net(adata_2, rad_cutoff=r)
            #STAGATE.Stats_Spatial_Net(adata_2)
            STAGATE.Cal_Spatial_Net_3D(adata_2, rad_cutoff_2D=rad_cutoff_2D, rad_cutoff_Zaxis=rad_cutoff_Zaxis,
                                       key_section=key_section,section_order = section_order, verbose=True)

            adata_2 = STAGATE.train_STAGATE(adata_2, alpha=alpha,n_epochs=1000,pre_resolution=pre_resolutation,hidden_dims=[256, 10])

            if cluster_method=='louvain':
                sc.tl.louvain(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['louvain']))
            elif cluster_method=='mclust':
                adata = STAGATE.mclust_R(adata_2, used_obsm='STAGATE', num_cluster=k_class)
                k=k_class
            elif cluster_method=='leiden':
                sc.tl.leiden(adata_2, resolution=k_resolutation)
                k=len(np.unique(adata_2.obs['leiden']))

            sc.tl.rank_genes_groups(adata_2,cluster_method, method=rank_method)
    else:
        
        print("Please ifpicking enter 0/1: 0 menas no pricking and 1 means peak picking")
        
        adata_2=adata.copy()
        
    return adata_2




