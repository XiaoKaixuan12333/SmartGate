#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[21]:


def SmartGate_TIC(data,c):
    #--data:input data
    #--c:is an empirical constant. For the multicellular resolution data, C is set to the default value 1. 
    #For the subcellular resolution data, C is set to their mean intensity accordingly
    
    df=pd.DataFrame(data)
    
    TIC=df.sum(axis=1)
    df=c*(df.div(TIC,axis=0))
    return df


# In[24]:


try:  
    get_ipython().system('jupyter nbconvert --to python TIC_normalization.ipynb')
except:
    pass


# In[ ]:




