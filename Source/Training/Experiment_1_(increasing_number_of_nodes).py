#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TimeSeriesModelCreator_Parallel_talos import TimeSeriesModelCreator_Parallel_talos
import pandas as pd
import matplotlib.pyplot as plt
#import os
#os.environ["PATH"] += os.pathsep + 'C:\Program Files (x86)\Graphviz2.38\bin'


# In[7]:


look_backs = [1]
modelCreators = []
for look_back in look_backs:
    modelCreators.append(TimeSeriesModelCreator_Parallel_talos(look_back, r'..\Datasets\GEANTCombined\all_in_one_complete_appended.csv'))


# In[ ]:


for modelCreator in modelCreators:
    batch_sizes = [100]
    epochs = [100]
    nodes = [1, 2, 3, 8, 16, 32, 64, 128]
    layers = [1]
    optimizers = ['adam']
    losses = ['mean_squared_error']
    modelCreator.test_for_optimal_config('Epxeriment_1', 1, 5, 11, 11, 1000, batch_sizes, epochs, nodes, layers, optimizers, losses, 40, 1, 0, 200)




