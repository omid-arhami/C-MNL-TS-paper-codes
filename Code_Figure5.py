#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
get_ipython().run_line_magic('matplotlib', 'inline')

import pickle

import warnings
warnings.filterwarnings('ignore')

from scipy import interpolate


# # Reading data from CSVs

# In[ ]:


readdata = pd.DataFrame()

filenames = glob.glob("*.csv")
for filename in filenames:
    f = pd.read_csv(filename)
    readdata = readdata.append(f, ignore_index=True)


# In[ ]:


alldata = readdata.copy()


# In[ ]:


# Cleaning:

alldata["k"] = alldata["k"].astype(int)
alldata["N"] = alldata["N"].astype(int)
alldata["timepoint"] = alldata["timepoint"].astype(int)

alldata = alldata[ alldata["k"] >= 1 ]
alldata = alldata[alldata["N"] >= alldata["k"] ]

alldata = alldata[alldata["C"] >= alldata["k"] ]

alldata = alldata[ alldata["timepoint"] != 0 ]


# # Figure 5. Comparison of optimal k for oracle and a retailer following our algorithm at different levels of C.

# In[71]:


C_set = [2,3,4,5,6,8,10,12,15]
comparison_data = alldata[alldata["C"].isin(C_set)]


# In[72]:


comparison_data = comparison_data.groupby(["Algorithm", "N", "C", "k", 'timepoint'],
                                  as_index=False).mean()

mydata = comparison_data[comparison_data["Algorithm"] == "TS"]


# In[74]:


mydata["Gain (%)"] = 100* mydata["expected revenue"] / mydata["Optimum revenue"]


# In[75]:


# Filter on last 10% times,
# groupby C, k, and mean on Gain%
"""
columns: C/customer

x: final_rev/opt_cap (Gain%)
y: k/N

"""
final_percentage = mydata[(mydata["timepoint"] > 90000)
                          & (mydata["C"] <= 25)][[
                              "C", "k", "expected revenue", "Optimum revenue", "Gain (%)"
                          ]]
final_percentage = final_percentage.groupby(["C", "k"], as_index=False).mean()


# In[76]:


dfs = []
for CC in final_percentage['C'].unique():
    f = final_percentage[ final_percentage['C'] == CC ]
    f["Gain (%) Vs Oracle w/o k"] = 100*f["expected revenue"] / f["Optimum revenue"].max()
    f["Optimal k for Agent"] = f.loc[f['expected revenue'].idxmax()]["k"]
    if CC < 10:
        f["Optimal k for Oracle"] = CC
    else:
        f["Optimal k for Oracle"] = 10
    dfs.append(f)
final_percentage = pd.concat(dfs, ignore_index=True)


# In[77]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(5.1667, 5.1667))

ax.plot('C', 'Optimal k for Agent', data=final_percentage, label='Agent', color='#1038ba', linewidth=2)
ax.plot('C', 'Optimal k for Oracle', data=final_percentage, label='Oracle', color='#b53d1c', linestyle='dashed', linewidth=2)
ax.set(ylim=(0, 11))
ax.set_xlabel('$C$')
ax.set_ylabel('$k$')

plt.legend(loc='best')

# SAVE
plt.savefig("Optimal_k.png", dpi=300, bbox_inches='tight')

