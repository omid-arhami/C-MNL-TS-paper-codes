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


# # processing data:

# In[ ]:


data_to_process = alldata[
    ((alldata["Algorithm"] == "greedy") & (alldata["C"] == 25) & (alldata["k"] == 4)) |
    ((alldata["Algorithm"] == "TS-Agrawal") & (alldata["k"] == 4)) |
    ((alldata["Algorithm"] == "TS") & (alldata["k"] == 4) & (alldata["C"].isin([5,8,10,12,25,50,100])))
]


# In[ ]:


nsim = 100
grouped = data_to_process.groupby(["Algorithm", "C", "k"])
dfs = []
for name, group in grouped:
    f = grouped.get_group(name)
    for i in range(nsim):
        g = f.iloc[i*10000:(i+1)*10000 , :]
        g["regret"] = g["Optimum revenue"] - g["expected revenue"]
        g["Cumulative regret"] = g["regret"].cumsum()
        dfs.append(g)
        
processed_data = pd.concat(dfs, ignore_index=True)


# In[ ]:





# # Figure 1. Performance of the algorithm for different values of C in the numerical study defined in (4-2)

# ## C = 25, 50

# In[4]:


C_set = [25,50]
sublin_data = processed_data[processed_data["C"].isin(C_set)]


# In[ ]:


sublin_data["C"] = sublin_data["C"].astype(int)
sublin_data["k"] = sublin_data["k"].astype(int)
sublin_data["N"] = sublin_data["N"].astype(int)
sublin_data["timepoint"] = sublin_data["timepoint"].astype(int)


# In[6]:


# Only TS (Removing greedy with k=4) :
mydata = sublin_data[(sublin_data["Algorithm"] == "TS")&
                        (sublin_data["k"] == 4)]


# In[8]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(5.1667, 5.1667))
sns.lineplot(data=mydata, x="timepoint", y="Cumulative regret", 
             style="C", hue="C", palette="dark", ci=None, linewidth=2)

#ax.set(ylim=(0, 800))
#plt.title("(a)")
ax.set_xlabel('$T$')
ax.set_ylabel('Cumulative regret')
#plt.grid(which='both', axis='both')

plt.legend(loc='best', labels = ['$C$ = 25', '$C$ = 50'])

# SAVE:
plt.savefig("Cumulative_regret_25_50.png", dpi=300, bbox_inches='tight')


# ## C = 10,12

# In[9]:


C_set = [10,12]
sublin_data = processed_data[processed_data["C"].isin(C_set)]


# In[ ]:


sublin_data["C"] = sublin_data["C"].astype(int)
sublin_data["k"] = sublin_data["k"].astype(int)
sublin_data["N"] = sublin_data["N"].astype(int)
sublin_data["timepoint"] = sublin_data["timepoint"].astype(int)


# In[11]:


# Only TS :
mydata = sublin_data[(sublin_data["Algorithm"] == "TS")&
                        (sublin_data["k"] == 4)]


# In[12]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(5.1667, 5.1667))
sns.lineplot(data=mydata, x="timepoint", y="Cumulative regret", 
             style="C", hue="C", palette="dark", ci=None, linewidth=2)

#ax.set(ylim=(0, 800))
#plt.title("(b)")
ax.set_xlabel('$T$')
ax.set_ylabel('Cumulative regret')
#plt.grid(which='both', axis='both')

plt.legend(loc='best', labels = ['$C$ = 10', '$C$ = 12'])

# SAVE
plt.savefig("Cumulative_regret_10_12.png", dpi=300, bbox_inches='tight')

