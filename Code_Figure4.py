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


# # Figure 4. Performance of our algorithm at different levels of C, compared to an oracle with the same constraints.

# In[65]:


alldata["Gain (%)"] =100* alldata["expected revenue"] / alldata["Optimum revenue"]


# In[66]:


comparison_data = alldata.groupby(["Algorithm", "N", "C", "k", 'timepoint'],
                                  as_index=False).mean()

mydata_k4 = comparison_data[(comparison_data["Algorithm"] == "TS")
                            & (comparison_data["k"] == 4)
                           & (comparison_data["C"].isin([5,8,10,12,25,50]))]


# In[68]:


# Filter on last 10% times
final_percentage_k4 = mydata_k4[ mydata_k4["timepoint"] >90000 ][["C", "Gain (%)"]]
final_percentage_k4["C"] = final_percentage_k4["C"].astype(int)


# In[69]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(5.1667, 5.1667), dpi=300)
# Hide the top and right spines of the axis
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#ax.grid(which='both', axis='y')

sns.barplot(data=final_percentage_k4,
             x="C",
             y="Gain (%)",
             color='#062182')

ax.set_xlabel('$C$')
ax.set_ylabel('Gain (%)')

# SAVE
plt.savefig("Gain_C.png", dpi=300, bbox_inches='tight')

