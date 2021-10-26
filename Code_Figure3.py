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


# # Figure 3. Comparison of the performance of our Algorithm to that of Agrawal et al.'s (2019b) and Greedy
# 
# ## C25, k4

# In[56]:


alldata["Gain (%)"] =100* alldata["expected revenue"] / alldata["Optimum revenue"]


# In[ ]:


comparison_data = alldata.groupby(["Algorithm", "N", "C", "k", 'timepoint'],
                                  as_index=False).mean()

comparison_data.reset_index(drop=True, inplace=True)


# ### Smoothing the comparison curves:

# In[59]:


nsample = 500
C = 25
mydata = comparison_data[comparison_data["Algorithm"] == "TS"]


# In[ ]:


mydata_C25_k4 = mydata[(mydata["C"] == C) & (mydata["k"] == 4)]
mydata_C25_k4.reset_index(drop=True, inplace=True)

a_BSpline = interpolate.make_interp_spline(
    np.array(mydata_C25_k4["timepoint"]),
    np.array(mydata_C25_k4["Gain (%)"]))

x_my = np.linspace(mydata_C25_k4["timepoint"][0], mydata_C25_k4["timepoint"][9999], nsample)

y_my = a_BSpline(x_my)


# In[ ]:


greedy_data = comparison_data[comparison_data["Algorithm"] == "greedy"]
greedy_data_C50_k4 = greedy_data[(greedy_data["C"] == C)
                                 & (greedy_data["k"] == 4)]
greedy_data_C50_k4.reset_index(drop=True, inplace=True)

a_BSpline = interpolate.make_interp_spline(
    np.array(greedy_data_C50_k4["timepoint"]),
    np.array(greedy_data_C50_k4["Gain (%)"]))

x_greedy = np.linspace(greedy_data_C50_k4["timepoint"][0],
                    greedy_data_C50_k4["timepoint"][9999], nsample)

y_greedy = a_BSpline(x_greedy)


# In[ ]:


Agrawal_data = comparison_data[comparison_data["Algorithm"] == "TS-Agrawal"]

Agrawal_data_C50_k4 = Agrawal_data[(Agrawal_data["k"] == 4)]
Agrawal_data_C50_k4.reset_index(drop=True, inplace=True)

a_BSpline = interpolate.make_interp_spline(
    np.array(Agrawal_data_C50_k4["timepoint"]),
    np.array(Agrawal_data_C50_k4["Gain (%)"]))

x_Agrawal = np.linspace(Agrawal_data_C50_k4["timepoint"][0],
                    Agrawal_data_C50_k4["timepoint"][9998], nsample)

y_Agrawal = a_BSpline(x_Agrawal)


# In[63]:


comparison_plot = pd.DataFrame({
    "Constrained MNL-TS": y_my,
    "Greedy": y_greedy,
    "Agrawal et al.": y_Agrawal
}, index=x_my)


# In[64]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

f, ax = plt.subplots(figsize=(2*5.1667, 5.1667), dpi=300)
sns.lineplot(data=comparison_plot, palette="dark", ci=None, linewidth=2 )
plt.xscale('log')
plt.title('$C$ = '+str(C))
#ax.grid(which='both', axis='y')

ax.set_xlabel("$T$ (in log scale)")
ax.set_ylabel("Gain (%)")

plt.show()

# SAVE
plt.savefig("comparison_C25.png", dpi=300, bbox_inches='tight')

