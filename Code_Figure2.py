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


# # Figure 2. Estimating the behavior of regret for two different values of C in the study defined in (4-2).
# 
# ## C = 100
# 
# Best Fit for the Regret of Algorithm on the Parametric Instance:
# k = 4, C = 100

# In[13]:


mydata_C100_k4 = processed_data[(processed_data["Algorithm"] == "TS")&
                               (processed_data["C"] == 100)&
                               (processed_data["k"] == 4)]
mydata_C100_k4 = mydata_C100_k4[["timepoint", "Cumulative regret"]]
mydata_C100_k4 = mydata_C100_k4.groupby('timepoint', as_index=False).mean()


# In[14]:


t = np.arange(10, 100001, 10)
sqrt_t = np.sqrt(t)
log_t = np.log(t)

sqrt_t_df = pd.DataFrame({"timepoint": t, "sqrt_t": sqrt_t})

log_t_df = pd.DataFrame({"timepoint": t, "log_t": log_t})


# In[15]:


reg_df = pd.merge(mydata_C100_k4, sqrt_t_df, on="timepoint", how="left")
reg_df = pd.merge(reg_df, log_t_df, on="timepoint", how="left")


# ## sqrt_t

# In[16]:


X = sm.add_constant(reg_df['sqrt_t'])
fitted_model = sm.OLS(reg_df['Cumulative regret'], X, missing='drop').fit()
b0 = fitted_model.params[0]
b1 = fitted_model.params[1]


# In[17]:


reg_df["y = "+ str(round(b0,2))+" + "+str(round(b1,2))+ r'$\sqrt{T}$'] = b0 + b1*np.sqrt(reg_df["timepoint"])


# ## log_t

# In[18]:


X = sm.add_constant(reg_df['log_t'])
fitted_model2 = sm.OLS(reg_df['Cumulative regret'], X, missing='drop').fit()
b0 = fitted_model2.params[0]
b1 = fitted_model2.params[1]


# In[19]:


reg_df["y = "+ str(round(b0,2))+" + "+str(round(b1,2))+" log$T$"] = b0+1.9 + b1*np.log(reg_df["timepoint"])


# In[20]:


del reg_df["sqrt_t"]
del reg_df["log_t"]
reg_df.rename(columns={"Cumulative regret": "Constrained MNL-TS Algorithm"}, inplace=True)


# In[21]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(6.1667, 6.1667))

sns.lineplot(x='timepoint', y='Cumulative Regret', hue='variable', style='variable', 
             data=pd.melt(reg_df, ['timepoint']).rename(columns={"value": "Cumulative Regret"}),
             linewidth=2, palette="dark")
ax.set(ylim=(-40, 125))
#plt.title("(a) $C$ = 100")
ax.set_xlabel('$T$')
ax.set_ylabel('Cumulative regret')

plt.legend(loc='lower right')

# SAVE
plt.savefig("regret_order_C100.png", dpi=300, bbox_inches='tight')


# ## C = 25
# 
# Best Fit for the Regret of Algorithm 1 on the Parametric Instance:
# k = 4, C = 25

# In[22]:


mydata_C25_k4 = processed_data[(processed_data["Algorithm"] == "TS")&
                               (processed_data["C"] == 25)&
                               (processed_data["k"] == 4)]
mydata_C25_k4 = mydata_C25_k4[["timepoint", "Cumulative regret"]]
mydata_C25_k4 = mydata_C25_k4.groupby('timepoint', as_index=False).mean()


# In[23]:


t = np.arange(10, 100001, 10)
sqrt_t = np.sqrt(t)
log_t = np.log(t)

sqrt_t_df = pd.DataFrame({"timepoint": t, "sqrt_t": sqrt_t})

log_t_df = pd.DataFrame({"timepoint": t, "log_t": log_t})


# In[24]:


reg_df = pd.merge(mydata_C25_k4, sqrt_t_df, on="timepoint", how="left")
reg_df = pd.merge(reg_df, log_t_df, on="timepoint", how="left")


# ## sqrt_t

# In[25]:


X = sm.add_constant(reg_df['sqrt_t'])
fitted_model = sm.OLS(reg_df['Cumulative regret'], X, missing='drop').fit()
b0 = fitted_model.params[0]
b1 = fitted_model.params[1]


# In[26]:


reg_df["y = "+ str(round(b0,2))+" + "+str(round(b1,2))+ r'$\sqrt{T}$'] = b0 + b1*np.sqrt(reg_df["timepoint"])


# ## log_t

# In[27]:


X = sm.add_constant(reg_df['log_t'])
fitted_model2 = sm.OLS(reg_df['Cumulative regret'], X, missing='drop').fit()
b0 = fitted_model2.params[0]
b1 = fitted_model2.params[1]


# In[28]:


reg_df["y = "+ str(round(b0,2))+" + "+str(round(b1,2))+" log$T$"] = b0 + b1*np.log(reg_df["timepoint"])


# In[29]:


del reg_df["sqrt_t"]
del reg_df["log_t"]
reg_df.rename(columns={"Cumulative regret": "Constrained MNL-TS Algorithm"}, inplace=True)


# In[30]:


# Set fontsize:
plt.rcParams.update({'font.size': 9}) # Pick between 7 and 10

# A square large figure:
fig, ax = plt.subplots(figsize=(6.1667, 6.1667))

sns.lineplot(x='timepoint', y='Cumulative Regret', hue='variable', style='variable', 
             data=pd.melt(reg_df, ['timepoint']).rename(columns={"value": "Cumulative Regret"}),
             linewidth=2, palette="dark")
ax.set(ylim=(-40, 125))
#plt.title("(b) $C$ = 25")
ax.set_xlabel('$T$')
ax.set_ylabel('Cumulative regret')

plt.legend(loc='lower right')

# SAVE
plt.savefig("regret_order_C25.png", dpi=300, bbox_inches='tight')

