#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy
import pandas
import itertools
import pp #Parallel Python module: https://pypi.org/project/pp/
import datetime


# In[ ]:


# # Functions:
# 
#     """
#     * v_env is the real utilities of products and assumed: v_i <= v_nopurchase=1
#     * ~vi : a geometric variable of number of purchases observed.
#     * Theta: samples from Beta(n, V).
#     * mu:= 1/theta - 1 is a RV that represents vi the number of demand.
#     * theta is a probability which is inverse of probabilities to be selected.
#     * Beta(ni, Vi) is the distribution over theta, that has higher probabilities over 
#       low thetas for good products, Because of inversed alpha and beta.
#     
#     """

# ## Scenario (constants) generator:
# 
# * Get N, n, k, C, Period_time, Total_horizon, customer arrival rate lambda
# 
# * Generate N products, (n categories), real v (including the no-purchase as the v0 = 1) and profit vector
# 
# * Initialize the prior parameters vi = 1 , ni = 1 for i = 1, … , N


def scenario_gen(N):
    # v_env is the real utilities of products and assumed: v_i <= v_nopurchase=1
    v_env = numpy.random.uniform(0, 1, N)
    # revenue:
    r = numpy.ones(N)
    # Initial prior parameters:
    V = numpy.ones(N)
    n = numpy.ones(N)
    return (v_env, r, V, n)


# ## Demand generator:
# 

def customer_type_determinator(S, v_env):
    # S is the offered assortment
    # v_env: the vector of real utilities of products.

    # Nulling the probabilities of absent products:
    mu = S * v_env
    sum_mu = numpy.sum(mu)
    proba = mu / (1 + sum_mu)
    cum_proba = numpy.cumsum(proba)
    u = numpy.random.uniform()
    i = -1  # No-purchase
    for j in range(len(cum_proba)):
        if u <= cum_proba[j]:
            i = j
            break
    return (i)


# ## Posterior sampler:


# ### 1.4 Find the best Action of Agent:
def mu_sampler_TS(n, V):
    theta = numpy.random.beta(n, V)
    mu = 1 / theta - 1  # mu is an RV vector that represents vi the number of demand.
    return mu


def mu_sampler_greedy(n, V):
    """
    Just an average:
    E[X]= alpha/alpha+\beta
    """
    theta = n / (n + V)
    mu = 1 / theta - 1  # mu is an RV that represents vi the number of demand.
    return mu


# ## Find the best Action of Agent:
# 

def expected_rev_per_customer(S, mu, r):
    """
    mu: vector
    r: revenue, vector
    In the environment side, mu = v_env,
    We first remove the nonprovided goods, then calculate the probabilities 
     based on v_env for present goods, at last, calculate the R
    """
    mu = S * mu
    R = numpy.dot(r, mu) / (1 + numpy.sum(mu) )
    return R


# An assortment optimizer for expected_rev_per_customer:
def assortment_optimizer(mu, r, N, k):
    # Generating all possible subsets of length N:
    all_combinations = numpy.array(list(map(list, itertools.product([0, 1], repeat=N))))
    # Now that r = 1 for all:
    assortment_search_domain = [x for x in all_combinations if numpy.sum(x) == k]
    numpy.random.shuffle(assortment_search_domain)

    Max_R = 0
    best_S = numpy.zeros(N)

    for S in assortment_search_domain:
        # To Compare:
        # if sum(S) == k, Sometimes sum(S) < k is better because new
        # less probable product reduces the service level of other better products.
        R = expected_rev_per_customer(S, mu, r)
        if R > Max_R:
            Max_R = R
            best_S = S
    return (best_S, Max_R)


def assortment_optimizer_different_r(mu, r, N, k):
    # Generating all possible subsets of length N:
    all_combinations = numpy.array(list(map(list, itertools.product([0, 1], repeat=N))))
    assortment_search_domain = [x for x in all_combinations if numpy.sum(x) <= k]
    numpy.random.shuffle(assortment_search_domain)
    Max_R = 0
    best_S = numpy.zeros(N)
    for S in assortment_search_domain:
        R = expected_rev_per_customer(S, mu, r)
        if R > Max_R:
            Max_R = R
            best_S = S
    return (best_S, Max_R)


# ##  Inventory Capacity Allocator:
# 


def capacity_allocator(S, mu_sample, C):
    mu_sample = S * mu_sample
    inventory = S.copy()
    for i in range(len(S)):
        # Dividing according to the purchase probability:
        # in inventory planning we don't plan for no-purchasers intentionally, so no sum_mu+1
        inventory[i] += numpy.floor( (C-sum(S)) * mu_sample[i] / sum(mu_sample) )
        
    return(inventory, S)


def capacity_equal_allocator(S, mu_sample, C):
    qouta = numpy.floor(C / sum(S))
    inventory = numpy.ones(len(S)) * qouta
    inventory = inventory * S
    return(inventory, S)


def capacity_unlimited_allocator(S, mu_sample, C):
    inventory = numpy.ones(len(S)) * 10**7
    inventory = inventory * S
    return(inventory, S)


# In[ ]:





# In[ ]:


# # Simulation
# 

# ## Our Model: TSATA
# 
# * Time of system or customer arrival : t
# * Replenishment Periods: l

def TSATA(N, k, C, avg_customers_pp, Total_horizon, numsim, algo_type):
    res = pandas.DataFrame({
        'Algorithm': [],
        'timepoint': [],
        'expected revenue': [],
        'Optimum revenue': [],
        'N': [],
        'C': [],
        'k': []
        })
    
    for sim in range(numsim):
        (v_env, r, V, n) = scenario_gen(N)
        (optimal_S, optimal_rev_per_customer) = assortment_optimizer(v_env, r, N, k)
        customers_countor = 0
        
        for l in range(Total_horizon):
            customers_pp = avg_customers_pp
            customers_countor += customers_pp
            # Theoritical Expected Optimum: (Always constant during this sim)
            optimal_rev_per_period = min(C, customers_pp * optimal_rev_per_customer) ## OUTPUT ##
            intraperiod_expected_rev = 0 ## OUTPUT ##
            #drop this epoch, Don’t update the prior, only update the regret, and sampling
            v = numpy.zeros(N)  #Reset observed demand vector
            
            # prep for the future:
            if algo_type == "TS":
                mu_sample = mu_sampler_TS(n, V)
            if algo_type == "greedy":
                mu_sample = mu_sampler_greedy(n, V)
            if algo_type == "e-greedy":
                u = numpy.random.uniform()
                if u <= 0.95:
                    mu_sample = mu_sampler_greedy(n, V)
                else:  # random
                    mu_sample = mu_sampler_TS(numpy.ones(N), numpy.ones(N))

            S = assortment_optimizer(mu_sample, r, N, k)[0]
            (inventory, S) = capacity_allocator(S, mu_sample, C)

            for i in range(customers_pp):
                # If customer is not a no-purchase, he sure buys 1 from current S because S gets updated,
                # so we use expected revenue as is conventional.
                # Expected revenue accounts for no-purchase in itself.
                intraperiod_expected_rev += expected_rev_per_customer(S, v_env, r)
                # Inventory management:
                demanded_i = customer_type_determinator(S, v_env)
                if demanded_i == -1:  #end epoch, update the priors, don’t change the S
                    # Update prior:
                    # Vi = Vi + vi , ni =ni +1 :
                    V = V + v
                    n = n + S
                    v = numpy.zeros(N)  #Reset observed demand vector
                else:
                    v[int(demanded_i)] += 1
                    inventory[int(demanded_i)] -= 1
                    if inventory[int(demanded_i)] == 0:
                        #drop this epoch, start a new epoch with S-i, don’t update the priors
                        S[int(demanded_i)] = 0
                        v = numpy.zeros(N)  #Reset observed demand vector
                if sum(S) == 0 :
                    break
            
            new_row = {
                'Algorithm': algo_type,
                'timepoint': customers_countor,
                'expected revenue': intraperiod_expected_rev,
                'Optimum revenue': optimal_rev_per_period,
                'N': N,
                'C': C,
                'k': k
            }
            res = res.append(new_row, ignore_index=True)
            
    res.to_csv(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S-%f')+'.csv', index = False, header=True)
    return()



# Preparing the scenarios:
scenarios = []

# Global Input Parameters:
N = 10
avg_customers_pp = 10
numsim = 100
Total_horizon = 10000
algo_type = "TS"  # "e-greedy"


for C in [2, 3, 4, 5, 8, 10, 12, 25, 50, 100]:
    for k in numpy.arange(1, C+1):
        scenarios.append([N, k, C, avg_customers_pp, Total_horizon, numsim, algo_type])


# parallelpython:
job_server = pp.Server()
jobs = [
    (row,
     job_server.submit(
         TSATA, tuple(row),
         (scenario_gen, customer_type_determinator, mu_sampler_TS,
          mu_sampler_greedy, expected_rev_per_customer, assortment_optimizer,
          assortment_optimizer_different_r, capacity_allocator,
          capacity_equal_allocator, capacity_unlimited_allocator),
         ("numpy", "pandas", "itertools",
           "datetime"))) for row in scenarios
]

for row, job in jobs:
    exper = job()

