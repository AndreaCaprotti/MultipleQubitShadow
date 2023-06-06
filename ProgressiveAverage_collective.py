# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Progressive average analysis
# Aims to study the behaviour and scaling of expected value, compared to the actual average, by considering all the data which has in common:
# - Target state
# - Observable (and thus expectation value)
# - Locality of measurement, i.e. how many qubits are measured at the same time
#
# Most of methods are simply inherited from the previous post-processing notebook, I've decided to separate them to keep better order.

# +
import numpy as np
import qutip as qt # handles all the quantum operations
import qfunk.generator as gg # used to randomly sample states 
import stim 

import random # local random generator (i.e. for random observable)
import time   # for time comparison and optimization
import sys
import os
import glob

import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -

# ## Load functions

# ### File name and directory

# +
# for now, I can just think of these…
param_title = ["_obs",
               "_ntot",
               "_nblock",
               "_eps"
              ]

i_obs  = 0
i_ntot = 1
i_nblock = 2
i_eps = 3
i_loc = 4 # announced, TBD


# -

def define_file_name (params, index): # thought to be, eventually, extended for more parameters
    starting_block = f"{index}"       # and on this, it doesn't rain (italian expression)
    for i in range(len(params)):
        starting_block+=(param_title[i]+f"_{params[i]}")
    
    return starting_block


def define_dir_name(dir_header, parameters):# identifies directory name
    obs_ind = parameters[i_obs]
    fixed_dir = '/Users/andrea/vienna_data/clifford_data/' 
    dir_specific = define_file_name (parameters, f"{dir_header}_{obs_ind}")
    main_dir = fixed_dir+dir_header+'/'
    obs_dir  = main_dir + f"obs{obs_ind}/" 
    dir_name = obs_dir+dir_specific+'/'
    
    return main_dir, obs_dir, dir_name


# ## Statistical functions

# +
def prog_avg (vec):
    avg_array = []
    avg = 0
    
    for n in range(len(vec)):
        avg += np.real(vec[n])
        avg_array.append(avg/(n+1))
    
    return np.array(avg_array)

def prog_var (avg_vec, var_vec):
    var_array = []
    var = 0
    avg = 0
    if(len(avg_vec)!=len(var_vec)):
         raise NameError("Vectors are not the same size")
    
    for n in range(len(avg_vec)):
        avg += np.real(avg_vec[n])
        var += var_vec[n]
        var_array.append((var/(n+1)-(avg/(n+1))**2)+1e-10)
    
    return np.array(var_array)

def actual_prog_var (vec_avg, exp_val):
    avg_array = []
    avg = 0
    
    for n in range(len(vec_avg)):
        avg += (np.real(vec_avg[n]) - exp_val)**2
        avg_array.append(avg/(n+1))
    
    return np.array(avg_array)


# + [markdown] tags=[]
# ## Data class
# Compared to the previous class, this gathers all the relevant data: essentially, an element of these corresponds to snapshot of same state (and, for now, also observable). That is, all the data should correspond to the same expectation value
#
# Members of class:
# - header
# - path to state and observable
# - parameters : 
#     - state size (number of qubits)
#     - list of locality of measurements (number of qubits on which the measurement is performed on)
#     - Actual expectation value
# - Methods:
#     - reshuffling and division based on accuracy (external parameter)
#     - progressive average for each of these smaller "bunches" or over all the vector
#     - progressive variance (obviously corresponding to the progressive average)
#     

# +
# indices
i_exp  = 0
i_exp2 = 1

i_avg = 0
i_var = 2


# -

class cs_collection ():
    def __init__ (self, head, obs_ind, tot_qubits, qb_per_block):
        # assigns characteristic variables
        self.header = head
        
        self.n_qubits = tot_qubits
        self.qb_per_block = np.array(qb_per_block)
        self.obs_ind = obs_ind

        self.foobar = "*" # not defined a priori, obsolete parameter which has come back to haunt me
        self.len_fact = 2 # used to define how long a trajectory is
        
        # identifies path to relevant files
        self.__define_paths()
        # finds actual expectation value
        self.__true_eval = self.__find_eval()
        
        # loads data in array
        self.__eval   = []    
        self.__eval2  = []
        self.__avg_eval = np.zeros(len(self.qb_per_block))
        
        self.__load_all()
        return
        
    def __define_paths (self):   # identifies uniquely state, observable and particular file path
        fixed_dir = '/Users/andrea/vienna_data/clifford_data/' 
        main_dir = fixed_dir+self.header+'/'
        obs_dir  = main_dir + f"obs{self.obs_ind}/" 
        
        self.state_path = main_dir + f"{self.header}_state_{self.n_qubits}_qubits.npy"
        self.obs_path = obs_dir + f"{self.header}_obs{self.obs_ind}_{self.n_qubits}_qubits.npy"    
        return
    
    def __find_eval(self):
        rho = qt.Qobj(np.array(np.load(self.state_path)))
        obs = qt.Qobj(np.array(np.load(self.obs_path)))
        
        return np.real((rho*obs).tr())
    
    def get_true_eval(self):
        return self.__true_eval
    
    def get_final_eval(self, n_qb):
        i_block = self.corresponding_index(n_qb)
        return self.__avg_eval[i_block]
    
    def __load_all(self):              # in order to guarantee that order of evals is the same as qb_block
        print(f"Commencing automatic data load for state {self.header}")
        for qb in self.qb_per_block:
            i = self.corresponding_index(qb)
            self.load_data_per_qb(qb)
            self.__avg_eval[i] = self.final_avg(qb)
            print(f"Correctly loaded data for {qb} qubits per block")
        return
        
    def load_data_per_qb(self, n_qb): # for now, it just finds all relevant data with the same header, n_qb and n_block
        i_block=self.corresponding_index(n_qb) 
        params = [self.obs_ind, self.n_qubits, n_qb, self.foobar]
        main, obs, dir_name = define_dir_name (self.header, params)
        file_name = define_file_name(params, self.foobar)
        relevant_files = glob.glob(dir_name+file_name)
        
        for file in relevant_files:
            foo_vec = np.real(np.load(file))
            self.__increase_eval(i_block, foo_vec[i_exp],foo_vec[i_exp2])
        return
        
    def __increase_eval(self, i_block, vec_eval, vec_var):
        if (len(self.__eval) < i_block+1):
            self.__eval.append(vec_eval)
            self.__eval2.append(vec_var)
        else:
            self.__eval[i_block]  = np.concatenate([self.__eval[i_block],vec_eval])
            self.__eval2[i_block] = np.concatenate([self.__eval2[i_block],vec_var])
        return
            
    def final_avg (self, n_qb): # returns the corresponding average, given localilty of measurement
        i_block = self.corresponding_index(n_qb)
        return np.sum(self.__eval[i_block])/len(self.__eval[i_block])
        
    def corresponding_index(self, n_qb):
        return int(np.where(n_qb == self.qb_per_block)[0])
    
    def prefactor (self,n_block):
        return (2**n_block+1)**(self.n_qubits/n_block)
    
    def expected_scale (self,n_block,eps):
        return self.len_fact*self.prefactor(n_qb)/eps**2

    def __shuffle_blocks(self, index):
        assert len(self.__eval[index]) == len(self.__eval2[index])
        p = np.random.permutation(len(self.__eval[index]))
        self.__eval[index] = self.__eval[index][p]
        self.__eval2[index] = self.__eval2[index][p]
        return 
    
    def __qb_divide(self, vec, n_qb, eps):
        #print(len(vec), self.expected_scale(n_qb, eps))
        no_chunks = int(len(vec) / self.expected_scale(n_qb, eps))
        return np.array_split(vec, no_chunks)
        
    def progressive_traj(self, n_qb, eps, shuffle):
        i_block = self.corresponding_index(n_qb)
        
        if (shuffle):
            self.__shuffle_blocks(i_block)
        
        split_data = self.__qb_divide(np.array(self.__eval[index]), n_qb, eps)
        split_var  = self.__qb_divide(np.array(self.__eval2[i_block]), n_qb, eps)
        
        return [prog_avg(chunk) for chunk in split_data], [prog_var(data, var) for data, var in zip(split_data,split_var)]
    
    def cheating_progressive_traj(self, n_qb):
        i_block = self.corresponding_index(n_qb)
        self.__shuffle_blocks(i_block)
        
        return prog_avg(self.__eval[i_block]), prog_var(self.__eval[i_block],self.__eval2[i_block])


# ## Bennacer data

# +
header = "4Bennacer"
qb_tot = 8
qb_block = [1,2,4,8]
no_obs = 0

isma = cs_collection(header, no_obs, qb_tot, qb_block)

# -

# ### Progressive average comparison

# +
isma.len_fact = 1.5
rows = len(qb_block)
cols = 1
eps = 0.5
ymax = 2

isma_exp_val = isma.get_true_eval()

fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])#, sharex = True)

for n_qb in qb_block:
    index = isma.corresponding_index(n_qb)
    eval_block = isma.get_final_eval(n_qb)
    prefactor = isma.prefactor(n_qb)
    eps_len = isma.len_fact*prefactor/eps**2
    
    avg, var = isma.progressive_traj(n_qb, eps, False)
    for actual_vec in avg:
        axs[index].plot(np.arange(len(actual_vec)),actual_vec, linewidth = 0.3)
    
    axs[index].hlines(np.sign(eval_block)*isma_exp_val + eps, 0, eps_len, linewidth = 2, linestyle=':')
    axs[index].hlines(np.sign(eval_block)*isma_exp_val, 0, eps_len, linewidth = 2, label = 'expected value')
    axs[index].hlines(np.sign(eval_block)*isma_exp_val - eps, 0, eps_len, linewidth = 2, linestyle=':' )
    axs[index].hlines(eval_block, 0, eps_len, linewidth = 2, linestyle = '--', label = 'final average')
    
    axs[index].vlines(eps_len, -ymax, ymax, linestyle = '--')
    
    axs[index].set_ylim(-ymax,ymax)
    #axs[index].set_xscale('log')
    axs[index].legend()
    axs[index].set_title(f"Measurements on {n_qb} qubits at the same time")

# + [markdown] tags=[]
# ### **Cheating** progressive average
# Longer progressive average on all data

# +
isma.len_fact = 1.5
rows = len(qb_block)
cols = 1
epsilons = [1,0.5,0.3]
colors = ['xkcd:azure', 'xkcd:bright orange', 'xkcd:kelly green']

ymax = 1

n_traj = 30

isma_exp_val = isma.get_true_eval()

fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])#, sharex = True)

for n_qb in qb_block:
    index = isma.corresponding_index(n_qb)
    eval_block = isma.get_final_eval(n_qb)
    prefactor = isma.prefactor(n_qb)
    
    for j in range(n_traj):
        avg, var = isma.cheating_progressive_traj(n_qb)
        axs[index].plot(np.arange(len(avg)), avg, linewidth=0.3)

    axs[index].hlines(np.sign(eval_block)*isma_exp_val, 0, len(avg), linewidth = 2, label = 'expected value', color='k')
    axs[index].hlines(eval_block, 0, len(avg), linewidth = 2, linestyle = '--', label = 'final average', color='k')
        
    for i in range(len(epsilons)):
        eps = epsilons[i]
        eps_len = prefactor/eps**2
        axs[index].hlines(np.sign(eval_block)*isma_exp_val + eps, 0, len(avg), linewidth = 2, linestyle=':',color=colors[i])
        axs[index].hlines(np.sign(eval_block)*isma_exp_val - eps, 0, len(avg), linewidth = 2, linestyle=':',color=colors[i] )
        axs[index].vlines(eps_len, -ymax, ymax, linestyle = '--', label = f"$\epsilon={eps}$",color=colors[i])
    
    axs[index].set_ylim(-ymax,ymax)
    #axs[index].set_xscale('log')
    axs[index].legend()
    axs[index].set_title(f"Measurements on {n_qb} qubits at the same time")
# -

# ### Progressive variance comparison


