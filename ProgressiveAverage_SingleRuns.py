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

# # Progressive average comparison

# +
import numpy as np
import qutip as qt # handles all the quantum operations
import qfunk.generator as gg # used to randomly sample states 
import stim 

import random # local random generator (i.e. for random observable)
import time   # for time comparison and optimization
import sys
import os

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

def bunch_collect(bunch, param_to_bunch, i_param):
    return [np.concatenate([t.eval for t in bunch if (t.parameters[i_param] == p) ]) for p in param_to_bunch]         


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
# Since all these different runs are characterised by many parameters and require ulterior post-processing, maybe it's better to gather all the information in a class for easier handling. This also helps keeping post-processing internal and not messing up all different runs…
# Members of class:
# - header
# - path to state and observable
# - parameters : 
#     - state size (number of qubits)
#     - locality (to be implemented)
#     - locality of measurements (number of qubits on which the measurement is performed on)
#     - reference $\varepsilon$ -> essentially, only influences the lenght of the vector
#     - Actual expectation value
# - Methods:
#     - progressive average (with possible reshuffling…)
#     - progressive variance (obviously corresponding to the progressive average)
#     

# +
# indices
i_exp  = 0
i_exp2 = 1

i_avg = 0
i_var = 2


# -

class cs_run ():
    def __init__ (self, head, parameters, run_index):
        # assigns characteristic variables
        self.header = head
        '''
        self.n_qubits = parameters[i_ntot]
        self.n_in_block  = parameters[i_nblock]
        self.epsilon  = parameters[i_eps]
        self.obs_ind  = parameters[i_obs]
        '''
        self.parameters = parameters
        self.index    = run_index
        
        self.locality = self.parameters[i_ntot]

        # identifies path to relevant files
        self.define_paths()
        
        # loads data in array
        self.eval   = np.real(np.load(self.file_name)[i_exp])
        self.eval2  = np.real(np.load(self.file_name)[i_exp2])
        
        # finds actual expectation value
        self.true_eval = self.find_eval()
        
        # progressive averages and variances
        self.prog_avg_std = prog_avg(self.eval)
        self.prog_var_statistical = prog_var (self.eval,self.eval2)
        self.prog_var_effective   = actual_prog_var(self.eval, self.true_eval)
        
        self.avg = self.prog_avg_std[-1]
        
    def define_paths (self):   # identifies uniquely state, observable and particular file path
        main_dir, obs_dir, dir_name = define_dir_name(self.header, self.parameters)
        
        self.state_path = main_dir + f"{self.header}_state_{self.parameters[i_ntot]}_qubits.npy"
        self.obs_path = obs_dir + f"{self.header}_obs{self.parameters[i_obs]}_{self.parameters[i_ntot]}_qubits.npy"
        
        filename = define_file_name (self.parameters, self.index)
        self.file_name = dir_name + filename + ".npy"
        
        return
    
    def find_eval(self):
        rho = qt.Qobj(np.array(np.load(self.state_path)))
        obs = qt.Qobj(np.array(np.load(self.obs_path)))
        
        return np.real((rho*obs).tr())
            
    def prefactor (self):
        return (2**self.parameters[i_nblock]+1)**(self.locality/self.parameters[i_nblock])


# + [markdown] tags=[]
# ## Parameters considered
# -

# ### Load function
# Generic load function which takes parameters from a particular set, a "bunch" so to say, and automatically loads a vector of `cs_run` for each value corresponding to the particular header. This is to use data from different *bunches* at the same time

def load_data_from_bunch(header, possible_parameters):
    all_runs = []

    for p in itertools.product(*possible_parameters):
        param = list(p)
        main, obs, dir_name = define_dir_name (header, param)
        tot_ind = len(os.listdir(dir_name))
        for i in range(0,tot_ind):
            all_runs.append(cs_run(header, param, i))
        
    return all_runs


# +
# First bunch
header = "1Tata"

n_qubits = [4]
n_blocks = [1]
epsilon = [0.5]
no_diff_obs_tata = 4

possible_parameters = [np.arange(0, no_diff_obs_tata), n_qubits, n_blocks , epsilon] 

tata_bunch = load_data_from_bunch(header,possible_parameters)
# -

# ### Data elaboration

color_array = ["xkcd:azure", "xkcd:kelly green", "xkcd:red", "xkcd:bright orange"]
dark_color  = ["xkcd:navy blue", "xkcd:forest green", "xkcd:scarlet", "xkcd:burnt orange"]

# + [markdown] tags=[]
# #### Tata
# - $N_{qb} = 4$
# - $N_{block} = 1$
# - $\varepsilon = 0.5$
# -

# ##### Values distribution

# +
fig,axs = plt.subplots(4,1,figsize=[10,4*4])

eval_dist = bunch_collect(tata_bunch, np.arange(0,no_diff_obs_tata), i_obs)
true_evals = [t.true_eval for t in tata_bunch if (t.index == 0)]

for index in range(no_diff_obs_tata):
    axs[index].scatter(np.arange(len(eval_dist[index])),eval_dist[index]-true_evals[index]*np.ones(len(eval_dist[index])), color = color_array[index])
    #plt.vlines (true_evals[index], 0, yvec.max(), color = dark_color[index])
# -

# ##### Progressive average

# +
no_diff_obs = no_diff_obs_tata

fig, axs = plt.subplots(no_diff_obs, 1, figsize=[10,no_diff_obs*7])

y_max = 2

sums = np.zeros(no_diff_obs)
tot_tries = np.zeros(no_diff_obs)
lens = np.ones(no_diff_obs)

for t in tata_bunch: 
    expected_prefactor = t.prefactor()
    n_tries = len(t.eval)

    index = int(np.where(t.parameters[i_obs] == np.arange(0,no_diff_obs))[0])
    sums[index] += t.avg
    tot_tries[index]+= 1
    if (n_tries > lens[index] ):
        lens[index] = n_tries
            
    axs[index].plot(np.arange(0,n_tries), t.prog_avg_std-t.true_eval*np.ones(n_tries), color = color_array[index])
    #axs[index].errorbar(np.arange(0,n_tries), t.prog_avg_std, np.sqrt(t.prog_var_effective), errorevery = int(n_tries/10+t.index), color = color_array[i_color])
    axs[index].vlines(expected_prefactor/t.parameters[i_eps]**2, -y_max, y_max, color = dark_color[index])

    if (t.index == 0):
        axs[index].hlines(t.true_eval, 0, n_tries, color=dark_color[index], label = 'actual expectation value')
        axs[index].hlines(t.true_eval - t.parameters[i_eps], 0, n_tries, color=dark_color[index], linestyle='--')
        axs[index].hlines(t.true_eval + t.parameters[i_eps], 0, n_tries, color=dark_color[index], linestyle='--')

avgs = np.divide(sums,tot_tries)
    
for i in range(len(axs)):
    axs[i].set_ylim([0,y_max])
    axs[i].hlines(avgs[i], 0, lens[i], color = dark_color[i], linestyle=':', label = 'Average expectation value')
    axs[i].legend()
    
plt.show()                    

# + [markdown] tags=[]
# #### Calabria
# - $N_{qb} = 4$
# - $N_{block} = 1, 2, 4$
# - $\varepsilon = 0.5, 0.1$

# +
# Second bunch
header = "2Calabria"

n_qubits = [4]
n_blocks = [1,2,4]
epsilon = [0.5,0.1]
no_diff_obs_cala = 1

possible_parameters = [np.arange(0,no_diff_obs_cala), n_qubits, n_blocks , epsilon] 

cala_bunch = load_data_from_bunch(header,possible_parameters)

# +
# all comparisons

no_diff_obs = no_diff_obs_cala

n_blocks_cala = np.array([1,2,4])
epsilon_cala = np.array([0.5,0.1])

rows = len(n_blocks_cala)
cols = 1#len(epsilon_cala)

fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])

y_max = 0.75

sums = np.zeros((rows))
tot_tries = np.zeros((rows))
lens = np.ones((rows))

true_eval = np.zeros((rows))

for t in cala_bunch: 
    expected_prefactor = t.prefactor()
    n_tries = len(t.eval)

    i_row = int(np.where(t.parameters[i_nblock] == n_blocks_cala)[0])

    sums[i_row] += t.avg
    tot_tries[i_row]+= 1
    if (n_tries > lens[i_row] ):
        lens[i_row] = n_tries
            
    axs[i_row].plot(np.arange(0,n_tries), t.prog_avg_std , linewidth =0.5)
    #axs[index].errorbar(np.arange(0,n_tries), t.prog_avg_std, np.sqrt(t.prog_var_effective), errorevery = int(n_tries/10+t.index), color = color_array[i_color])
    
    
true_eval = [np.random.choice(cala_bunch).true_eval]*rows

avgs = np.divide(sums,tot_tries)
    
for r in range(rows):
    axs[r].set_title(f"{n_blocks_cala[r]} qubits measured at the same time")
    axs[r].set_ylim([-y_max,y_max])
    axs[r].hlines(avgs[r], 0, lens[r], color = color_array[r],label='final average')
    axs[r].hlines(true_eval[r], 0, lens[r], color = dark_color[r],label='expected value')

    #axs[r].hlines( epsilon_cala[c], 0, lens[r], color=dark_color[i_row], linestyle='--', label = 'expected deviation')#t.true_eval +
    axs[r].hlines(true_eval[r]+ epsilon_cala[1], 0, lens[r], color = dark_color[r], linestyle = ':')
    axs[r].hlines(true_eval[r]- epsilon_cala[1], 0, lens[r], color = dark_color[r], linestyle = ':')

plt.show()                    

# +
# let's find some use for these shorter simulations:
the_long_run = bunch_collect(cala_bunch, n_blocks_cala, i_nblock)

subind = 5
split_lr = [np.split(tlr, subind) for tlr in the_long_run]

# +
rows = len(n_blocks_cala)
cols = 1
y_max = 0.5
fig, axs = plt.subplots(rows, 1, figsize=[7*cols,5*rows])

actual_eval = np.random.choice(cala_bunch).true_eval

for i in range(len(split_lr)):
    for j in range(subind):
        for k in range(3):
            np.random.shuffle(split_lr[i][j])
            new_prog_avg = prog_avg(split_lr[i][j])
            axs[i].plot(np.arange(0,len(new_prog_avg)), new_prog_avg)
    axs[i].set_ylim([-y_max,y_max])
    axs[i].hlines(actual_eval, 0 , len(new_prog_avg), color='k')

# + [markdown] tags=[]
# #### Maldini
# - $N_{qb} = 4$
# - $N_{block} = 1, 2, 4$
# - $\varepsilon = 0.5$

# +
# Third bunch
header = "3Maldini"

n_qubits = [4]
n_blocks = [1,2,4]
epsilon = [0.1]
no_diff_obs_paolo = 1

possible_parameters = [np.arange(0,no_diff_obs_paolo), n_qubits, n_blocks , epsilon] 

paolo_bunch = load_data_from_bunch(header,possible_parameters)

# +
# all comparisons

no_diff_obs = no_diff_obs_paolo

n_blocks_paolo = np.array([1,2,4])
epsilon_paolo = np.array([0.1])

rows = len(n_blocks_paolo)
cols = 1#len(epsilon_paolo)

fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])

y_max = 0.75

sums = np.zeros((rows))
tot_tries = np.zeros((rows))
lens = np.ones((rows))

true_eval = np.zeros((rows))

for t in paolo_bunch: 
    expected_prefactor = t.prefactor()
    n_tries = len(t.eval)

    i_row = int(np.where(t.parameters[i_nblock] == n_blocks_paolo)[0])

    sums[i_row] += t.avg
    tot_tries[i_row]+= 1
    if (n_tries > lens[i_row] ):
        lens[i_row] = n_tries
            
    axs[i_row].plot(np.arange(0,n_tries), t.prog_avg_std , linewidth =0.5)
    #axs[index].errorbar(np.arange(0,n_tries), t.prog_avg_std, np.sqrt(t.prog_var_effective), errorevery = int(n_tries/10+t.index), color = color_array[i_color])
    
    
true_eval = [np.random.choice(paolo_bunch).true_eval]*rows

avgs = np.divide(sums,tot_tries)
    
for r in range(rows):
    axs[r].set_title(f"{n_blocks_cala[r]} qubits measured at the same time")
    axs[r].set_ylim([-y_max,y_max])
    axs[r].hlines(avgs[r], 0, lens[r], color = color_array[r],label='final average')
    axs[r].hlines(true_eval[r], 0, lens[r], color = dark_color[r],label='expected value', linestyle='--')

    #axs[r].hlines( epsilon_cala[c], 0, lens[r], color=dark_color[i_row], linestyle='--', label = 'expected deviation')#t.true_eval +
    axs[r].hlines( true_eval[r]+ epsilon_paolo[0], 0, lens[r], color = dark_color[r], linestyle = ':')
    axs[r].hlines(true_eval[r]- epsilon_paolo[0], 0, lens[r], color = dark_color[r], linestyle = ':')
    axs[r].legend()
    
plt.show()                    
# -



# let's find some use for these shorter simulations:
the_long_run = bunch_collect(paolo_bunch, n_blocks_paolo, i_nblock)

# +
rows = len(n_blocks_paolo)

cols = 1
y_max = 0.5
fig, axs = plt.subplots(rows, cols, figsize=[7*cols,5*rows])

actual_eval = np.random.choice(paolo_bunch).true_eval

for i in range(len(the_long_run)):
    for j in range(10):
        np.random.shuffle(the_long_run[i])
        new_prog_avg = prog_avg(the_long_run[i])
        axs[i].plot(np.arange(0,len(new_prog_avg)), new_prog_avg)
    axs[i].set_ylim([-y_max,y_max])
    axs[i].hlines(actual_eval, 0 , len(new_prog_avg), color='k')
# -


