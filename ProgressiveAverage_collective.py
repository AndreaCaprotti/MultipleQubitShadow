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


# + [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ## Load functions

# + [markdown] tags=[]
# ### File name and directory

# +
# for now, I can just think of these…
fixed_dir = '/Users/andrea/vienna_data/clifford_data/' 
param_title = ["_obs",
               "_ntot",
               "_nblock",
               "_eps",
               "_meas"
              ]

i_obs  = 0
i_ntot = 1
i_nblock = 2
i_eps = 3
i_meas = 5 


# -

def define_file_name (params, index): # thought to be, eventually, extended for more parameters
    starting_block = f"{index}"       # and on this, it doesn't rain (italian expression)
    for i in range(len(params)):
        starting_block+=(param_title[i]+f"_{params[i]}")
    
    return starting_block


def define_dir_name(dir_header, parameters):# identifies directory name
    obs_ind = parameters[i_obs]
    dir_specific = define_file_name (parameters, f"{dir_header}_{obs_ind}")
    main_dir = fixed_dir+dir_header+'/'
    obs_dir  = main_dir + f"obs{obs_ind}/" 
    dir_name = obs_dir+dir_specific+'/'
    
    return main_dir, obs_dir, dir_name


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
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


# +
def rolling_avg(vec, avg_range):
    length = len(vec)
    roll_array = np.zeros(length)
    
    for i in range(length):
        min_ind = max(0,i-avg_range)      # checks possible overflow of data
        max_ind = min(length-1, i+avg_range)
        
        for j in [min_ind, max_ind]:
            roll_array[j]+=vec[i]
    
    return roll_array/avg_range

def copies_needed (true_eval, vec, eps):   # simply determines the first time the progressive average 
    start_ind = 0

    for i in range(len(vec)):
        if (np.abs(vec[i]-true_eval)>eps): 
            start_ind = i
            break

    for j in range(start_ind, len(vec)):
        if (np.abs(vec[j]-true_eval)<eps): 
            return j
        
    return j


# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ## Plot functions

# + [markdown] tags=[]
# ### Progressive average comparison
# -

def progressive_average_comparison(collection,length_factor):
    
    collection.len_fact = length_factor
    qb_block = collection.qb_per_block
    rows = len(qb_block)
    cols = 1
    eps = 0.5
    ymax = 2

    exp_val = collection.get_true_eval()

    fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])#, sharex = True)

    if (collection.meas):
        fig.suptitle(f"{collection.meas} measurements")
        
    for n_qb in qb_block:
        index = collection.corresponding_index(n_qb)
        eval_block = collection.get_final_eval(n_qb)
        prefactor = collection.prefactor(n_qb)
        eps_len = collection.len_fact*prefactor/eps**2

        avg, var = collection.progressive_traj(n_qb, eps, True)
        for actual_vec in avg:
            axs[index].plot(np.arange(len(actual_vec)),actual_vec, linewidth = 0.3)

        axs[index].hlines(np.sign(eval_block)*exp_val + eps, 0, eps_len, linewidth = 2, linestyle=':')
        axs[index].hlines(np.sign(eval_block)*exp_val, 0, eps_len, linewidth = 2, label = 'expected value')
        axs[index].hlines(np.sign(eval_block)*exp_val - eps, 0, eps_len, linewidth = 2, linestyle=':' )
        axs[index].hlines(eval_block, 0, eps_len, linewidth = 2, linestyle = '--', label = 'final average')

        axs[index].vlines(eps_len, -ymax, ymax, linestyle = '--')

        axs[index].set_ylim(-ymax,ymax)
        #axs[index].set_xscale('log')
        axs[index].legend(loc='lower right')
        axs[index].set_title(f"Measurements on {n_qb} qubits at the same time")


# + [markdown] tags=[]
# ### Progressive *variance* comparison
# -

def progressive_variance_comparison(collection,length_factor):
    
    collection.len_fact = length_factor
    qb_block = collection.qb_per_block
    rows = len(qb_block)
    cols = 1
    eps = 0.5
    epss = [0.5,0.1,0.05,0.01]
    
    exp_val = collection.get_true_eval()

    fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])#, sharex = True)

    if (collection.meas):
        fig.suptitle(f"{collection.meas} measurements")
        
    for n_qb in qb_block:
        index = collection.corresponding_index(n_qb)
        eval_block = collection.get_final_eval(n_qb)
        prefactor = collection.prefactor(n_qb)
        eps_len = collection.len_fact*prefactor/eps**2

        avg, var = collection.progressive_traj(n_qb, eps, True)
        #for actual_vec in var:
        #    axs[index].plot(np.arange(len(actual_vec)),actual_vec/prefactor, linewidth = 0.3)

        for actual_vec in avg:
            axs[index].plot(np.arange(len(actual_vec)),np.abs(actual_vec-exp_val), linewidth = 0.3, linestyle='-.')

        ymax = 3#3*exp_var
        for e in epss:
            axs[index].hlines(e, 0, eps_len, linewidth = 2, linestyle=':', label=f'$\epsilon={e}$')
        #axs[index].hlines(prefactor, 0, eps_len, linewidth = 2, linestyle='--', label='expected variance',color='r')
        axs[index].vlines(eps_len, 0, ymax, linestyle = '--')

        #axs[index].set_ylim(0,ymax)
        axs[index].set_yscale('log')
        axs[index].legend(loc='lower right')
        axs[index].set_title(f"Measurements on {n_qb} qubits at the same time")


# + [markdown] tags=[]
# ### Measurement comparison
# -

def measurement_comparison(comp_collection, bell_collection, n_qb, length_factor):
    
    comp_collection.len_fact = length_factor
    bell_collection.len_fact = length_factor

    rows = 2
    cols = 1
    eps = 0.3
    ymax = 2
    c=['xkcd:navy blue', 'xkcd:scarlet']
    
    exp_val = comp_collection.get_true_eval() # should be equivalent…
    eps_len = comp_collection.expected_scale(n_qb,eps)
    pref = comp_collection.prefactor(n_qb)
    
    fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows], sharex = True)
    fig.suptitle(f"{n_qb} qubits per block, measurement comparison")
    
    index = comp_collection.corresponding_index(n_qb)

    # computational basis
    axs[0].set_title("Computational basis")
    eval_c = comp_collection.final_avg(n_qb)
    
    
    avg_c, var_c = comp_collection.progressive_traj(n_qb, eps, False)
    line_length = np.max([len(avg_c[0]), eps_len])
    
    for vec_c in avg_c:
        axs[0].plot(np.arange(len(vec_c)),vec_c, linewidth = 0.3,c='xkcd:cerulean')
    
    axs[0].hlines(eval_c, 0, line_length, linewidth = 2, linestyle = '--', label = 'final average',color='xkcd:navy blue')
    
    # Bell basis
    axs[1].set_title("Bell basis")
    eval_b = bell_collection.final_avg(n_qb)
    axs[1].hlines(eval_b, 0, line_length, linewidth = 2, linestyle = '--', label = 'final average', color='xkcd:scarlet')
    
    avg_b, var_b = bell_collection.progressive_traj(n_qb, eps, False)
    for vec_b in avg_b:
        axs[1].plot(np.arange(len(vec_b)),vec_b, linewidth = 0.3,c='xkcd:bright red')
    
    for i in (0,1):
        axs[i].hlines(np.sign(eval_c)*exp_val + eps, 0, line_length, linewidth = 2, linestyle=':',color='k')
        axs[i].hlines(np.sign(eval_c)*exp_val, 0, line_length, linewidth = 2, label = 'expected value',color='k')
        axs[i].hlines(np.sign(eval_c)*exp_val - eps, 0, line_length, linewidth = 2, linestyle=':' ,color='k')
        axs[i].set_ylim(-ymax,ymax)
        #axs[index].set_xscale('log')
        
    facts = np.arange(length_factor+1)
    for i in (0,1):
        for f in facts:
            eps_len = f*pref/eps**2
            axs[i].vlines(eps_len, -ymax, ymax, linestyle = '--',color=c[i])

    axs[1].legend(loc='lower right')


# + [markdown] tags=[]
# ### *Cheating* progressive average
# Longer progressive average on all data
# -

def cheating_progressive_average(collection, lenght_fact):
    collection.len_fact = lenght_fact
    qb_block = collection.qb_per_block
    rows = len(qb_block)
    cols = 1
    epsilons = [1,0.5,0.3]#,0.1,0.05]
    colors = ['xkcd:azure', 'xkcd:bright orange', 'xkcd:kelly green', 'xkcd:red','xkcd:violet']

    ymax = 2
    n_traj = 30
    exp_val = collection.get_true_eval()

    fig, axs = plt.subplots(rows, cols, figsize=[10*cols,4*rows])#, sharex = True)
    if (collection.meas):
        fig.suptitle(f"{collection.meas} measurements")

    
    for n_qb in qb_block:
        index = collection.corresponding_index(n_qb)
        eval_block = collection.get_final_eval(n_qb)
        prefactor = collection.prefactor(n_qb)

        for j in range(n_traj):
            avg, var = collection.full_progressive_traj(n_qb)
            axs[index].plot(np.arange(len(avg)), avg, linewidth=0.3)

        axs[index].hlines(np.sign(eval_block)*exp_val, 0, len(avg), linewidth = 2, label = 'expected value', color='k')
        axs[index].hlines(eval_block, 0, len(avg), linewidth = 2, linestyle = '--', label = 'final average', color='k')

        for i in range(len(epsilons)):
            eps = epsilons[i]
            eps_len = prefactor/eps**2
            axs[index].hlines(np.sign(eval_block)*exp_val + eps, 0, len(avg), linewidth = 2, linestyle=':',color=colors[i])
            axs[index].hlines(np.sign(eval_block)*exp_val - eps, 0, len(avg), linewidth = 2, linestyle=':',color=colors[i] )
            axs[index].vlines(eps_len, -ymax, ymax, linestyle = '--', label = f"$\epsilon={eps}$",color=colors[i])

        axs[index].set_ylim(-ymax,ymax)
        #axs[index].set_xscale('log')
        axs[index].legend(loc='lower right')
        axs[index].set_title(f"Measurements on {n_qb} qubits at the same time")


# ## Index determination

# + [markdown] tags=[] jp-MarkdownHeadingCollapsed=true
# ### Plot comparison for trajectories

# + tags=[]
def plot_comparison_trajectories(qb_coll, n_qb_block, avg_range, epsilons,traj):
    
    fig = plt.figure(figsize=[10,5])

    colors = ['xkcd:azure', 'xkcd:bright orange', 'xkcd:kelly green', 'xkcd:red','xkcd:violet']
    max_length = 0
    y_max = 10
    for collection in qb_coll:

        prefactor = collection.prefactor(n_qb_block)
        eval_coll = collection.get_true_eval()

        for j in range(traj):
            avg, var = collection.smooth_progressive_traj(n_qb_block, avg_range)
            length= len(avg)
            plt.plot(np.arange(length), np.abs(avg-eval_coll), linewidth=0.3)
            if (length>max_length):
                max_length = length

    for i in range(len(epsilons)):
        eps = epsilons[i]
        eps_len = prefactor/eps**2
        plt.hlines(eps, 0, max_length, linewidth = 2, linestyle=':',color=colors[i])
        plt.vlines(eps_len, 0, y_max, linestyle = '--', label = f"$\epsilon={eps}$",color=colors[i])

    plt.xlim([0,eps_len])
    plt.ylim([0.01,y_max])
    plt.yscale('log')
    #plt.xscale('log')
    plt.show()

    return


# + [markdown] tags=[]
# ### Copies required
# Also saves the plot! Amazing!

# + tags=[]
def copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, traj):
    
    colors = ['b','g','grey','y','m','c','w','orange','purple']
    markers = ['o','s','v','x']
    tot_avg = np.zeros((len(qb_block),len(list_eps)))
    actual_size = np.zeros((len(qb_block),len(list_eps)))
    
    global_copies = []
    
    fig,axs = plt.subplots(len(qb_block),1,figsize = [10,5*len(qb_block)],sharex=True)
    fig.suptitle(f"{qb_tot} qubits, {meas} measurement")
    colls = []
        
    for header in coll_headers:
        for obs in range(0,no_obs+1):
            try:
                collection = cs_collection(header, no_obs, qb_tot, qb_block, meas)
            except:
                collection = cs_collection(header, no_obs, qb_tot, qb_block)
    
            if collection.do_I_exist():
                colls.append(collection)
                
    for collection in colls:
        c_ind = coll_headers.index(collection.header)
        if (header=="6Baresi") :
            file_true_eval = '/Users/andrea/vienna_data/clifford_data/6Baresi/obs0/6Baresi_obs0_8_qubits_TrueEval.npy'
            baresi_true_eval = float(np.load(file_true_eval))
            collection.reassign_eval(baresi_true_eval)

        eval_coll = collection.get_true_eval()
        prefs = []
        for qb in collection.qb_per_block:

            prefactor = collection.prefactor(qb)
            prefs.append(prefactor)
            qb_ind = qb_block.index(qb)

            all_ind = []
            
            for k in range(traj):
                avg,  ind_collection, = collection.copies_needed_eps(qb, list_eps) 
                all_ind.append(ind_collection)
            
                global_copies.append([qb_tot, qb]+ind_collection)

            all_ind = np.array(all_ind)
            slit = len(all_ind)
            avg_ind = [np.sum(all_ind[0:slit,j])/slit for j in range(len(list_eps))]

            axs[qb_ind].scatter(avg_ind, list_eps, c = colors[c_ind])

            for i in range(len(list_eps)):
                tot_avg[qb_ind,i] += avg_ind[i]
                actual_size[qb_ind,i]+=1

    tot_avg = np.divide(tot_avg,actual_size)

    pref_scale = [[((2**qb+1)**(int(qb_tot/qb)))/e**2 for e in list_eps] for qb in qb_block]
    for i in range(len(qb_block)):
        axs[i].scatter(tot_avg[i],list_eps,c='red', marker = 'x', s=70, label = f'Average')  
        axs[i].scatter(pref_scale[i],list_eps ,c='k', marker = '*', s=70, label = "expected scaling")  

    [axs[j].set_title(f"Measurement over {qb_block[j]} qubits") for j in range(len(qb_block))]
    [axs[j].legend() for j in range(len(qb_block))]
    [axs[j].grid(True) for j in range(len(qb_block))]
    [axs[j].set_xscale('log') for j in range(len(qb_block))]
    
    plt.savefig(f'figures/copies_req_{qb_tot}_{meas}_qubits.png',format='png',bbox_inches = 'tight')
    np.save(f'{fixed_dir}0files/{qb_tot}_qubits_{meas}_copies.npy', global_copies)
    np.save(f'{fixed_dir}0files/average_{qb_tot}_qubits_{meas}_copies.npy', tot_avg)
    
    return global_copies, tot_avg


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
    def __init__ (self, head, obs_ind, tot_qubits, qb_per_block, measure=None):
        # assigns characteristic variables
        self.header = head
        
        self.n_qubits = tot_qubits
        self.qb_per_block = qb_per_block.copy()
        self.obs_ind = obs_ind
        self.meas = measure
        
        self.foobar = "*" # not defined a priori, obsolete parameter which has come back to haunt me
        self.len_fact = 1.5 # used to define how long a trajectory is
        
        
        # identifies path to relevant files
        self.__exists = False                 # assigned beforehand to avoid stalemate
        self.__exists = self.__define_paths() ## reassigns existence if it finds the relevant files
        if not self.__exists:
            return None
        # finds actual expectation value
        self.__true_eval = self.__find_eval()
        
        # loads data in array
        self.__eval   = []    
        self.__eval2  = []
        
        self.__load_all()
        self.__avg_eval = np.zeros(len(self.qb_per_block))
        for n_qb in self.qb_per_block:
            i = self.corresponding_index(n_qb)
            self.__avg_eval[i] = self.get_final_eval(n_qb)
            
        return
    
    # ---------------------------------------- #
    # reference to state and expectation value
    # ---------------------------------------- #
    
    def do_I_exist(self):        # I get it, it's kind of an hack…
        return self.__exists
    
    def __define_paths (self):   # identifies uniquely state, observable and particular file path
        fixed_dir = '/Users/andrea/vienna_data/clifford_data/' 
        main_dir = fixed_dir+self.header+'/'
        obs_dir  = main_dir + f"obs{self.obs_ind}/" 
        
        self.state_path = main_dir + f"{self.header}_state_{self.n_qubits}_qubits.npy"
        self.obs_path = obs_dir + f"{self.header}_obs{self.obs_ind}_{self.n_qubits}_qubits.npy"    

        if(glob.glob(self.obs_path)):
            return True
        
        return False
    
    def __find_eval(self):
        rho = np.array(np.load(self.state_path))
        if (len(rho[0]) == 1):
            rho = np.outer(rho,rho.conjugate())
        rho = qt.Qobj(rho)
            
        load_array = np.array(np.load(self.obs_path))
        if (len(load_array)==self.n_qubits):
            # consideres possibility of having saved only array of Paulis instead of full observable
            obs = qt.Qobj(qt.tensor([qt.Qobj(el) for el in load_array]).full())
        else:
            obs = qt.Qobj(load_array)
        
        return np.real((rho*obs).tr())
    
    def get_true_eval(self):
        return self.__true_eval
    
    def reassign_eval(self, val): ## failsafe for potential problems with overwriting files
        if (self.header == "6Baresi"):
            self.__true_eval = val
        return
    
    def get_final_eval(self, n_qb):
        i_block = self.corresponding_index(n_qb)
        return self.__avg_eval[i_block]
    
    # ----------------------- #
    # load data in collection
    # ----------------------- #
    
    def __load_all(self):              # in order to guarantee that order of evals is the same as qb_block
        #print(f"Commencing automatic data load for state {self.header}")
        for qb in self.qb_per_block:
            self.load_data_per_qb(qb)
            #print(f"Correctly loaded data for {qb} qubits per block")
        return
        
    def load_data_per_qb(self, n_qb): # for now, it just finds all relevant data with the same header, n_qb and n_block
        
        params = [self.obs_ind, self.n_qubits, n_qb, self.foobar]
        if (self.meas):
            params.append(self.meas)
            
        main, obs, dir_name = define_dir_name (self.header, params)
        
        file_name = define_file_name(params, self.foobar)
        relevant_files = glob.glob(dir_name+file_name+'.npy')

        if (relevant_files):
            for file in relevant_files:
                foo_vec = np.real(np.load(file,allow_pickle=True))
                self.__increase_eval(n_qb, foo_vec[i_exp],foo_vec[i_exp2])
        else:
            self.qb_per_block.remove(n_qb)
            
        return
        
    def __increase_eval(self, n_qb, vec_eval, vec_var):
        i_block=self.corresponding_index(n_qb)         
        if (len(self.__eval) == i_block):
            self.__eval.append(vec_eval)
            self.__eval2.append(vec_var)
        else:
            self.__eval[i_block]  = np.concatenate([self.__eval[i_block],vec_eval])
            self.__eval2[i_block] = np.concatenate([self.__eval2[i_block],vec_var])
        return
     
    # -------------------------------------------- #
    # functions for returning characteristic values
    # -------------------------------------------- # 
    
    def final_avg (self, n_qb): # returns the corresponding average, given localilty of measurement
        i_block = self.corresponding_index(n_qb)
        return np.sum(self.__eval[i_block])/len(self.__eval[i_block])
        
    def corresponding_index(self, n_qb):
        return self.qb_per_block.index(n_qb)
    
    def prefactor (self,n_block):
        return (2**n_block+1)**(self.n_qubits/n_block)
    
    def expected_scale (self,n_block,eps):
        return self.len_fact*self.prefactor(n_block)/eps**2

    # -------------------- #
    # progressive averages
    # -------------------- #
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
        
        split_data = self.__qb_divide(np.array(self.__eval[i_block]), n_qb, eps)
        split_var  = self.__qb_divide(np.array(self.__eval2[i_block]), n_qb, eps)
        
        return [prog_avg(chunk) for chunk in split_data], [prog_var(data, var) for data, var in zip(split_data,split_var)]
    
    def full_progressive_traj(self, n_qb):
        i_block = self.corresponding_index(n_qb)
        self.__shuffle_blocks(i_block)
        
        return prog_avg(self.__eval[i_block]), prog_var(self.__eval[i_block],self.__eval2[i_block])
    
    def smooth_progressive_traj(self,n_qb, avg_range):
        avg, var = self.full_progressive_traj(n_qb)
        
        return rolling_avg(avg, avg_range), rolling_avg(var, avg_range)
    
    def copies_needed_eps(self, n_qb, list_eps):
        avg, var = self.full_progressive_traj(n_qb)
        pref = int(self.prefactor(n_qb))
        
        return avg,  [pref+copies_needed(self.__true_eval, avg[pref:], e) for e in list_eps]
            

# ## Bennacer data
# Before different measurements were implemented, so loaded separately

# +
qb_tot = 8
qb_block = [1,2,4,8]
no_obs = 0

coll_headers = "4Bennacer"
isma  = cs_collection(coll_headers, no_obs, qb_tot, qb_block)

# + [markdown] tags=[]
# ## Adli data
# For now, the only one with 4 qubits total

# +
header = "7Adli"
qb_tot = 4
qb_block = [1,2,4]
no_obs = 0

adli_comp = cs_collection(header, no_obs, qb_tot, qb_block,'comp')
#adli_bell = cs_collection(header, no_obs, qb_tot, qb_block,'bell')
#adli_sic = cs_collection(header, no_obs, qb_tot, qb_block,'bell')


# -

adli_comp.do_I_exist() ##this is kind of a hack…

# ## Comparison of different observables
# Instead of trying to divide the data sets into smaller samples, let's do it the old-fashioned way: just direct comparison of different trajectories, seeing when each surpasses a certain line. Then, we use **this** result to divide the data set, in order to increase the statistics a little

# indices
i_qb_tot = 0
i_qb_block = 1
i_eps = 2

list_eps = [0.5,0.3,0.2,0.1,0.05]
n_traj = 30
meas = 'comp'

# ### 8 qubits

# +
qb_tot = 8
qb_block = [1,2,4,8]
no_obs = 0

coll_headers = ["6Baresi","8Tonali","9Giroud","10Diaz","11Ibra","12Rebic"]#"4Bennacer"]#
#

# +
#global_indices_8, tot_avg_8 = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)
# -

# ### 6 qubits

# +
qb_tot = 6
qb_block = [1,2,3,6]
no_obs = 1

coll_headers = ["16Maignan","17Leao","18Montolivo","19Hernandez"]

# -

global_indices_6, tot_avg_6 = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)

# ### 4 qubits

# +
qb_tot = 4
qb_block = [1,2,4]
no_obs = 3

coll_headers = ["7Adli", "14Baka","15Hauge"]

# -

global_indices_4, tot_avg_4 = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)

# ## Bell Measurements

meas = 'bell'

# ### 8 qubits

# +
qb_tot = 8
qb_block = [1,2,4,8]
no_obs = 0

coll_headers = ["6Baresi","8Tonali","9Giroud","10Diaz","11Ibra","12Rebic"]
#"4Bennacer"
# -

global_indices_8b, tot_avg_8b = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)

# ### 6 qubits

# +
qb_tot = 6
qb_block = [1,2,3,6]
no_obs = 1

coll_headers = ["16Maignan","17Leao","18Montolivo","19Hernandez"]

# -

global_indices_6b, tot_avg_6b = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)

# ### 4 qubits

# +
qb_tot = 4
qb_block = [1,2,4]
no_obs = 3

coll_headers = ["7Adli","14Baka","15Hauge"]

# -

global_indices_4b, tot_avg_4b = copies_required(coll_headers, qb_tot, qb_block, list_eps, no_obs, meas, n_traj)

# ## Averages comparisons

# +
dir_files = fixed_dir+'0files/'
meas_array = ['comp','bell','sic']

qb_tot = [4,6,8]

file_names = [f'{qb}_qubits_{meas}_copies.npy' for qb in qb_tot for meas in meas_array]
# -

# ### 8 qubits
# #### Plot: measurement over different qubits

# +
qb_tot = 8
qb_block = [1,2,4,8]

markers = ['o','s','v','x']
    
fig,axs = plt.subplots(len(qb_block),1,figsize = [10,5*len(qb_block)],sharex=True)

prefactor = [((2**qb+1)**(int(qb_tot/qb))) for qb in qb_block]
pref_scale_var = [[((2**qb+1)**(int(qb_tot/qb)))/e  for e in list_eps ]for qb in qb_block]
pref_scale = [[((2**qb+1)**(int(qb_tot/qb)))/e**2  for e in list_eps ]for qb in qb_block]

for i in range(len(qb_block)):
    axs[i].scatter(tot_avg_8[i],list_eps,c='red', marker = 'x', s=70, label = f'Classical shadow')  
    axs[i].scatter(tot_avg_8b[i],list_eps,c='blue', marker = 'o', s=70, label = f'Bell measurements')
    axs[i].scatter(pref_scale[i],list_eps ,c='k', marker = '*', s=70, label = "expected scaling")  
    axs[i].scatter(pref_scale_var[i],list_eps ,c='k', marker = 'v', s=70, label = "alternative scaling")  
        
        
[axs[j].set_title(f"Measurement over {qb_block[j]} qubits") for j in range(len(qb_block))]
[axs[j].legend() for j in range(len(qb_block))]
[axs[j].grid(True) for j in range(len(qb_block))]
[axs[j].set_xscale('log') for j in range(len(qb_block))]
    
# -

# #### Plot: measurement for different $\varepsilon$

# +
qb_tot = 8
qb_block = [1,2,4,8]

markers = ['o','s','v','x']
    
fig,axs = plt.subplots(len(list_eps),1,figsize = [7,5*len(qb_block)],sharex=True)

prefactor = np.array([((2**qb+1)**(int(qb_tot/qb))) for qb in qb_block])

for i in range(len(list_eps)):
    e = list_eps[i]
    axs[i].scatter(qb_block,tot_avg_8[:,i],c='red', marker = 'x', s=70, label = f'Classical shadow')  
    axs[i].scatter(qb_block,tot_avg_8b[:,i],c='blue', marker = 'o', s=70, label = f'Bell measurements')
    
    axs[i].plot(np.arange(1,qb_tot+1), np.array([((2**qb+1)**((qb_tot/qb))) for qb in np.arange(1,qb_tot+1)])/(e**2), c='k', marker='*', label='Expected scaling')
    axs[i].plot(np.arange(1,qb_tot+1), np.array([((2**qb+1)**((qb_tot/qb))) for qb in np.arange(1,qb_tot+1)])/(e), c='gray', marker='v', label='variation scaling')
        
[axs[i].set_title(f"$\epsilon$= {list_eps[j]}") for i in range(len(list_eps))]
[axs[i].legend() for i in range(len(list_eps))]
[axs[i].grid(True) for i in range(len(list_eps)) ]
[axs[i].set_yscale('log') for i in range(len(list_eps))] 
    
# -

# #### Ratio

qb_tot = 8
block_8 = [1,2,4,8]
pref_8  = [((2**qb+1)**(int(qb_tot/qb))) for qb in block_8]
ratio_8 = [tot_avg_8[i]/pref_8[i] for i in range(len(block_8))]
ratio_8b = [tot_avg_8b[i]/pref_8[i] for i in range(len(block_8))]

qb_tot = 6
block_6 = [1,2,3,6]
pref_6  = [((2**qb+1)**(int(qb_tot/qb))) for qb in block_6]
ratio_6 = [tot_avg_6[i]/pref_6[i] for i in range(len(block_6))]
ratio_6b = [tot_avg_6b[i]/pref_6[i] for i in range(len(block_6))]

qb_tot = 4
block_4 = [1,2,4]
pref_4  = [((2**qb+1)**(int(qb_tot/qb))) for qb in block_4]
ratio_4 = [tot_avg_4[i]/pref_4[i] for i in range(len(block_4))]
ratio_4b = [tot_avg_4b[i]/pref_4[i] for i in range(len(block_4))]

# +
qb_tot = 4
tot_block = [1,2,3,4,6,8]
list_eps = np.array(list_eps)

fig, axs = plt.subplots(len(tot_block),2,figsize=[2*10,7*len(tot_block)])
fig.suptitle (r"Ratio between number of copies and expected scaling, depending on $\varepsilon$", size=30)
for i in range(len(block_8)):
    qb = block_8[i]
    ind = tot_block.index(qb)
    axs[ind,0].scatter(list_eps,ratio_8[i],c='red', marker = 'x', s=70, label = f'8 qubits')
    axs[ind,1].scatter(list_eps,np.divide(ratio_8[i], list_eps**(-1)), c='red', marker = 'o', s=70,label=r'Division by $\varepsilon^{-1}$')
    axs[ind,1].scatter(list_eps,np.divide(ratio_8[i], list_eps**(-2)), c='red', marker = '*', s=70,label=r'Division by $\varepsilon^{-2}$')
    #axs[ind].scatter(list_eps,pref_8[i],c='red', marker = '*', s=70)
    
for i in range(len(block_6)):
    qb = block_6[i]
    ind = tot_block.index(qb)
    axs[ind,0].scatter(list_eps,ratio_6[i],c='blue', marker = 'x', s=70, label = f'6 qubits')  
    axs[ind,1].scatter(list_eps,np.divide(ratio_6[i], list_eps**(-1)), c='blue', marker = 'o', s=70)
    axs[ind,1].scatter(list_eps,np.divide(ratio_6[i], list_eps**(-2)), c='blue', marker = '*', s=70)

    #axs[ind].scatter(list_eps,pref_6[i],c='blue', marker = '*', s=70)

for i in range(len(block_4)):
    qb = block_4[i]
    ind = tot_block.index(qb)
    axs[ind,0].scatter(list_eps,ratio_4[i],c='green', marker = 'x', s=70, label = f'4 qubits')  
    axs[ind,1].scatter(list_eps,np.divide(ratio_4[i], list_eps**(-1)), c='green', marker = 'o', s=70)
    axs[ind,1].scatter(list_eps,np.divide(ratio_4[i], list_eps**(-2)), c='green', marker = '*', s=70)


fact1 = 0.8
fact2 = 0.05

for j in range(len(tot_block)):
    axs[j,0].plot(list_eps,fact1*list_eps**(-1),c='grey',label=r'$\varepsilon^{-1}$') 
    axs[j,0].plot(list_eps,fact2*list_eps**(-2),c='k',label=r'$\varepsilon^{-2}$')
    axs[j,0].set_ylabel(r'$N / (2^l+1)^{n/l}$',size=20) 
    
    axs[j,1].set_ylabel(r'$N / (2^l+1)^{n/l} / \varepsilon$',size=20)
    axs[j,1].hlines(1,list_eps[0],list_eps[-1],label='expected ratio',color='k', linestyle='--')

    for i in range(2):
        axs[j,i].legend(fontsize=15) 
        axs[j,i].set_xlabel(r'$\varepsilon$',size=20) 
        axs[j,i].set_title(f"Measurement over {tot_block[j]} qubits",size=25) 
        axs[j,i].grid(True)
        axs[j,i].set_xscale('log') 

# -


