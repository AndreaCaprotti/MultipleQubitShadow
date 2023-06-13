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

# # Multiple-qubit classical snapshots
# The present code is a variation of the selective quantum state tomography based on the work [https://arxiv.org/abs/1909.05880]. It samples a complete SIC-POVM (might also add possibility to sample MUBs at some point…) in order to reconstruct the mean value of an observable.
# This preliminary version considers the estimation only for traceless Paulis (no qubit reordering needed), but needs to be expanded for any locality.

# +
import numpy as np
import scipy as sc
import qutip as qt # handles all the quantum operations
import qbism as qb
import qfunk.generator as gg # used to randomly sample states 
import stim 

import itertools

import random # local random generator (i.e. for random observable)
import time   # for time comparison and optimization
import sys
import os


# + [markdown] tags=[]
# ## Random observable sampling
# For now, we consider tensor products of non-trivial Paulis. This shold not be the general case (at some point a kind of "qubit reordering" would be needed) but for now it's good enough
# -

# ### Measurement POVM
# We consider different possibilities of POVM on which one might want to measure its state. Mainly to compare with results of [https://arxiv.org/abs/2305.10723]

# + [markdown] tags=[]
# #### SIC-POVM
# Actually substitutes the unitary scrambling instead of Clifford measurements. Since SIC-POVM form a 2-design, we expect not to have the same optimal scaling also for variance. For completeness, we keep the function to generate a full SIC-POVM; however, we mainly use the sampler for random POVM element chosen based on the Haar measure.

# +
def sic_povm(no_qubits):
    return qb.sic_povm(2**no_qubits)

def random_sic (dim_block):
    return qb.random_haar_povm(dim_block, k=1, n=1, real=False).full()


# -

# ### Single-qubit Paulis
# More or less the basic building blocks we'll always need

# + tags=[]
qubit_d = 2
Id2 = qt.qeye(qubit_d)
Sx  = qt.sigmax()
Sy  = qt.sigmay()
Sz  = qt.sigmaz()

pauli_array = [Id2,Sx,Sy,Sz]
spin_array = [Sx,Sy,Sz]


# -

# ### Random observable sampler
# Compared to the first version, it is actually better to store the *array* of Paulis and only afterwards to turn it into a tensor product. This is simpler when considering the POVM decomposition of the observable.
# Moreover, since we're always using the data array more than the `qt.Qobj()`, it is actually equivalent to take the tensor product afterwards. 
# In order to possibly handle also previous observables, a very basic `tensor_decomposition` mode is considered, to return a way to handle already "tensored" Paulis for POVM decomposition.

def tensor_decomposition(observable, obs_array, no_qubits):
    perms = list(itertools.product(obs_array, repeat=no_qubits))
    
    for element in perms:
        if (np.allclose(observable, qt.tensor(list(element)).full())):
            return list(element) # returns list of *qt.Qobj*
        
    return "ERROR" # check if loop never closes


def rand_pauli_list (obs_array, no_qubits): # returns *list* of arrays instead of tensor product
    return [obs_array[ind] for ind in np.random.randint(0,len(obs_array)-1,size=no_qubits)]


# ## Selective POVM sampling
# In order to randomly sample an observable expectation value using a POVM, we first need to know the decomposition of the observable in said POVM. 
# Then I can think of two different ways of handling the data:
# - Randomly sample SIC-POVM using Haar measure and afterwards reconstruct state based on this decomposition
# - Sample SIC-POVM following the said distribution and then just average over the results 
#     - this second one is a little bit cheating, but it's also easier to implement

# #### Distribution sampling

# +
def prob_vec_to_sample (prob_vec):
    length = len(prob_vec)
    vec = np.zeros(length)
    vec[0] = prob_vec[0]
    for i in range(1,length):
        vec[i] = vec[i-1] + prob_vec[i]
    return vec

def sample_from_vec(prob_vec_cumulative):
    rand_val = random.random()
    
    for j in range(len(prob_vec_cumulative)):
        if (rand_val < prob_vec_cumulative[j]):
            return j
        
    if (j == len(prob_vec_cumulative)):
        return j


# -

# ### Optimized sparse multiplication

# #### Tensor product and sparse unitary generator

def tensor_sparse(*args):
    if not args:
        raise TypeError("Requires at least one input argument")

    if len(args) == 1 and isinstance(args[0], (list, np.ndarray)):
        # this is the case when tensor is called on the form:
        # tensor([q1, q2, q3, ...])
        qlist = args[0]

    elif len(args) == 1 and isinstance(args[0], Qobj):
        # tensor is called with a single Qobj as an argument, do nothing
        return args[0]

    else:
        # this is the case when tensor is called on the form:
        # tensor(q1, q2, q3, ...)
        qlist = args
    
    out = [1]
    for n, q in enumerate(qlist):
        if n == 0:
            out = q
            
        else:
            out  = sc.sparse.kron(out, q)
    return out


def sparse_whole_matrix (element,  qubits_per_block, no_block, position):

    dim_before = 2**(qubits_per_block*position)
    dim_after  = 2**(qubits_per_block*(no_block - position-1))
    
    return tensor_sparse(sc.sparse.identity(dim_before),element,sc.sparse.identity(dim_after))


# ## File save
# Includes possible list of columns to be saved and automatic choice of file name.
# Not using `.txt` format since I don't actually need to read the data, so better to just shove it into some `.npy` file; all relevant data already indicated in file name

# for now, I can just think of these…
param_title = ["_obs",
               "_ntot",
               "_nblock",
               "_eps",
               "_meas",
              ]


def define_file_name (params, index): # thought to be, eventually, extended for more parameters
    starting_block = f"{index}"       # and on this, it doesn't rain (italian expression)
    for i in range(len(params)):
        starting_block+=(param_title[i]+f"_{params[i]}")
    
    return starting_block


# ## Simulation parameters
# Adding the possibility to add parameters from command line, to better control the parameters in question

# + [markdown] tags=[]
# ### Parameters from command line

# +
if (len(sys.argv) == 10):
    dir_header = sys.argv[1]
    n_qb_tot = int(sys.argv[2])
    n_qb_block = int(sys.argv[3])
    n_runs = int(sys.argv[4])
    init_run = int(sys.argv[5])
    n_traj = int(sys.argv[6])
    load_state = int(sys.argv[7])
    load_obs = int(sys.argv[8])
    obs_ind = int(sys.argv[9])
    
else:
    dir_header = "MultipleQubit"
    n_qb_tot = 4
    n_qb_block = 2
    n_runs = 1 # test 
    init_run = 0 # fail-safe in case of interruption
    n_traj = 10 #int(2* prefactor(1, n_qb_tot)/epsilon**2)
    load_state = False
    load_obs = False
    obs_ind = 1
# -

# ### Default parameters

# +
qubit_dim = 2
dim_tot = qubit_dim**n_qb_tot
dim_block = qubit_dim**n_qb_block

if (n_qb_tot%n_qb_block == 0):
    no_of_blocks = int(n_qb_tot / n_qb_block)
else:
    raise NameError("Blocks are not all of the same size")

meas = "sic"
parameters = [obs_ind, n_qb_tot, n_qb_block, epsilon, meas]
# -

# Here we generate random states and observables in order to have no bias for the estimation. Should *definitely* be generalised, but for now good enough

# ### Target state and observable

# +
# identifies directory name
fixed_dir = '/Users/andrea/vienna_data/clifford_data/' 
dir_specific = define_file_name (parameters, f"{dir_header}_{obs_ind}")
main_dir = fixed_dir+dir_header+'/'
obs_dir  = main_dir + f"obs{obs_ind}/" 
dir_name = obs_dir+dir_specific+'/'

os.system(f'mkdir -p {main_dir}')
os.system(f'mkdir -p {obs_dir}')
os.system(f'mkdir -p {dir_name}') #should only create it once…

# -

# The following save the states and observables for future use, or load more if already present.
# NB: it's actually better in these cases to save everything in .npy format: not readable but much more convenient

# +
# header specifies state
state_file = dir_header+f"_state_{n_qb_tot}_qubits.npy"

if (load_state):
    try:
        rho = np.array(np.load(main_dir + state_file))
    except IOError: # if, for the first time, the state is not present
        rho = qt.rand_ket(dim_tot)  # using a pure state is equivalent but advantageous numerically
        rho = qt.ket2dm(rho)
        np.save(main_dir + state_file, rho.full())
        print("I've saved a new file!")
else:
    rho = qt.rand_ket(dim_tot)
    rho = qt.ket2dm(rho)
    np.save(main_dir + state_file, rho.full())
# -

obs_file = dir_header+f"_obs{obs_ind}_{n_qb_tot}_qubits.npy"
if (load_obs):
    try:
        obs = np.array(np.load(obs_dir + obs_file))
        if (len(obs) == 1):
            obs = tensor_decomposition(obs, spin_array, n_qb_tot) # turns full observable into list of Paulis
    except:
        obs = rand_pauli_list (spin_array, n_qb_tot) 
        np.save(obs_dir + obs_file, obs)
else:
    obs = rand_pauli_list (spin_array, n_qb_tot) 
    np.save(obs_dir + obs_file, obs)

# If we're considering sparse matrices, turns `qt.Qobj` into a sparse matrix. Since sometimes it starts from a `np.array`, in general it's not the smartest move. For now we'll manage…

# ### SIC - POVM

# #### Definition

# +
povm = qb.sic_povm(dim_block)

array_povm = [el.data.toarray() for el in povm]
coeff_mat = np.array([np.ndarray.flatten(array.data.toarray()) for array in povm]).transpose()
coeff_mat = build_coef_mat (povm).transpose()
# -

# #### Observable decomposition

# +
sampling_distribution = []

for j in range(no_of_blocks):
    subsystem = np.ndarray.flatten(qt.tensor([ obs[i] for i in np.arange((j)*n_qb_block,(j+1)*n_qb_block)]).data.toarray())

    vec = np.real(sc.linalg.solve(coeff_mat, subsystem))
    vec = np.abs(vec) /np.sum(np.abs(vec))
    sampling_distribution.append(prob_vec_to_sample(vec))
# -

# ## Classical shadow collection

for i in range(init_run, init_run+n_runs):
    save_array=[]
    
    for t in range(n_traj):        
        rot_rho = rho.data.toarray()
        ind = []
        for j in range(no_of_blocks):
            k = sample_from_vec(sampling_distribution[i])
            povm_el = array_povm[k]

            ind.append(k)
            povm_el = sparse_whole_matrix (povm_el, n_qb_block, no_of_blocks, j)
            rot_rho = povm_el@rot_rho
        exp_val = np.real((sc.sparse.csr_matrix(rot_rho)).trace())
        
        save_array.append([exp_val,exp_val**2]+ind)
   
    ## save finale vectors in separate file
    filename = define_file_name (parameters, i)
    np.save(dir_name+filename, np.array(save_array))


