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
# The present code is based on the classical shadow algorithm base on [https://arxiv.org/abs/2002.08953]. However, instead of considering random local measurements, the random sampling is made over a number *l* of qubits at the same time, in order to have a better scaling in number of copies to achieve small-enough variance.
# Uses 
# - `QuTiP` to handle all quantum objects and their operations (not the fastest, but already optimised)
# - `stim` to generate random Clifford unitaries of given size
#
# This preliminary version considers the estimation only for traceless Paulis (no qubit reordering needed), but needs to be expanded for any locality.

# +
import numpy as np
import scipy as sc
import qutip as qt # handles all the quantum operations
import qfunk.generator as gg # used to randomly sample states 
import stim 



import random # local random generator (i.e. for random observable)
import time   # for time comparison and optimization
import sys
import os


# -

# # TO DO:
# - Understanding where the problem of sign comes at
# - Check if reshaping of quantum objects lowers computational time
# - Generalise to local observables (which might need reordering of qubits)
# - Consider stabilizer formalism for faster simulation and lower computational value -> `QuTiP` actually meant to simulate dynamics, what we consider is more attinent to a *quantum circuit*

# + [markdown] tags=[]
# ## Random observable sampling
# For now, we consider tensor products of non-trivial Paulis. This shold not be the general case (at some point a kind of "qubit reordering" would be needed) but for now it's good enough
# -

# ### Computational basis
# Needs to be defined by hand, not optimal at all

def build_comp_basis (dim_tot):
    return [qt.basis(dim_tot,z) for z in range (dim_tot)]


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

def rand_pauli_sampler (obs_array, no_qubits, qubits_per_block):
    no_blocks = int(no_qubits/qubits_per_block)   # not so useful for now, let's just keep it aside
    no_obs = len(obs_array)
    tot_obs = []
    
    for i in range(no_qubits):
        ind = random.randint(0,no_obs-1)
        tot_obs.append(obs_array[ind])
    
    return qt.Qobj(qt.tensor(tot_obs).full())      # operation needed so that dimensions are always compatible  


# ## Random unitary sampler
# Uses `stim`circuit sampler to build unitary as tensor product of smaller Cliffords.
# `stim` actually samples a Tableau in the *stabilizer formalism*, which is then transformed into a unitary matrix.
# Notice how the final unitary itself *might not* be Clifford itself, but it is a tensor product of different Cliffords

def tot_clifford_sampler (no_qubits, qubits_per_block):
    no_blocks = int(no_qubits/qubits_per_block)
    
    return qt.Qobj(qt.tensor([qt.Qobj(stim.Tableau.random(qubits_per_block).to_unitary_matrix(endian='little')) for n in range(no_blocks)]).full())


# ## Classical snapshot expectation value 
# In general, one should build a classical snapshot, store it and then use it to estimate the expectation value of some observable afterwards (in order to also possibly reuse data for other observables). Since this requires a non-trivial ammount of storage space, for now we just use a faster version which already considers the observable to be evaluated.
#
# Now 

def prefactor (qubits_per_block,locality):
    return (2**qubits_per_block+1)**(locality/qubits_per_block)


# ### First *basic* example

# +
def mq_classical_snapshot (state, basis, no_qubits, qubits_per_block): # "actual" version
    pref = prefactor(qubits_per_block,no_qubits)
    
    unitary = tot_clifford_sampler(no_qubits, qubits_per_block)
    rot_rho = unitary*state*unitary.dag()
    meas, state = qt.measurement.measure(rot_rho, basis)
    
    return pref*(unitary.dag()*state*unitary)

def mq_cs_direct_estimation (state, observable, basis, no_qubits, qubits_per_block): # faster, direct version
    pref = prefactor(qubits_per_block,no_qubits)
    
    unitary = tot_clifford_sampler(no_qubits, qubits_per_block)
    rot_rho = unitary*state*unitary.dag()
    rot_rho = rot_rho/rot_rho.tr()     # normalization needed, since unitaries are not *exactly* unitary

    meas, cond_state = qt.measurement.measure(rot_rho, basis)
    
    return pref*(unitary*observable*unitary.dag()*cond_state).tr()


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


def sparse_l_matrix (qubits_per_block, dim_block, no_block, position):
    
    no_id_before = [sc.sparse.identity(dim_block)] * (position) # since numbering starts at 0
    no_id_after  = [sc.sparse.identity(dim_block)] * (no_block - position - 1)
    
    unitary = stim.Tableau.random(qubits_per_block).to_unitary_matrix(endian='little')
    return tensor_sparse(no_id_before+[unitary]+no_id_after)


def sparse_whole_matrix ( qubits_per_block,no_block, position):

    dim_before = 2**(qubits_per_block*position)
    dim_after  = 2**(qubits_per_block*(no_block - position-1))
    
    unitary = stim.Tableau.random(qubits_per_block).to_unitary_matrix(endian='little')
    return tensor_sparse(sc.sparse.identity(dim_before),unitary,sc.sparse.identity(dim_after))


# +
def id_sparse_list (no_qubits,qubits_per_block):
    no_blocks = int(no_qubits/qubits_per_block)
    dim_block = 2**qubits_per_block
    return [sc.sparse.identity(dim_block)]*no_blocks

def listsparse_l_matrix (no_qubits, qubits_per_block, position, which_list):
    unitary = stim.Tableau.random(qubits_per_block).to_unitary_matrix(endian='little')

    return tensor_sparse(which_list[:position] + [unitary] + which_list[position+1:])


# -

# #### Snapshot generator

# + tags=[]
# Handles unitary evolution on sc.sparse matrices

def mq_cs_sparse_estimation (state, observable, basis, no_qubits, qubits_per_block):  
    pref = prefactor(qubits_per_block,no_qubits)
    no_block = int(no_qubits/qubits_per_block)
    dim_block = 2**qubits_per_block

    rot_rho = state
    for j in range(no_block):
        unitary = sparse_whole_matrix (qubits_per_block, no_block, j)
        rot_rho = unitary@rot_rho@unitary.H
        observable = unitary@observable@unitary.H

    state = qt.Qobj(rot_rho)

    meas, cond_state = qt.measurement.measure(state/state.tr(), basis)
    
    return pref*(observable@sc.sparse.csr_matrix(cond_state)).trace()


# -

# ## Simulation function

def classical_shadow_simulation (rho, obs, qb_block, qb_tot, n_tries, sparse):
    avg_array = []
    var_array = []

    time_tot = time.time()

    for t in range(n_tries):
        if(sparse):
            exp_val = mq_cs_sparse_estimation (rho, obs, comp_basis, n_qb_tot, n_qb_block)
        else:
            exp_val = mq_cs_direct_estimation (rho, obs, comp_basis, n_qb_tot, n_qb_block)
        
        avg_array.append(exp_val)
        var_array.append(exp_val**2)

    print(f"{qb_block} qubits per block take: {time.time()-time_tot} seconds to complete")
    
    return avg_array, var_array


# ## File save
# Includes possible list of columns to be saved and automatic choice of file name.
# Not using `.txt` format since I don't actually need to read the data, so better to just shove it into some `.npy` file; all relevant data already indicated in file name

# for now, I can just think of these…
param_title = ["_obs",
               "_ntot",
               "_nblock",
               "_eps"
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
if (len(sys.argv) == 11):
    dir_header = sys.argv[1]
    n_qb_tot = int(sys.argv[2])
    n_qb_block = int(sys.argv[3])
    epsilon = sys.argv[4]
    n_runs = int(sys.argv[5])
    n_traj = int(sys.argv[6])
    load_state = bool(sys.argv[7])
    load_obs = bool(sys.argv[8])
    obs_ind = int(sys.argv[9])
    sparse  = bool(sys.argv[10])
    
else:
    dir_header = "MultipleQubit"
    n_qb_tot = 8
    n_qb_block = 2 
    epsilon = 0.5
    n_runs = 1 # test 
    #n_runs = 30 # statistical significance
    n_traj = 10 #int(2* prefactor(1, n_qb_tot)/epsilon**2)
    load_state = False
    load_obs = False
    obs_ind = 0
    sparse  = False
# -

# ### Default parameters

# +
qubit_dim = 2
dim_tot = qubit_dim**n_qb_tot
dim_block = qubit_dim**n_qb_block

if (dim_tot%dim_block == 0):
    no_of_blocks = dim_tot / dim_block
else:
    raise NameError("Blocks are not all of the same size")

expected_prefactor = prefactor(n_qb_block,n_qb_tot)

comp_basis = build_comp_basis(qubit_dim**n_qb_tot)

parameters = [obs_ind, n_qb_tot, n_qb_block, epsilon]
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
state_file = dir_header+f"_state_{n_qb_tot}_qubits"

if (load_state):
    try:
        rho = qt.Qobj(np.array(np.load(main_dir + state_file)))
    except IOError: # if, for the first time, the state is not present
        rho = qt.rand_ket(dim_tot)  # using a pure state is equivalent but advantageous numerically
        rho = qt.ket2dm(rho)
        np.save(main_dir + state_file, rho.full())
else:
    rho = qt.rand_ket(dim_tot)
    rho = qt.ket2dm(rho)
    np.save(main_dir + state_file, rho.full())
# -

obs_file = dir_header+f"_obs{obs_ind}_{n_qb_tot}_qubits"
if (load_obs):
    try:
        obs = qt.Qobj(np.array(np.load(obs_dir + obs_file)))
    except:
        obs = rand_pauli_sampler (spin_array, n_qb_tot, n_qb_block) 
        np.save(obs_dir + obs_file, obs.full())
else:
    obs = rand_pauli_sampler (spin_array, n_qb_tot, n_qb_block) 
    np.save(obs_dir + obs_file, obs.full())

# If we're considering sparse matrices, turns `qt.Qobj` into a sparse matrix. Since sometimes it starts from a `np.array`, in general it's not the smartest move. For now we'll manage…

true_expectation_value = (rho*obs).tr()
if (sparse):
    rho = sc.sparse.csr_matrix(rho.full())
    obs = sc.sparse.csr_matrix(obs.full())

# ## Classical shadow collection

for i in range(n_runs):
    avg, var = classical_shadow_simulation (rho, obs, n_qb_block, n_qb_tot, n_traj, sparse)
    filename = define_file_name (parameters, i)
    np.save(dir_name+filename, np.array([avg,var]))
    print('\n')


