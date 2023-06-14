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

# # Simulation Handler - SQOT
# Just some quicks lines of code to simplify my life in handling multiple simulations at the same time
#

import numpy as np
import time   # for time comparison and optimization
import sys
import os


def prefactor (qubits_per_block,locality):
    return (2**qubits_per_block+1)**(locality/qubits_per_block)


save = True

# +
header = ["7Adli"]

n_qubits = 4
n_blocks = [1,2,4]
epsilon = [1]
n_runs = 10
init_runs = 0 # also to add
no_diff_obs = 1

for head in header:
    for obs in range(0,no_diff_obs):
        for nb in n_blocks:
            for e in epsilon:
                n_traj = int(1.5*prefactor(nb, n_qubits)/(e**2))
                command_line = f' {head} {int(n_qubits)} {int(nb)} {e} {int(n_runs)} {int(init_runs)} {int(n_traj)} {obs}'
                if (save):
                    time_start = time.time()
                    os.system(f'python MQ_SQT.py {command_line}')
                    print (time.time()-time_start, command_line)
# -


