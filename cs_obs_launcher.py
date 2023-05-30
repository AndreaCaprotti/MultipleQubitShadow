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

# # Simulation Handler
# Just some quicks lines of code to simplify my life in handling multiple simulations at the same time

import numpy as np
import time   # for time comparison and optimization
import sys
import os


def prefactor (qubits_per_block,locality):
    return (2**qubits_per_block+1)**(locality/qubits_per_block)


# +
header = ["3Maldini"]

n_qubits = 4
n_blocks = [1,2]#4
epsilon = [0.5,0.1]#,0.05]
n_runs = 30
no_diff_obs = 1

load_state = True
load_obs = True

for head in header:
    for obs in range(0,no_diff_obs):
        for nb in n_blocks:
            for e in epsilon:
                n_traj = int(1.5*prefactor(1,n_qubits)/(e**2))
                command_line = f' {head} {int(n_qubits)} {int(nb)} {e} {int(n_runs)} {int(n_traj)} {load_state} {load_obs} {obs}'
                print(command_line)
                os.system(f'python MQ_ClassicalShadow_Generator.py {command_line}')

# -


