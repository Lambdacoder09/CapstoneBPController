import numpy as np
from config import *
from pk import C1_phe, C1_nic
from pd import R, C

# x[k+1] = A x[k] + B u[k]
# y[k]   = C x[k] + D u[k] crt / to # 

a11 = 1 - dt*(k10_phe + k12_phe) # phe central compartment
a22 = 1 - dt*(k10_nic + k12_nic) # nic central compartment 
a31 = dt * Qmean * (Emax_Rphe * EC50_Rphe) / (EC50_Rphe + C1_phe)^2 # coming from PK
a32 = dt * Qmean * (Emax_Rnic * EC50_Rnic) / (EC50_Rnic + C1_nic)^2
a33 = 1 / (1 + dt/(R*C)) # MAP decay term 

# A is the internal state dynamics
A = np.array([                 
    [a11, 0,   0],
    [0,   a22, 0],
    [a31, a32, a33]
])

# 3 x 2 matrix 
b11 = dt / V1_phe
b22 = dt / V1_nic
b31 = 0 
b32 = 0
# b31 and b32 are zero b/c no direct feedthrough from infusion to MAP

# B is the effect of infusion pumps on the states
B = np.array([                
    [b11, 0],
    [0, b22],
    [0, 0]
])

# we care only about MAP (Mean Arterial Pressure)
C = np.array([
    [0, 0, 1]
])

# D is direct effect input on output. 
# It's zero because there is no direct effect of drugs on MAP
D = np.zeros((1,2))

