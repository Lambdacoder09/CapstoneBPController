from config import *
#from control import u_phe, u_nic
import numpy as np

u_phe = np.ones(N) * 0.01
u_nic = np.ones(N) * 0.01 

# State arrays: central and peripheral concentrations
C1_phe = np.zeros(N)
C2_phe = np.zeros(N)
C1_nic = np.zeros(N)
C2_nic = np.zeros(N)

# 2 Compartment PK update loop
for k in range(N-1):
    # Phenylephrine
    dC1_phe = (u_phe[k]/V1_phe) \
              - k10_phe*C1_phe[k] \
              - k12_phe*C1_phe[k] \
              + k21_phe*C2_phe[k]
    dC2_phe = k12_phe*C1_phe[k] - k21_phe*C2_phe[k]
    C1_phe[k+1] = C1_phe[k] + dt*dC1_phe
    C2_phe[k+1] = C2_phe[k] + dt*dC2_phe

    # Nicardipine
    dC1_nic = (u_nic[k]/V1_nic) \
              - k10_nic*C1_nic[k] \
              - k12_nic*C1_nic[k] \
              + k21_nic*C2_nic[k]
    dC2_nic = k12_nic*C1_nic[k] - k21_nic*C2_nic[k]
    C1_nic[k+1] = C1_nic[k] + dt*dC1_nic
    C2_nic[k+1] = C2_nic[k] + dt*dC2_nic