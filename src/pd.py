from config import *
from pk import C1_phe, C1_nic
import numpy as np

def compute_R(C1_phe, C1_nic):
    
    eps = 1e-8

    C1_phe = np.maximum(C1_phe, 0)
    C1_nic = np.maximum(C1_nic, 0)

    R_phe = (Emax_Rphe * C1_phe) / (EC50_Rphe + C1_phe + eps)
    R_nic = (Emax_Rnic * C1_nic) / (EC50_Rnic + C1_nic + eps)

    R = R0 + R_phe + R_nic
    R = np.clip(R, 0.3*R0, 3*R0)    
    # np.clip limits value to stay within a range (array, min_value, max_value)

    return R

R = compute_R(C1_phe, C1_nic)
C = C0
Z = Z0


