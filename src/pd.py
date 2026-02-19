from config import *
from pk import C1_phe, C1_nic
import numpy as np


eps = 1e-8

C1_phe = np.maximum(C1_phe, 0)
C1_nic = np.maximum(C1_nic, 0)


# # R = (
#     R0
#     + (Emax_Rphe * C1_phe) / (EC50_Rphe + C1_phe + eps)
#     + (Emax_Rnic * C1_nic) / (EC50_Rnic + C1_nic + eps)
# )

R = R0
C = C0
Z = Z0