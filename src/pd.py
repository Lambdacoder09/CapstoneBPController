from config import *
from pk import C1_phe, C2_phe, C1_nic, C2_nic

# Only update R it's easier also ignore second compartment
R = R0 + (Emax_Rphe * C1_phe) / (EC50_Rphe + C1_phe) + (Emax_Rnic * C1_nic) / (EC50_Rnic + C1_nic)
C = C0
Z = Z0
