import numpy as np
from config import *
from pump import HR, Qin
from signal_process import Pin, MAP_est, Z_est, R_est, C_est
from pd import R, C, Z
from pk import C1_phe, C1_nic

# Creating a new state vector 
state_vector = np.column_stack ((
        C1_phe,
        C1_nic,
        MAP_est
))
                                




# Quadratic cost: squared deviation from target MAP
cost_function = (MAP_est[-1] - target_map)**2

# creating a new state vector 
