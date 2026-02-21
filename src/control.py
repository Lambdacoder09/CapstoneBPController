import numpy as np
from config import *
from pump import HR, Qin
from signal_process import Pin, MAP_est, Z_est, R_est, C_est
from pd import R, C, Z
from scipy.linalg import solve_discrete_are
from pk import C1_phe, C1_nic


# x[k+1] = A x[k] + B u[k]
# y[k]   = C x[k] + D u[k] crt / to # 



# Creating a new state vector 
state_vector = np.column_stack ((
        C1_phe,
        C1_nic,
        MAP_est
))
                                
# create a step state vector, Xk+1, updated state after one time step
step_state_vector = np.coloumn_stack((
    
    np.zeros_likeC1_phe), 
    np.zeros_like(C1_nic),
    np.ones_like(MAP_est) * target_map #we want the arterial pressure to reach target 

)

error_state = state_vector - step_state_vector

#Q matrix or penalizes state error
Q = np.diag([
    1.0,     # C1_phe weight (small)
    1.0,     # C1_nic weight (small)
    100.0    # MAP weight (large — we care most)
])

# R matrix, it penalizes aggresive drug infusion 
R_lqr = np.diag([
    0.1,   # phe infusion penalty
    0.1    # nic infusion penalty
])


# optimal control R

u = -K @ error_state.T

# Solve riccati equation 
P = solve_discrete_are(A, B, Q, R_lqr)
#then computing the gain matrix 
K = np.linalg.inv(B.T @ P @ B + R_lqr) @ (B.T @ P @ A)

# Quadratic cost: squared deviation from target MAP
cost_function = (MAP_est[-1] - target_map)**2

# Controller 

u_history = np.array([
    -K @ (state_vector[k] - step_state_vector[k])
    for k in range(len(state_vector))
])