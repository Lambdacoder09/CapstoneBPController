import numpy as np
import cvxpy as cp
from config import *

# -----------------------------
# Beat-Synchronous Constrained MPC Controller (fixed 5-step horizon)
# -----------------------------
def beat_synchronous_controller(C1_phe_k, C1_nic_k, MAP_k, MAP_error_integral, A, B, Q, R_lqr):
    """
    Multi-step (5-beat horizon) constrained MPC for two-drug infusion with integral action.
    Returns the first infusion command.
    """

    N = 5  # multi-step horizon so integral term isn't ignored

    # Current state
    x0 = np.array([C1_phe_k, C1_nic_k, MAP_k, MAP_error_integral])
    x_ref = np.array([0.0, 0.0, target_map, 0.0])

    # Decision variables
    x = cp.Variable((4, N+1))
    u = cp.Variable((2, N))

    constraints = [x[:,0] == x0]
    cost = 0

    for k in range(N):
        # Dynamics constraint
        constraints += [x[:,k+1] == A @ x[:,k] + B @ u[:,k]]
        # No negative infusion
        constraints += [u[:,k] >= 0]
        # Quadratic cost
        cost += cp.quad_form(x[:,k+1] - x_ref, Q) + cp.quad_form(u[:,k], R_lqr)

    # Solve QP
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.OSQP)

    # Return only the first control action
    u_phe_k, u_nic_k = u.value[:,0]
    return u_phe_k, u_nic_k