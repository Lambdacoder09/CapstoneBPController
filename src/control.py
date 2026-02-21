import numpy as np
from scipy.linalg import solve_discrete_are
from config import *
from state_space import A, B


# Cost function matrices

Q = np.diag([
    1.0,
    1.0,
    100.0
])

R_lqr = np.diag([
    0.1,
    0.1
])

# Riccati
P_lqr = solve_discrete_are(A, B, Q, R_lqr)
K = np.linalg.inv(B.T @ P_lqr @ B + R_lqr) @ (B.T @ P_lqr @ A)


# =====================================
# Beat-Synchronous Controller Function
# =====================================

def run_controller(C1_phe, C1_nic, MAP_beats, beat_indices):
    """
    Beat-synchronous LQR controller.

    Inputs:
        C1_phe       : array of central phe concentrations
        C1_nic       : array of central nic concentrations
        MAP_beats    : beat-by-beat MAP values
        beat_indices : trough indices defining beat boundaries

    Returns:
        u_phe, u_nic : infusion arrays (length N)
    """

    u_phe = np.zeros(N)
    u_nic = np.zeros(N)

    for i in range(len(beat_indices) - 1):

        start = beat_indices[i]
        end   = beat_indices[i+1]

        x_k = np.array([
            C1_phe[start],
            C1_nic[start],
            MAP_beats[i]
        ])

        x_ref = np.array([
            0.0,
            0.0,
            target_map
        ])

        u_k = -K @ (x_k - x_ref)

        # Hold infusion constant for this beat
        u_phe[start:end] = u_k[0]
        u_nic[start:end] = u_k[1]

    return u_phe, u_nic