import numpy as np
from config import *


def compute_state_space(C1_phe=0.0, C1_nic=0.0, R_op=R0):

    eps = 1e-8  # small number to avoid divide-by-zero

    # State Vector:
    # x = [C1_phe, C1_nic, MAP, MAP_error_integral]

    # PK dynamics
    # Central compartment discrete decay
    a11 = 1 - dt * (k10_phe + k12_phe)
    a22 = 1 - dt * (k10_nic + k12_nic)

    # Linearized PD sensitivity

    # dR/dC1_phe
    dR_dC1 = (Emax_Rphe * EC50_Rphe) / (EC50_Rphe + C1_phe + eps) ** 2

    # dR/dC1_nic
    dR_dC2 = (Emax_Rnic * EC50_Rnic) / (EC50_Rnic + C1_nic + eps) ** 2

    # MAP dynamics (Windkessel linearization)

    # Sensitivity of MAP to concentrations
    a31 = dt * Qmean * dR_dC1
    a32 = dt * Qmean * dR_dC2

    # MAP self-dynamics
    a33 = 1 - dt / (R_op * C)

    # Integral of MAP error dynamics
    # z[k+1] = z[k] + dt*(target_map - MAP_k)
    # Linearized: z[k+1] = z[k] - dt*MAP_k

    a43 = dt       # dependence on MAP
    a44 = 1.0      # integrator persistence

    # Assemble full A matrix (4x4)-

    A = np.array([
        [a11, 0.0, 0.0, 0.0],   # C1_phe
        [0.0, a22, 0.0, 0.0],   # C1_nic
        [a31, a32, a33, 0.0],   # MAP
        [0.0, 0.0, a43, a44]    # Integral state
    ])

    # Input matrix

    # Direct effect on central compartments
    b11 = dt / V1_phe
    b22 = dt / V1_nic

    # Indirect effect on MAP via resistance
    b31 = dt * Qmean * dR_dC1 / V1_phe
    b32 = dt * Qmean * dR_dC2 / V1_nic

    # Integrator has no direct input
    B = np.array([
        [b11, 0.0],   # phe infusion → C1_phe
        [0.0, b22],   # nic infusion → C1_nic
        [b31, b32],   # infusion → MAP
        [0.0, 0.0]    # no direct effect on integral
    ])

    return A, B