import numpy as np
from config import *


def initialize_windkessel():
    """
    Returns initial pressure state.
    """
    P0 = starting_map  # start near venous pressure or set externally
    return 0.7 * P0


def update_windkessel(P_k, R_k, Qin_k):

    dPdt = (Qin_k / C) - (P_k - Pv) / (R_k * C)

    P_k1 = P_k + dt * dPdt

    Pin_k  = P_k + Z * Qin_k
    Qout_k = (P_k - Pv) / R_k

    return P_k1, Pin_k, Qout_k


# def update_windkessel(P_k, R_k, Qin_k):
#     """
#     One-step Windkessel update.

#     Inputs:
#         P_k   : current pressure
#         R_k   : current resistance
#         Qin_k : inflow at time k

#     Returns:
#         P_k1  : next pressure
#         Pin_k : inlet pressure
#         Qout_k: outflow
#     """

#     tau = R_k * C
#     alpha = dt / tau
#     beta  = dt / C

#     P_k1 = (P_k + beta * Qin_k + alpha * Pv) / (1.0 + alpha)

#     Pin_k  = P_k + Z * Qin_k
#     Qout_k = (P_k - Pv) / R_k

#     return P_k1, Pin_k, Qout_k