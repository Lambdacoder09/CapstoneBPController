import numpy as np
from config import *

# Initial Pressure = Initial MAP

def initialize_windkessel(SV, HR, Pv, R0):
    CO = SV * HR # cardiac output
    initial_map = Pv + R0 * CO 
    return initial_map * 0.5 # start with low value and let system 'ramp up'

# Pressure, Resistance, Outflow update

def update_windkessel(P_k, R_k, Qin_k): 
    # third order windkessel with Zc (just Z here) 'characteristic impedance' on top of
    # R and C

    dPdt = (Qin_k / C) - (P_k - Pv) / (R_k * C)

    P_k1 = P_k + dt * dPdt

    Pin_k  = P_k + Z * Qin_k
    Qout_k = (P_k - Pv) / R_k

    return P_k1, Pin_k, Qout_k
