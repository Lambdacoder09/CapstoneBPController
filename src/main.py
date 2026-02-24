import numpy as np
import matplotlib.pyplot as plt

from config import *
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import compute_lqr_gain
from signal_process import BPProcessor


# =====================================
# Linear Model + LQR Gain
# =====================================

A, B = compute_state_space()
K = compute_lqr_gain(A, B, Q, R_lqr)


# =====================================
# Allocate State Arrays
# =====================================

C1_phe = np.zeros(N)
C2_phe = np.zeros(N)

C1_nic = np.zeros(N)
C2_nic = np.zeros(N)

R = np.zeros(N)

P = np.zeros(N)
Pin = np.zeros(N)
Qout = np.zeros(N)

u_phe = np.zeros(N)
u_nic = np.zeros(N)

P[0] = initialize_windkessel()


# =====================================
# Run Simulation
# =====================================

for k in range(N - 1):

    # PK
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(
        C1_phe[k], C2_phe[k], u_phe[k]
    )

    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(
        C1_nic[k], C2_nic[k], u_nic[k]
    )

    # PD
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # Windkessel
    P[k+1], Pin[k], Qout[k] = update_windkessel(
        P[k], R[k], Qin[k]
    )

print("Simulation complete.")


# =====================================
# Beat Detection
# =====================================

fs = int(1/dt)
bp = BPProcessor(fs, HR)

# Direct detection on raw pressure
peaks, troughs = bp.detect_beats(P)



if len(troughs) >= 2:
    MAP_beats = bp.estimate_map(P)
    beat_indices = troughs[:-1]  # align lengths
else:
    MAP_beats = np.array([])
    beat_indices = np.array([])

# =====================================
# Plot Estimated MAP Only
# =====================================

# plt.figure(figsize=(10, 5))

if len(MAP_beats) > 0:

    beat_times = t[beat_indices]

    plt.plot(beat_times, MAP_beats, 'o-', label="Estimated MAP")
    plt.axhline(target_map, linestyle='--', color='black', label="Target MAP")

    plt.xlabel("Time (s)")
    plt.ylabel("Mean Arterial Pressure (mmHg)")
    plt.title("Beat-Averaged MAP")
    plt.legend()
    plt.grid(True)

else:
    print("No MAP beats detected.")

plt.show()



# plt.figure(figsize=(10,5))
# plt.plot(t[:1000], P[:1000])
# plt.show()

# Send to database: 
# P(pressure)
# MAP_beats, beat_times (beat averaged MAP and beat time indices) 
# C1_phe, C1_nic (drug concentrations)