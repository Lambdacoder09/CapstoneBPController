import numpy as np
from config import *
import plots
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import beat_synchronous_controller
from signal_process import BPProcessor

# -----------------------
# Initialize simulation arrays
# -----------------------

# Drugs
C1_phe = np.zeros(N)
C2_phe = np.zeros(N)
C1_nic = np.zeros(N)
C2_nic = np.zeros(N)
u_phe = np.zeros(N)
u_nic = np.zeros(N)
current_u_phe = 0.0
current_u_nic = 0.0

# Cardio state
R = np.zeros(N)
P = np.zeros(N)
P[0] = initialize_windkessel(SV=SV, HR=HR, Pv=Pv, R0=R0)
Pin = np.zeros(N)
Qout = np.zeros(N)

# Beat tracking (dynamically sized because total # of beats should be treated as unknown)
beat_indices = []
MAP_beats = []
MAP_error_integral = 0
last_trough = 0

# Signal processing
bp = BPProcessor(fs, HR)

# Warm-up (avoid transients)

for k in range(warmup_steps - 1):
    # PK update (baseline, no infusion)
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(C1_phe[k], C2_phe[k], current_u_phe)
    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(C1_nic[k], C2_nic[k], current_u_nic)

    # PD update
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # Windkessel update
    P[k+1], Pin[k], Qout[k] = update_windkessel(P[k], R[k], Qin[k])

# -----------------------
# Main simulation loop
# -----------------------
for k in range(warmup_steps - 1, N - 1):
    # 1. Apply current infusion command
    u_phe[k] = current_u_phe
    u_nic[k] = current_u_nic

    # 2. PK update
    C1_phe[k+1], C2_phe[k+1] = update_pk_phe(C1_phe[k], C2_phe[k], u_phe[k])
    C1_nic[k+1], C2_nic[k+1] = update_pk_nic(C1_nic[k], C2_nic[k], u_nic[k])

    # 3. PD update
    R[k] = compute_R(C1_phe[k], C1_nic[k])

    # 4. Windkessel update
    P[k+1], Pin[k], Qout[k] = update_windkessel(P[k], R[k], Qin[k])

    # 5. Beat detection
    peaks, troughs = bp.detect_beats(P[:k+1])

    if len(troughs) > 0 and troughs[-1] != last_trough:
        start = last_trough
        end = troughs[-1]
        last_trough = end

        # 6. Compute map and update related data
        MAP_k = np.mean(P[start:end])
        MAP_error_integral += MAP_k - target_map
        MAP_beats.append(MAP_k)
        beat_indices.append(end)

        # 7. Compute linearized state-space for this beat
        A, B = compute_state_space(C1_phe[start], C1_nic[start], R[start])

        # 8. Run constrained LQR controller (QP)
        current_u_phe, current_u_nic = beat_synchronous_controller(
            C1_phe[start],
            C1_nic[start],
            MAP_k,
            MAP_error_integral,
            A,
            B,
            Q=Q,
            R_lqr=R_lqr
        )

plots.plot_pressure_waveform(t=t, P=P, warmup_time=warmup_time)
plots.plot_map_response(beat_times=[bi * dt for bi in beat_indices], 
                        MAP_beats=MAP_beats, target_map=target_map, warmup_time=warmup_time)
plots.plot_drug_concentrations(t=t, C1_phe=C1_phe, C1_nic=C1_nic, warmup_time=warmup_time)
plots.plot_infusion(t=t, u_phe=u_phe, u_nic=u_nic, warmup_time=warmup_time)
plots.plot_resistance(t=t, R=R, warmup_time=warmup_time)


# Send to database: 
# P(pressure)
# MAP_beats, beat_times (beat averaged MAP and beat time indices) 
# C1_phe, C1_nic (drug concentrations)