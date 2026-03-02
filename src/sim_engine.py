import numpy as np
from config import *
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import beat_synchronous_controller
from signal_process import BPProcessor

# the pump should go in here eventually.
class BPSimEngine:
    def __init__(self, N=N, target_map=target_map):
        self.N = N
        self.k = 0        # sim step index
        self.mode = "live"
        self.target_map = target_map

        # Drug arrays
        self.C1_phe = np.zeros(N)
        self.C2_phe = np.zeros(N)
        self.C1_nic = np.zeros(N)
        self.C2_nic = np.zeros(N)
        self.u_phe = np.zeros(N)
        self.u_nic = np.zeros(N)
        self.current_u_phe = 0.0
        self.current_u_nic = 0.0

        # Cardio state
        self.R = np.zeros(N)
        self.R[0] = R0
        self.P = np.zeros(N)
        self.P[0] = initialize_windkessel(SV=SV, HR=HR, Pv=Pv, R0=R0)
        self.Pin = np.zeros(N)
        self.Qout = np.zeros(N)

        # Beat tracking
        self.beat_indices = []
        self.MAP_beats = []
        self.MAP_error_integral = 0
        self.last_trough = 0

        # Signal processor
        self.bp = BPProcessor(fs, HR)

    # Run simulation in current mode
    def run(self):
        while self.k < self.N - 1:
            # automatic handoff from live to intervention for now
            if self.k < int(self.N * 0.2):
                self.mode = "live"
                self._run_live_step()
            else:
                self.mode = "intervention"
                self._run_intervention_step()

    def _run_live_step(self):
        """Perform a single step in live mode."""
        # Windkessel update
        self.R[self.k] = R0
        self.P[self.k+1], self.Pin[self.k], self.Qout[self.k] = update_windkessel(
            self.P[self.k], self.R[self.k], Qin[self.k]
        )

        # Beat detection
        peaks, troughs = self.bp.detect_beats(self.P[:self.k+1])
        if len(troughs) > 0 and troughs[-1] != self.last_trough:
            start = self.last_trough
            end = troughs[-1]
            self.last_trough = end

            MAP_k = np.mean(self.P[start:end])
            self.MAP_beats.append(MAP_k)
            self.beat_indices.append(end)

        # Increment step
        self.k += 1

    def _run_intervention_step(self):
        """Perform a single step in intervention mode."""
        # 1. Apply current infusion
        self.u_phe[self.k] = self.current_u_phe
        self.u_nic[self.k] = self.current_u_nic

        # 2. PK update
        self.C1_phe[self.k+1], self.C2_phe[self.k+1] = update_pk_phe(
            self.C1_phe[self.k], self.C2_phe[self.k], self.u_phe[self.k]
        )
        self.C1_nic[self.k+1], self.C2_nic[self.k+1] = update_pk_nic(
            self.C1_nic[self.k], self.C2_nic[self.k], self.u_nic[self.k]
        )

        # 3. PD update
        self.R[self.k] = compute_R(self.C1_phe[self.k], self.C1_nic[self.k])

        # 4. Windkessel update
        self.P[self.k+1], self.Pin[self.k], self.Qout[self.k] = update_windkessel(
            self.P[self.k], self.R[self.k], Qin[self.k]
        )

        # 5. Beat detection
        peaks, troughs = self.bp.detect_beats(self.P[:self.k+1])
        if len(troughs) > 0 and troughs[-1] != self.last_trough:
            start = self.last_trough
            end = troughs[-1]
            self.last_trough = end

            MAP_k = np.mean(self.P[start:end])
            self.MAP_error_integral += MAP_k - self.target_map
            self.MAP_beats.append(MAP_k)
            self.beat_indices.append(end)

            # 6. Compute linearized state-space
            A, B = compute_state_space(self.C1_phe[start], self.C1_nic[start], self.R[start])

            # 7. Beat-synchronous controller
            self.current_u_phe, self.current_u_nic = beat_synchronous_controller(
                self.C1_phe[start],
                self.C1_nic[start],
                MAP_k,
                self.MAP_error_integral,
                A,
                B,
                Q=Q,
                R_lqr=R_lqr
            )

        # Increment step
        self.k += 1