import numpy as np
from flask import Flask, render_template, jsonify
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# import simulation components
from config import *
from pump import Qin
from pk import update_pk_phe, update_pk_nic
from pd import compute_R
from windkessel import initialize_windkessel, update_windkessel
from state_space import compute_state_space
from control import compute_lqr_gain, beat_synchronous_controller
from signal_process import BPProcessor

app = Flask(__name__)

def run_simulation():
    # Initialize - Drugs
    C1_phe = np.zeros(N)
    C2_phe = np.zeros(N)
    C1_nic = np.zeros(N)
    C2_nic = np.zeros(N)
    u_phe = np.zeros(N)
    u_nic = np.zeros(N)
    cur_u_phe = 0.0 
    cur_u_nic = 0.0
    
    # Initialize - Cardio State
    R_arr = np.zeros(N)
    P = np.zeros(N)
    P[0] = initialize_windkessel()
    Pin = np.zeros(N)
    Qout = np.zeros(N)
    
    # Initialize - Beat Times
    beat_indices = []
    MAP_beats = []
    
    # Initialize - Signal Processing
    bp = BPProcessor(fs, HR)
    last_trough = 0
    
    # Initialize - Controller
    A_ss, B_ss = compute_state_space()
    K = compute_lqr_gain(A_ss, B_ss, Q, R_lqr)

    # Only check for beats every ~half a beat period (much faster than every sample)
    beat_check_interval = max(1, int(0.5 * beat_period * fs))

    # Pre-extract constants for inlined computation (avoid repeated attribute lookups)
    _eps = 1e-8
    _dt = dt
    _V1p, _k10p, _k12p, _k21p = V1_phe, k10_phe, k12_phe, k21_phe
    _V1n, _k10n, _k12n, _k21n = V1_nic, k10_nic, k12_nic, k21_nic
    _EmaxRp, _EC50Rp = Emax_Rphe, EC50_Rphe
    _EmaxRn, _EC50Rn = Emax_Rnic, EC50_Rnic
    _R0, _C_wk, _Z, _Pv = R0, C, Z, Pv
    _R0_lo, _R0_hi = 0.3 * R0, 3.0 * R0
    _inv_C = 1.0 / C
    _Qin = Qin  # local reference

    for k in range(N - 1):
        u_phe[k] = cur_u_phe
        u_nic[k] = cur_u_nic

        # ── Inline PK: phenylephrine ──
        c1p = C1_phe[k]; c2p = C2_phe[k]; up = cur_u_phe
        dC1p = (up / _V1p) - _k10p * c1p - _k12p * c1p + _k21p * c2p
        dC2p = _k12p * c1p - _k21p * c2p
        C1_phe[k+1] = c1p + _dt * dC1p
        C2_phe[k+1] = c2p + _dt * dC2p

        # ── Inline PK: nicardipine ──
        c1n = C1_nic[k]; c2n = C2_nic[k]; un = cur_u_nic
        dC1n = (un / _V1n) - _k10n * c1n - _k12n * c1n + _k21n * c2n
        dC2n = _k12n * c1n - _k21n * c2n
        C1_nic[k+1] = c1n + _dt * dC1n
        C2_nic[k+1] = c2n + _dt * dC2n

        # ── Inline PD: compute R ──
        c1p_pos = max(c1p, 0.0)
        c1n_pos = max(c1n, 0.0)
        R_phe = (_EmaxRp * c1p_pos) / (_EC50Rp + c1p_pos + _eps)
        R_nic = (_EmaxRn * c1n_pos) / (_EC50Rn + c1n_pos + _eps)
        Rk = _R0 + R_phe + R_nic
        if Rk < _R0_lo: Rk = _R0_lo
        elif Rk > _R0_hi: Rk = _R0_hi
        R_arr[k] = Rk

        # ── Inline Windkessel ──
        Pk = P[k]; Qk = _Qin[k]
        dPdt = (Qk * _inv_C) - (Pk - _Pv) / (Rk * _C_wk)
        P[k+1] = Pk + _dt * dPdt
        Pin[k] = Pk + _Z * Qk
        Qout[k] = (Pk - _Pv) / Rk

        # Only run beat detection periodically, not every sample
        if (k + 1) % beat_check_interval == 0 or k == N - 2:
            peaks, troughs = bp.detect_beats(P[:k+2])

            if len(troughs) > 0 and troughs[-1] != last_trough:
                start = last_trough
                end = troughs[-1]
                last_trough = end
                MAP_k = np.mean(P[start:end])
                MAP_beats.append(MAP_k)
                beat_indices.append(end)
                
                current_u_phe, current_u_nic = beat_synchronous_controller(
                    C1_phe[start], C1_nic[start], MAP_k, K
                )
                cur_u_phe = current_u_phe
                cur_u_nic = current_u_nic

    beat_times = [float(idx * dt) for idx in beat_indices]
    
    # Downsample for frontend (N can be 100k at 1kHz)
    ds = max(1, N // 2000)  # ~2000 points for time-series charts
    # For the pressure waveform, show 5 beats worth of data at full resolution
    wave_samples = int(5 * beat_period * fs)
    # Pick a window after the system has settled (around 10s in)
    wave_start = int(10.0 * fs)
    wave_end = min(wave_start + wave_samples, N)
    # Downsample the waveform window for rendering (target ~2000 points)
    wave_ds = max(1, (wave_end - wave_start) // 2000)
    
    return {
        "time": t[::ds].tolist(),
        "map_beats": [float(m) for m in MAP_beats],
        "beat_times": beat_times,
        "target_map": float(target_map),
        "c1_phe": C1_phe[::ds].tolist(),
        "c1_nic": C1_nic[::ds].tolist(),
        "u_phe": u_phe[::ds].tolist(),
        "u_nic": u_nic[::ds].tolist(),
        # Pressure waveform (high-res window)
        "wave_time": t[wave_start:wave_end:wave_ds].tolist(),
        "wave_pin": Pin[wave_start:wave_end:wave_ds].tolist(),
        "wave_p": P[wave_start:wave_end:wave_ds].tolist(),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    try:
        data = run_simulation()
        return jsonify(data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
