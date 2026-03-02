import numpy as np

# Generic Simulation Parameters
dt = 0.05                   # sim rate
fs = int(1/dt)              # sampling rate
T  = 500                    # total sim time in seconds
N  = int(T/dt)              # number of discrete time steps
t  = np.arange(N) * dt      # time vector in seconds
warmup_time = T * 0.2       # avoid transients
warmup_steps = int(warmup_time / dt)

# Main cardiovascular parameters
HR = 75/60                        # heart beats per second not minute
SV = 70/1000                      # litres per heart beat
beat_period = 1.0/HR              # seconds per heart beat including diastole
sys_frac = 0.2                    # fraction of time heart is beating
sys_time = beat_period * sys_frac # seconds of time heart is beating
shape, scale = 3, sys_time/3      # gamma variable parameters for systole
tau_d = 0.15 * beat_period        # diastolic decay time constant
Qmean = SV / beat_period          # avg blow flow through cycle - tied to stroke volume

# Windkessel parameters
R0 = 1e3                          # peripheral resistance, mmHg·s/L
C = 1.5e-3                        # arterial compliance, L/mmHg
Z = 0.05                          # aortal characteristic imperance, mmHg·s/L
Pv = 5                            # background venous pressure, mmHg

# PK Parameters - Phenylephrine
V1_phe, V2_phe = 10.0, 20.0   # L, central and peripheral volumes
k10_phe = 0.2                 # elimination rate from central (1/s)
k12_phe = 0.05                # rate central -> peripheral (1/s)
k21_phe = 0.03                # rate peripheral -> central (1/s)
# PK Parameters - Nicardipine
V1_nic, V2_nic = 20.0, 40.0   # L, central and peripheral volumes
k10_nic = 0.15                # elimination rate from central (1/s)
k12_nic = 0.04                # rate central -> peripheral (1/s)               
k21_nic = 0.02                # rate peripheral -> central (1/s)

# PD Parameters - Phenylephrine
Emax_Rphe = 0.6 * R0    # Able to increase resistance by 60%
EC50_Rphe = 1.5         # ug/L
# PD Parameters - Nicardipine
Emax_Rnic = -0.6 * R0   # Able to drop resistance by 60%
EC50_Rnic = 1           # ug/L   

# Control Parameters
target_map = 95    # target mean arterial pressure mmHg
Q = np.diag([      # state penalty matrix
    0,
    0,
    1,               
    1            # the 'integral term' should be small
])
R_lqr = np.diag([  # control penalty matrix
    EC50_Rphe * 1, # base infusion penalty on potency
    EC50_Rnic * 1
])
