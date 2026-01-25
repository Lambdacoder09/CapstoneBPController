import numpy as np

# Generic Simulation Parameters
dt = 0.01                   # increment
T  = 100                    # total sim time in seconds
N  = int(T/dt)              # number of discrete time steps
t  = np.arange(N) * dt      # time vector in seconds

# Control Parameters
starting_map = 70 # initial mean arterial pressure
target_map = 80   # target mean arterial pressure mmHg

# Pump Parameters (input blood flow Qin)
HR = 75/60                        # heart beats per second not minute
SV = 70/1000                      # litres per heart beat
beat_period = 1.0/HR              # seconds per heart beat including diastole
sys_frac = 0.2                    # fraction of time heart is beating
sys_time = beat_period * sys_frac # seconds of time heart is beating
shape, scale = 3, sys_time/3   # gamma variable parameters for systole
tau_d = 0.15 * beat_period        # diastolic decay time constant
Qmean = SV / beat_period          # avg blow flow through cycle - tied to stroke volume

# Cardiovascular (Windkessel) Parameters
R0 = 1e3  # Peripheral resistance, mmHg·s/L
C0 = 1e-3  # Arterial compliance, L/mmHg
Z0 = 5        # Aortal characteristic imperance, mmHg·s/L
Pv = 5        # Background venous pressure, mmHg

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
Emax_Rphe = 0
EC50_Rphe = 0
Emax_Rphe = 0
EC50_Rphe = 0
Emax_Zphe = 0
EC50_Zphe = 0
# PD Parameters - Nicardipine
Emax_Rnic = 0
EC50_Rnic = 0
Emax_Rnic = 0
EC50_Rnic = 0
Emax_Znic = 0
EC50_Znic = 0
