from config import *
import numpy as np

# Blood flow shape function (systolic only)
def Qsys(shape, scale, t):
    return (t**(shape-1)) * np.exp(-t/scale)  # gamma-like profile

# --- Compute area of unscaled systolic profile ---
tb_sys = np.arange(0.0, sys_time, dt)
kernel_sys = Qsys(shape, scale, tb_sys)

area_sys = np.trapezoid(kernel_sys, tb_sys)

# Scale factor so that inflow per beat = SV
Qscale = SV / area_sys

# --- Generate full Qin over N samples ---
Qin = np.zeros(N)
for k in range(N):
    tb = t[k] % beat_period
    if tb < sys_time:
        Qin[k] = Qscale * Qsys(shape, scale, tb)
    else:
        Qin[k] = 0.0   # no diastolic decay, flat zero
