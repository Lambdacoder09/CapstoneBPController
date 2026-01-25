import numpy as np
import matplotlib.pyplot as plt
from config import *
from pump import Qin
from pk import C1_phe, C1_nic
from windkessel import Pin

beats_to_show = 3
tmax = beats_to_show * beat_period
mask = t <= tmax

plt.figure(figsize=(8,4))
plt.plot(t[mask], Qin[mask], label="Gamma-variate inflow")
plt.xlabel("Time (s)")
plt.ylabel("Flow (L/s)")
plt.title("Cardiac inflow waveform")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(t[mask], Pin[mask], label="Arterial Pressure Pin")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mmHg)")
plt.title("Arterial pressure waveform")
plt.legend()
plt.grid(True)
plt.show()

# fig, ax1 = plt.subplots(figsize=(8,4))
# ax1.plot(t, C1_phe, 'b-', label="PE")
# ax1.set_xlabel("Time (s)")
# ax1.set_ylabel("[PE]", color='b')

# ax2 = ax1.twinx()
# ax2.plot(t, C1_nic, 'r-', label="NI")
# ax2.set_ylabel("[NI]", color='r')

# fig.tight_layout()
# plt.show()