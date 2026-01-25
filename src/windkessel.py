from config import *
from pump import Qin
from pd import R, C, Za
import numpy as np

# Major cardiac variable estimates
MAP_est   = Pv + R * Qmean
PP_est    = SV / C
DBP_est   = MAP_est - PP_est / 3

# Initialize blood pressure & flow
P    = np.zeros(N)  # pressure across compliance
Pin  = np.zeros(N)  # inlet (aortic) pressure
Qout = np.zeros(N)  # outflow (Qin - Qc)
P[0]    = DBP_est #Q(0) = 0
Pin[0]  = DBP_est #Q(0) = 0
# Update loop

for k in range(N-1):
    tau = C*R
    alpha = dt / tau
    beta  = dt/ C
    P[k+1]  = (P[k] + beta*Qin[k] + alpha*Pv) / (1.0 + alpha)
    Pin[k]  = P[k] + Za*Qin[k]
    Qout[k] = (P[k] - Pv)/R