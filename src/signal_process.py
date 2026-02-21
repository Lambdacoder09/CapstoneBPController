import numpy as np
from config import *
from windkessel import P, Pin
from pump import HR
from scipy.signal import butter, filtfilt, find_peaks

fs = int(1/dt)

class BPProcessor:
  
    def __init__(self, pressure_waveform, sampling_rate, heart_rate):
        self.raw = np.asarray(pressure_waveform)
        self.fs = sampling_rate
        self.hr = heart_rate

        self.beat_interval_sec = 60.0 / self.hr
        self.samples_per_beat = int(self.fs * self.beat_interval_sec)

        self.filtered = None
        self.peaks = None
        self.troughs = None
        self.SBP = None
        self.DBP = None
        self.MAP = None

    def bandpass_filter(self, order=4):
        nyq = 0.5 * self.fs
        lowcut = (self.hr / 3.0) / nyq
        highcut = (self.hr * 5.0) / nyq
        b, a = butter(order, [lowcut, highcut], btype='band')
        self.filtered = filtfilt(b, a, self.raw)
        return self.filtered

    def detect_beats(self):
        signal = self.filtered if self.filtered is not None else self.raw
        min_distance = self.samples_per_beat

        self.peaks, _ = find_peaks(signal, distance=min_distance)
        self.troughs, _ = find_peaks(-signal, distance=min_distance)
        self.SBP = signal[self.peaks]
        self.DBP = signal[self.troughs]

        self.reject_artifacts()

        return self.peaks, self.troughs, self.SBP, self.DBP

    def estimate_map(self):
        if self.troughs is None:
            raise RuntimeError("Run detect_beats() first.")

        signal = self.filtered if self.filtered is not None else self.raw
        MAP = []

        for i in range(len(self.troughs) - 1):
            start = self.troughs[i]
            end = self.troughs[i+1]
            MAP.append(np.mean(signal[start:end]))

        self.MAP = np.array(MAP)
        return self.MAP

    def reject_artifacts(self, sbp_range=(60, 250), dbp_range=(30, 150)):
        if self.SBP is None or self.DBP is None:
            return

        valid_indices = [
            i for i in range(min(len(self.SBP), len(self.DBP)))
            if sbp_range[0] <= self.SBP[i] <= sbp_range[1]
            and dbp_range[0] <= self.DBP[i] <= dbp_range[1]
        ]

        self.peaks = self.peaks[valid_indices]
        self.troughs = self.troughs[valid_indices]
        self.SBP = self.SBP[valid_indices]
        self.DBP = self.DBP[valid_indices]


# run process immediately 

bp = BPProcessor(P, fs, HR)

bp.bandpass_filter()
bp.detect_beats()
MAP_beats = bp.estimate_map()

beat_indices = bp.troughs