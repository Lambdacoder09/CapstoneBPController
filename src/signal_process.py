from numpy import kaiser
from numpy.linalg import lstsq
from scipy.signal import butter, filtfilt, find_peaks
from scipy.optimize import curve_fit
from config import *
from windkessel import P, Pin

fs = int(1/dt) # sampling frequency, 1/dt from config
#my change
class BPProcessor:
    """
    Umbrella class for preprocessing and analyzing blood pressure waveforms.

    Parameters
    ----------
    pressure_waveform : array-like
        Raw blood pressure signal (in mmHg).
    sampling_rate : int
        Sampling frequency in Hz (samples per second).
    heart_rate : float
        Heart rate in beats per second (Hz), NOT beats per minute.

        Attributes
    ----------
    raw : ndarray
        Original waveform (unprocessed).
    fs : int
        Sampling frequency in Hz.
    hr : float
        Heart rate in beats per second (Hz).
    beat_interval_sec : float
        Expected beat interval in seconds, derived from HR.
    samples_per_beat : int
        Expected number of samples per beat, derived from HR and fs.

    filtered : ndarray or None
        Preprocessed waveform after bandpass/highpass/lowpass filtering.
    peaks : ndarray or None
        Indices of detected systolic peaks.
    troughs : ndarray or None
        Indices of detected diastolic troughs.
    SBP : ndarray or None
        Systolic blood pressure values (mmHg) at detected peaks.
    DBP : ndarray or None
        Diastolic blood pressure values (mmHg) at detected troughs.
    MAP : ndarray or None
        Mean arterial pressure values (mmHg), beat-by-beat, computed by averaging between troughs.

    CO_over_C : float or None
        Cardiac output over compliance estimate.
    tau : float or None
        RC time constant estimate.
    Za : float or None
        Aortic characteristic impedance estimate.
    """

    def __init__(self, pressure_waveform, sampling_rate, heart_rate):
        self.raw = np.asarray(pressure_waveform)
        self.fs = sampling_rate
        self.hr = heart_rate

        # Beat interval in seconds and samples
        self.beat_interval_sec = 60.0 / self.hr
        self.samples_per_beat = int(self.fs * self.beat_interval_sec)

        # Storage
        self.filtered = None
        self.peaks = None
        self.troughs = None
        self.SBP = None
        self.DBP = None
        self.MAP = None

    # ---------- Preprocessing ----------
    def bandpass_filter(self, order=4):
        """
        Bandpass filter tied to HR:
        - lowcut defaults to HR/3
        - highcut defaults to 5*HR
        """
        nyq = 0.5 * self.fs
        lowcut = (self.hr / 3.0) / nyq
        highcut = (self.hr * 5.0) / nyq
        b, a = butter(order, [lowcut, highcut], btype='band')
        self.filtered = filtfilt(b, a, self.raw)
        return self.filtered

    # ---------- Beat Detection ----------
    def detect_beats(self):
        """
        Detect systolic peaks and diastolic troughs.
        """
        signal = self.filtered if self.filtered is not None else self.raw
        min_distance = self.samples_per_beat

        self.peaks, _ = find_peaks(signal, distance=min_distance)
        self.troughs, _ = find_peaks(-signal, distance=min_distance)
        self.SBP = signal[self.peaks]
        self.DBP = signal[self.troughs]

        # Run artifact rejection immediately after detection
        self.reject_artifacts()

        return self.peaks, self.troughs, self.SBP, self.DBP

    # ---------- MAP Estimation ----------
    def estimate_map(self):
        """
        Compute beat-by-beat MAP by averaging between troughs.
        """
        if self.troughs is None:
            raise RuntimeError("Run detect_beats() first.")
        signal = self.filtered if self.filtered is not None else self.raw
        MAP = []
        for i in range(len(self.troughs) - 1):
            start, end = self.troughs[i], self.troughs[i+1]
            beat_segment = signal[start:end]
            MAP.append(np.mean(beat_segment))
        self.MAP = np.array(MAP)
        return self.MAP

    # ---------- Artifact Rejection ----------
    def reject_artifacts(self, sbp_range=(60, 250), dbp_range=(30, 150)):
        """
        Reject beats with implausible SBP/DBP values.
        """
        if self.SBP is None or self.DBP is None:
            return

        valid_indices = [
            i for i in range(min(len(self.SBP), len(self.DBP)))
            if sbp_range[0] <= self.SBP[i] <= sbp_range[1]
            and dbp_range[0] <= self.DBP[i] <= dbp_range[1]
        ]

        # Keep only valid beats
        self.peaks = self.peaks[valid_indices]
        self.troughs = self.troughs[valid_indices]
        self.SBP = self.SBP[valid_indices]
        self.DBP = self.DBP[valid_indices]

# # CO/C estimation
# def co_over_c_est(t, P)

# # RC time constant estimation
# def tau_est(t, P, ):
    
# # Aorta characteristic impedance estimation
# def Za_est(t, Pin, Q, fs, start):
    

