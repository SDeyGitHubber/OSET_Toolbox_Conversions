import numpy as np
import scipy.signal as signal
import scipy.stats as stats

def lp_filter_zero_phase(ecg: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    Mimic MATLAB's lp_filter_zero_phase function for baseline removal.
    
    Parameters:
    - ecg: ECG signal (leads x samples)
    - fc: Cut-off frequency (Hz)
    - fs: Sampling frequency (Hz)
    
    Returns:
    - Baseline signal
    """
    nyquist = fs / 2
    b, a = signal.butter(2, fc / nyquist, btype='low', analog=False)  # 2nd-order Butterworth
    return signal.filtfilt(b, a, ecg, axis=1)  # Zero-phase filtering like MATLAB

def ecg_polarity(ecg: np.ndarray, fs: float, fc: float = 3.0) -> np.ndarray:
    """
    Calculate ECG polarity by removing baseline and computing skewness.
    
    Parameters:
    - ecg: (leads x samples) Multilead ECG matrix
    - fs: Sampling frequency (Hz)
    - fc: Cut-off frequency for baseline removal (default = 3.0 Hz)
    
    Returns:
    - polarity: Boolean array (1 for positive, 0 for negative)
    """
    baseline = lp_filter_zero_phase(ecg, fc, fs)  # Baseline removal
    skw = stats.skew(ecg - baseline, axis=1)  # Compute skewness
    polarity = skw >= 0  # Positive skew means normal polarity
    
    return polarity