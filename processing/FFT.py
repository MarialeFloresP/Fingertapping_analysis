import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.stats import entropy

# 7. ANÁLISIS FOURIER -------------------------------------------------
def compute_fft(df):

    t = df['time'].values / 1000.0

    signal = df['amp_smooth'].values

    dt = np.mean(np.diff(t))

    fs = 1 / dt

    signal_detrended = signal - np.mean(signal)

    N = len(signal_detrended)

    fft_vals = np.fft.fft(signal_detrended)

    fft_freq = np.fft.fftfreq(N, d=dt)

    positive = fft_freq > 0

    freqs = fft_freq[positive]

    spectrum = (np.abs(fft_vals[positive])**2) / N

    dominant_freq = freqs[np.argmax(spectrum)]

    total_energy = np.sum(spectrum)

    # REGULARIDAD
    band_width = 0.5

    band_mask = (
        (freqs >= dominant_freq - band_width) &
        (freqs <= dominant_freq + band_width)
    )

    energy_f0 = np.sum(spectrum[band_mask])

    regularity_index = (
        energy_f0 / total_energy
        if total_energy > 0 else 0
    )

    # SPECTRAL ENTROPY
    power_norm = spectrum / np.sum(spectrum)

    spectral_entropy = entropy(power_norm)

    return {

        "freqs": freqs,
        "spectrum": spectrum,

        "dominant_freq": dominant_freq,

        "total_energy": total_energy,

        "regularity_index": regularity_index,

        "spectral_entropy": spectral_entropy
    }

