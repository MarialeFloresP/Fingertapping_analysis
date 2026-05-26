import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.stats import entropy

# 6. CÁLCULO DE MÉTRICAS --------------------------------------------
def compute_temporal_metrics(df):

    signal = df['amp_smooth'].values
    time_sec = df['time'].values / 1000.0

    # PROMINENCE ADAPTATIVO
    prominence = 0.15 * (np.max(signal) - np.min(signal))

    peaks, _ = find_peaks(
        signal,
        prominence=prominence
    )

    troughs, _ = find_peaks(
        -signal,
        prominence=prominence
    )

    # AMPLITUDES
    amps = signal[peaks]

    tap_count = len(amps)

    mean_amp = np.mean(amps) if tap_count > 0 else 0

    std_amp = np.std(amps) if tap_count > 0 else 0

    cv_amp = (
        std_amp / mean_amp
        if mean_amp > 0 else 0
    )

    # DECREMENTO DE AMPLITUD
    def safe_diff(amps, idx):

        if len(amps) > idx:
            return amps[0] - amps[idx]
        else:
            return np.nan

    diff3 = safe_diff(amps, 2)
    diff5 = safe_diff(amps, 4)
    diff7 = safe_diff(amps, 6)
    diff10 = safe_diff(amps, 9)

    # FRECUENCIA E ITI
    total_time = (
        time_sec[-1] - time_sec[0]
        if len(time_sec) > 1 else 0
    )

    ft_simple = (
        tap_count / total_time
        if total_time > 0 else 0
    )

    if tap_count > 1:

        iti = np.diff(time_sec[peaks])

        mean_iti = np.mean(iti)

        std_iti = np.std(iti)

        cv_iti = (
            std_iti / mean_iti
            if mean_iti > 0 else 0
        )

        ft_iti = (
            1 / mean_iti
            if mean_iti > 0 else 0
        )

    else:

        mean_iti = 0
        std_iti = 0
        cv_iti = 0
        ft_iti = 0


    # VELOCIDAD PICO NORMALIZADA
    if tap_count > 1:

        velocity = np.diff(signal) / np.diff(time_sec)

        peak_velocities = []

        for i in range(len(peaks) - 1):

            start = peaks[i]
            end = peaks[i + 1]

            segment = velocity[start:end]

            if len(segment) > 0:

                peak_velocities.append(
                    np.max(np.abs(segment))
                )

        if len(peak_velocities) > 0:

            # Velocidad pico promedio
            mean_velocity = np.mean(peak_velocities)

            # Velocidad máxima global
            max_velocity = np.max(peak_velocities)

            # NORMALIZACIÓN 
            signal_range = (
                np.max(signal) - np.min(signal)
            )

            if signal_range > 0:

                mean_velocity = (
                    mean_velocity / signal_range
                )

                max_velocity = (
                    max_velocity / signal_range
                )

        else:

            mean_velocity = 0
            max_velocity = 0

        # APERTURA vs CIERRE
        positive_vel = velocity[velocity > 0]

        negative_vel = velocity[velocity < 0]

        opening_velocity = (
            np.mean(positive_vel) / signal_range
            if len(positive_vel) > 0 and signal_range > 0
            else 0
        )

        closing_velocity = (
            np.mean(np.abs(negative_vel)) / signal_range
            if len(negative_vel) > 0 and signal_range > 0
            else 0
        )

    else:

        mean_velocity = 0
        max_velocity = 0
        opening_velocity = 0
        closing_velocity = 0

    # FATIGA / DECREMENTO GLOBAL
    if tap_count > 1:

        slope, _, _, _, _ = linregress(
            range(tap_count),
            amps
        )

    else:

        slope = 0

    return {

        "tap_count": tap_count,

        "mean_amp": mean_amp,
        "std_amp": std_amp,
        "cv_amp": cv_amp,

        "diff3": diff3,
        "diff5": diff5,
        "diff7": diff7,
        "diff10": diff10,

        "ft_simple": ft_simple,
        "ft_iti": ft_iti,

        "mean_iti": mean_iti,
        "std_iti": std_iti,
        "cv_iti": cv_iti,

        "mean_velocity": mean_velocity,
        "max_velocity": max_velocity,

        "opening_velocity": opening_velocity,
        "closing_velocity": closing_velocity,

        "slope_amplitude": slope,

        "peaks": peaks,
        "troughs": troughs
    }

