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

# 5. PRE-PREOCESAMIENTO DE LA SEÑAL -----------------------------------------
def preprocess_signal(
    df,
    polyorder=3,
    interpolation_threshold=0.02
):

    df = df.copy()

    # SEÑALES CRUDAS

    t = df['time'].values / 1000.0

    dist_raw = df['dist'].values

    amp_raw = df['amp'].values

    # ESTABILIDAD TEMPORAL

    dt = np.diff(t)

    mean_dt = np.mean(dt)

    std_dt = np.std(dt)

    cv = std_dt / mean_dt

    #print("\n===== ESTABILIDAD TEMPORAL =====")

    #print(f"Mean dt: {mean_dt:.6f}")
    #print(f"STD dt: {std_dt:.6f}")
    #print(f"CV: {cv:.4f}")

    # INTERPOLACIÓN
    if cv > interpolation_threshold:

        print("⚠ Muestreo irregular detectado → aplicando interpolación")

        fs_uniform = 1 / mean_dt

        t_uniform = np.arange(
            t[0],
            t[-1],
            1/fs_uniform
        )

        interp_dist = interp1d(
            t,
            dist_raw,
            kind='linear',
            fill_value="extrapolate"
        )

        interp_amp = interp1d(
            t,
            amp_raw,
            kind='linear',
            fill_value="extrapolate"
        )

        dist_raw = interp_dist(t_uniform)

        amp_raw = interp_amp(t_uniform)

        t = t_uniform

    else:

        print("Muestreo suficientemente uniforme → no se interpola")

    # WINDOW SAVITZKY-GOLAY
    window_length = 7
    print(f"Window length Savitzky-Golay: {window_length}")


    # SUAVIZADO
    if len(amp_raw) >= window_length:

        amp_smooth = savgol_filter(
            amp_raw,
            window_length,
            polyorder
        )

        dist_smooth = savgol_filter(
            dist_raw,
            window_length,
            polyorder
        )

    else:
        amp_smooth = amp_raw
        dist_smooth = dist_raw

    # DATAFRAME FINAL
    df_processed = pd.DataFrame({
        'time': t * 1000,
        'dist_raw': dist_raw,
        'amp_raw': amp_raw,
        'dist_smooth': dist_smooth,
        'amp_smooth': amp_smooth
    })

    return df_processed

