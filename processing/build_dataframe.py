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

# 4️. CONSTRUIR DATAFRAME DE DISTANCIAS ---------------------------------------------------------
def build_dataframe(np_allList):
    length = []

    for i in range(np_allList.shape[0]):
        x1, y1 = np_allList[i][0][1], np_allList[i][0][2]
        x2, y2 = np_allList[i][1][1], np_allList[i][1][2]
        ts = np_allList[i][0][3]

        dist = math.hypot(x2 - x1, y2 - y1)
        length.append([ts, dist])

    df = pd.DataFrame(length, columns=['time', 'dist'])

    # eliminar timestamps duplicados
    df['diff'] = df['time'].diff()
    df = df[df['diff'] > 0]
    df.drop(columns=['diff'], inplace=True)

    if df.empty:
        raise ValueError("DataFrame vacío después de limpiar timestamps.")

    # Normalización
    df['amp'] = (df['dist'] / df['dist'].max()) * 100
    #df.drop(columns=['dist'], inplace=True)

    # Limitar a primeros 20 segundos ---> MODIFICAR ESTO SI EL VIDEO ES MÁS LARGO
    df = df[df['time'] <= 20000]

    return df