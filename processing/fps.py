import numpy as np

# 3. ANÁLISIS FPS REAL -------------------------------------------------
def analyze_real_fps(timestamps):
    dt = np.diff(timestamps)

    if len(dt) == 0:
        print("No se pudo calcular FPS real.")
        return None

    fps_real = 1 / dt
    
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    cv = std_dt / mean_dt

    return fps_real