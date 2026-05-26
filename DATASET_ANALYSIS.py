import os
import cv2
import pandas as pd
import numpy as np
from fingertap_analysis import analyze_fingertap

# CONFIGURACIÓN GENERAL
VIDEOS_FOLDER = "videos"
DIAGNOSTIC_CSV = "fis_diagnostic.csv"
OUTPUT_METRICS = "ALL_METRICS.csv"
SIGNALS_FOLDER = "signals"

os.makedirs(SIGNALS_FOLDER, exist_ok=True)

# LEER CSV DIAGNÓSTICO
diagnostic_df = pd.read_csv(DIAGNOSTIC_CSV)


# DATAFRAME GLOBAL MÉTRICAS
all_metrics = []

video_files = os.listdir(VIDEOS_FOLDER)

# RECORRER TODOS LOS VIDEOS
for idx, row in diagnostic_df.iterrows():

    # DATOS DEL CSV
    video_id = row["ID"]
    updrs = row["UPDRS"]

    # EXTRAER MANO DESDE EL ID
    if "DCHA" in video_id:
        mano = "DCHA"

    elif "IZDA" in video_id:
        mano = "IZDA"

    else:
        mano = "UNKNOWN"

    # BUSCAR VIDEO
    video_path = None


    for file in video_files:

        filename_no_ext = os.path.splitext(file)[0]

        if filename_no_ext.upper() == video_id.upper():

            video_path = os.path.join(
                VIDEOS_FOLDER,
                file
            )

            break

    # SI NO EXISTE VIDEO
    if video_path is None:
        print(f"\n Video no encontrado: {video_id}")
        continue

    print("\n================================================")
    print(f"Procesando: {video_id}")
    print("================================================")

    # ANALIZAR VIDEO
    try:
        results = analyze_fingertap(video_path)

    except Exception as e:
        print(f"Error procesando {video_id}")
        print(e)
        continue


    # OBTENER DATAFRAME SEÑAL
    df_signal = results["df"]

    # GUARDAR CSV DE SEÑAL INDIVIDUAL
    signal_output_path = os.path.join(
        SIGNALS_FOLDER,
        f"{video_id}.csv"
    )

    df_signal.to_csv(
        signal_output_path,
        index=False
    )

    print(f"Señal guardada: {signal_output_path}")

    # OBTENER MÉTRICAS
    metrics = results["metrics"]

    # CREAR FILA MÉTRICAS
    metrics_row = {
        "ID": video_id,
        "UPDRS": updrs,
        "MANO": mano
    }

    # Agregar todas las métricas automáticamente
    metrics_row.update(metrics)

    # AGREGAR A LISTA GLOBAL
    all_metrics.append(metrics_row)


# GUARDAR ALL_METRICS
metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    OUTPUT_METRICS,
    index=False
)

print("\n==========================================")
print("ALL_METRICS.csv generado correctamente")
print("==========================================")