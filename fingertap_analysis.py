from processing.video import get_video_info
from processing.landmarks import extract_landmarks
from processing.fps import analyze_real_fps
from processing.build_dataframe import build_dataframe
from processing.preprocessing import preprocess_signal
from processing.metrics import compute_temporal_metrics
from processing.FFT import compute_fft

# PROCESAR UN VIDEO ---------------------------------------------------------

def analyze_fingertap(video_path):
    
    cap, _ = get_video_info(video_path)
    
    landmarks, timestamps = extract_landmarks(cap)
    analyze_real_fps(timestamps)

    if len(landmarks) == 0:
        raise ValueError("No se detectaron manos")

    df = build_dataframe(landmarks)
    df_processed = preprocess_signal(df)

    temporal_results = compute_temporal_metrics(df_processed)
    fft_results = compute_fft(df_processed)

    # Extraer peaks y troughs fuera del dict
    peaks = temporal_results.pop("peaks")
    troughs = temporal_results.pop("troughs")

    if fft_results is not None:
        temporal_results.update({
            "dominant_freq": fft_results["dominant_freq"],
            "total_energy": fft_results["total_energy"],
            "regularity_index": fft_results["regularity_index"]
        })

    return {
        "df": df_processed,
        "metrics": temporal_results,
        "peaks": peaks,
        "troughs": troughs,
        "freqs": fft_results["freqs"] if fft_results else None,
        "spectrum": fft_results["spectrum"] if fft_results else None
    }