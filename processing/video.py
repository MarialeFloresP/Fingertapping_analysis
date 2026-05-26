import cv2

# 1️. INFORMACIÓN DEL VIDEO -------------------------------------------------
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error abriendo el video")

    fps_metadata = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    return cap, fps_metadata