import mediapipe as mp
import cv2
import numpy as np

# 2️. EXTRACCIÓN DE LANDMARKS -------------------------------------------------------
def extract_landmarks(cap):
    mpHands = mp.solutions.hands
    allList = []
    timestamps_fps = []

    with mpHands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps_fps.append(ts / 1000)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            lmList = []

            if results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]

                for id, lm in enumerate(handLms.landmark):
                    if id == 4 or id == 8:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy, ts])

            if len(lmList) == 2:
                allList.append(lmList)

    cap.release()
    return np.array(allList), np.array(timestamps_fps)