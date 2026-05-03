"""
Data collection script for SignBridge.
Captures hand landmark data via MediaPipe and saves to CSV.
"""
import os

import cv2
import mediapipe as mp
import csv
import numpy as np

SIGNS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["NAMASTE", "YES", "NO", "HELP", "THANKS"]
SAMPLES_PER_SIGN = 100

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,     # True for better single-image/sample detection
    max_num_hands=1,
    min_detection_confidence=0.5,
)

os.makedirs("data", exist_ok=True)

# ─── Webcam auto-detect ──────────────────────────────────────────
def open_camera():
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"📷 Camera opened on index {idx}")
                return cap
            cap.release()
    raise RuntimeError("No camera found!")

cap = open_camera()

for sign in SIGNS:
    print(f"\n→ Show sign for: [{sign}]")

    # Wait for user to be ready
    skip = False
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        cv2.putText(
            display, f"Ready for [{sign}]? SPACE=record, Q=skip",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow("Collect Data", display)
        key = cv2.waitKey(1)
        if key == 32:
            break
        elif key == ord("q"):
            skip = True
            break

    if skip:
        continue

    samples = []
    count = 0
    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        display = frame.copy()
        cv2.putText(
            display, f"Recording [{sign}]: {count}/{SAMPLES_PER_SIGN}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )

        if result.multi_hand_landmarks:
            cv2.putText(
                display, "HAND DETECTED - CAPTURING",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2,
            )
            lm = result.multi_hand_landmarks[0].landmark
            row = [sign] + [v for p in lm for v in (p.x, p.y, p.z)]
            samples.append(row)
            count += 1

        cv2.imshow("Collect Data", display)
        if cv2.waitKey(10) == ord("q"):
            break

    with open("data/landmarks.csv", "a", newline="") as f:
        csv.writer(f).writerows(samples)
    print(f"  ✅ Saved {len(samples)} samples for {sign}")

cap.release()
cv2.destroyAllWindows()