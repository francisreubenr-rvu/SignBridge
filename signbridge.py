"""
SignBridge CLI — Standalone desktop sign language translator.
Uses the same model as the web app, with a local OpenCV window.
"""
import os

import cv2
import mediapipe as mp
import pickle
import serial
import pyttsx3
import numpy as np
import time
import threading
import glob
from collections import deque

# ─── Configuration ───────────────────────────────────────────────
def find_pico_port():
    """Attempt to auto-detect the RPi Pico serial port."""
    patterns = ["/dev/cu.usbmodem*", "/dev/ttyACM*", "/dev/tty.usbmodem*"]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            if "/dev/cu.usbmodem21101" in ports:
                return "/dev/cu.usbmodem21101"
            return ports[0]
    return "/dev/cu.usbmodem21101"

PICO_PORT = find_pico_port()
BAUD_RATE = 115200
CONFIRM_FRAMES = 8
CONFIDENCE_THRESH = 0.55

# ─── Webcam auto-detect ──────────────────────────────────────────
def open_camera():
    for idx in [1, 0]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                print(f"📷 Camera opened on index {idx}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
            cap.release()
    raise RuntimeError("No camera found! Check webcam connection.")

# ─── Load model ──────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print(f"✅ Model loaded — {len(model.classes_)} signs")

# ─── MediaPipe ───────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ─── Pico W OLED ─────────────────────────────────────────────────
try:
    pico = serial.Serial(PICO_PORT, BAUD_RATE, timeout=1)
    print("✅ Pico W connected")
except Exception:
    pico = None
    print("⚠️  Pico W not found — OLED disabled")

# ─── Text-to-speech ──────────────────────────────────────────────
tts = pyttsx3.init()
tts.setProperty('rate', 160)
_tts_lock = threading.Lock()


def speak(text):
    def _speak():
        with _tts_lock:
            tts.say(text)
            tts.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()


def send_oled(msg_type, value):
    if pico:
        try:
            pico.write(f"{msg_type}:{value}\n".encode())
        except Exception:
            pass


# ─── Normalization (must match trainmodel.py exactly) ─────────────
def normalize_landmarks(landmarks):
    bx, by, bz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    norm = []
    for p in landmarks:
        norm.extend([p.x - bx, p.y - by, p.z - bz])
    m = max(abs(v) for v in norm)
    if m > 0:
        norm = [v / m for v in norm]
    return norm


# ─── Main loop ───────────────────────────────────────────────────
cap = open_camera()
word_buffer = []
letter_hist = deque(maxlen=12)
last_letter = ""
last_speak_t = 0
last_hand_t = time.time()

print("🤟 SignBridge running. SPACE=finalize word, BACKSPACE=delete, Q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = None
    confidence = 0.0

    if result.multi_hand_landmarks:
        last_hand_t = time.time()
        lm = result.multi_hand_landmarks[0].landmark
        norm = normalize_landmarks(lm)
        features = np.array([norm])

        proba = model.predict_proba(features)[0]
        confidence = float(proba.max())
        if confidence > CONFIDENCE_THRESH:
            prediction = model.classes_[proba.argmax()]

        mp_draw.draw_landmarks(
            frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
        )
    else:
        if time.time() - last_hand_t > 0.5:
            last_letter = ""

    letter_hist.append(prediction)

    if (
        prediction
        and letter_hist.count(prediction) >= CONFIRM_FRAMES
        and prediction != last_letter
    ):
        last_letter = prediction
        word_buffer.append(prediction)
        current_word = "".join(word_buffer)
        send_oled("LETTER", prediction)
        send_oled("WORD", current_word)
        if time.time() - last_speak_t > 0.8:
            speak(prediction)
            last_speak_t = time.time()

    # ── HUD overlay ──────────────────────────────────────────
    current_word = "".join(word_buffer)
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)
    cv2.putText(
        frame, f"Letter: {last_letter}  ({confidence*100:.0f}%)",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2,
    )
    cv2.putText(
        frame, f"Word:   {current_word}",
        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2,
    )
    cv2.imshow("SignBridge", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        if word_buffer:
            speak("".join(word_buffer))
            send_oled("WORD", "".join(word_buffer))
            word_buffer = []
            last_letter = ""
    elif key == 8:  # BACKSPACE
        if word_buffer:
            word_buffer.pop()
            last_letter = ""
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if pico:
    pico.close()