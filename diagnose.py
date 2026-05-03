"""
Deep diagnostic: capture one frame with hand landmarks, normalize them,
compare against training data, and show what the model thinks.
"""
import os

import cv2
import mediapipe as mp
import pickle
import numpy as np
import csv
import time

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
print(f"✅ Model: {len(model.classes_)} classes, {model.n_features_in_} features")

# Load some training samples for comparison
train_samples = {}
with open("data/landmarks.csv") as f:
    for row in csv.reader(f):
        label = row[0]
        if label not in train_samples:
            raw = [float(x) for x in row[1:]]
            train_samples[label] = raw

# Normalize function (MUST match trainmodel.py)
def normalize(raw):
    bx, by, bz = raw[0], raw[1], raw[2]
    norm = []
    for i in range(0, len(raw), 3):
        norm.extend([raw[i] - bx, raw[i + 1] - by, raw[i + 2] - bz])
    m = max(abs(v) for v in norm)
    if m > 0:
        norm = [v / m for v in norm]
    return norm

# Test model on training data first
print("\n─── Model test on stored training samples ───")
for label in sorted(list(train_samples.keys()))[:5]:
    raw = train_samples[label]
    norm = normalize(raw)
    features = np.array([norm])
    proba = model.predict_proba(features)[0]
    pred = model.classes_[proba.argmax()]
    conf = proba.max()
    print(f"  {label}: predicted={pred} conf={conf*100:.1f}%")

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Warm up
for _ in range(30):
    cap.read()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
)

print("\n─── Capturing live frames with hand detection ───")
print("Show your hand to the camera. Looking for 20 detections...\n")

detections = 0
attempts = 0
max_attempts = 500

while detections < 20 and attempts < max_attempts:
    ret, frame = cap.read()
    if not ret:
        continue
    attempts += 1
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    
    if result.multi_hand_landmarks:
        detections += 1
        lm = result.multi_hand_landmarks[0].landmark
        
        # Method A: normalize using the app.py object-based approach
        bx, by, bz = lm[0].x, lm[0].y, lm[0].z
        norm_a = []
        for p in lm:
            norm_a.extend([p.x - bx, p.y - by, p.z - bz])
        m = max(abs(v) for v in norm_a)
        if m > 0:
            norm_a = [v / m for v in norm_a]
        
        # Method B: extract raw list first (like collectdata.py saves), then normalize
        raw_list = [v for p in lm for v in (p.x, p.y, p.z)]
        norm_b = normalize(raw_list)
        
        # Are they the same?
        diff = max(abs(a - b) for a, b in zip(norm_a, norm_b))
        
        features_a = np.array([norm_a])
        features_b = np.array([norm_b])
        
        proba_a = model.predict_proba(features_a)[0]
        proba_b = model.predict_proba(features_b)[0]
        
        pred_a = model.classes_[proba_a.argmax()]
        conf_a = proba_a.max()
        pred_b = model.classes_[proba_b.argmax()]
        conf_b = proba_b.max()
        
        if detections <= 5 or detections % 5 == 0:
            print(f"  Detection {detections}:")
            print(f"    Wrist: ({lm[0].x:.3f}, {lm[0].y:.3f}, {lm[0].z:.5f})")
            print(f"    Method A (object): pred={pred_a} conf={conf_a*100:.1f}%")
            print(f"    Method B (list):   pred={pred_b} conf={conf_b*100:.1f}%")
            print(f"    Max diff A vs B:   {diff:.10f}")
            print(f"    Feature vector[:6]: {[f'{v:.4f}' for v in norm_a[:6]]}")
            
            # Compare against closest training sample
            best_label = pred_a
            if best_label in train_samples:
                train_norm = normalize(train_samples[best_label])
                train_diff = np.mean([abs(a - b) for a, b in zip(norm_a, train_norm)])
                print(f"    Avg diff vs train '{best_label}': {train_diff:.4f}")
            print()
    
    time.sleep(0.03)

print(f"\n─── Summary: {detections}/{attempts} frames had hand detection ───")

cap.release()
hands.close()
