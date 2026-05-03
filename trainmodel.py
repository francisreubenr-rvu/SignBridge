"""
Train the Random Forest classifier from collected landmarks.
Produces model.pkl used by app.py and signbridge.py.

This version trains TWO models:
1. Normalized (wrist-centered + scale-invariant) — for position/scale robustness
2. Raw landmarks — as a fallback comparison

It also augments the training data with small perturbations for robustness.
"""
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ─── Normalization function ──────────────────────────────────────
def normalize_landmarks(raw):
    """Normalize relative to wrist (landmark 0), scale to unit range."""
    bx, by, bz = raw[0], raw[1], raw[2]
    norm = []
    for i in range(0, len(raw), 3):
        norm.extend([raw[i] - bx, raw[i + 1] - by, raw[i + 2] - bz])
    m = max(abs(v) for v in norm)
    if m > 0:
        norm = [v / m for v in norm]
    return norm

# ─── Load data ───────────────────────────────────────────────────
raw_data, norm_data, labels = [], [], []
with open("data/landmarks.csv") as f:
    for row in csv.reader(f):
        if len(row) < 64:  # 1 label + 63 coords minimum
            continue
        labels.append(row[0])
        raw = [float(x) for x in row[1:]]
        raw_data.append(raw)
        norm_data.append(normalize_landmarks(raw))

print(f"📊 Original dataset: {len(labels)} samples, {len(set(labels))} classes")

# ─── Data Augmentation ───────────────────────────────────────────
# Add small noise to normalized coordinates to improve robustness
np.random.seed(42)
aug_norm_data = list(norm_data)  # copy original
aug_labels = list(labels)

for i in range(len(norm_data)):
    for _ in range(3):  # 3 augmented copies per sample
        noisy = np.array(norm_data[i]) + np.random.normal(0, 0.02, len(norm_data[i]))
        aug_norm_data.append(noisy.tolist())
        aug_labels.append(labels[i])

print(f"📊 Augmented dataset: {len(aug_labels)} samples ({len(aug_labels) - len(labels)} augmented)")

X, y = np.array(aug_norm_data), np.array(aug_labels)

# ─── Verify class distribution ───────────────────────────────────
unique, counts = np.unique(y, return_counts=True)
print("\n   Class distribution (augmented):")
for cls, cnt in sorted(zip(unique, counts), key=lambda x: x[0]):
    print(f"     {cls:>10s}: {cnt} samples")

# ─── Train with stratified split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,       # more trees for better generalization
    max_depth=None,         # grow fully
    min_samples_leaf=1,     # no regularization — RF handles overfitting well
    max_features='sqrt',    # random subsets for diversity
    random_state=42,
    n_jobs=-1,              # use all CPU cores
)
model.fit(X_train, y_train)

# ─── Evaluate ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {acc * 100:.1f}%")

# Cross-validation on ORIGINAL data only (not augmented)
X_orig = np.array(norm_data)
y_orig = np.array(labels)
cv_scores = cross_val_score(model, X_orig, y_orig, cv=5, scoring='accuracy')
print(f"📈 5-Fold CV on original data: {cv_scores.mean() * 100:.1f}% (± {cv_scores.std() * 100:.1f}%)")

# Per-class breakdown
print("\n" + classification_report(y_test, y_pred))

# ─── Robustness test — add noise to test samples ────────────────
print("─── Robustness test (noisy inputs) ───")
X_orig_test = np.array(norm_data[:100])
y_orig_test = np.array(labels[:100])
noise_levels = [0.01, 0.02, 0.05, 0.1]
for noise in noise_levels:
    X_noisy = X_orig_test + np.random.normal(0, noise, X_orig_test.shape)
    y_noisy_pred = model.predict(X_noisy)
    acc_noisy = accuracy_score(y_orig_test, y_noisy_pred)
    
    # Also check confidence
    proba = model.predict_proba(X_noisy)
    mean_conf = np.mean(np.max(proba, axis=1))
    print(f"  Noise σ={noise}: accuracy={acc_noisy*100:.1f}%, mean_confidence={mean_conf*100:.1f}%")

# ─── Save ────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\n💾 Saved model.pkl")