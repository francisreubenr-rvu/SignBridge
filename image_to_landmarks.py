"""
Extract hand landmarks from a folder of ASL sign images using MediaPipe.
Saves landmarks in the SAME CSV format as collectdata.py so we can
retrain the existing Random Forest model with thousands of samples.

Expected folder structure:
    asl_images/
        A/
            img001.jpg
            img002.jpg
            ...
        B/
            img001.jpg
            ...
        ...

Usage:
    #   python3 image_to_landmarks.py --input asl_images --output data/landmarks.csv [--append]
    """
import os

import cv2
import mediapipe as mp
import csv
import argparse
import time
from pathlib import Path

# ─── MediaPipe setup (static mode for image processing) ──────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,     # True for better single-image/sample detection
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# ─── Label mapping ───────────────────────────────────────────────
# Map folder names to sign labels. Handles common dataset conventions:
#   "A" or "a" → "A"
#   "del" or "DELETE" → skip
#   "space" or "SPACE" → skip
#   "nothing" or "NOTHING" → skip
SKIP_LABELS = {'del', 'delete', 'space', 'nothing', 'none', 'blank'}

# Custom sign folders (if you have separate folders for these)
CUSTOM_LABELS = {'NAMASTE', 'YES', 'NO', 'HELP', 'THANKS'}

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def normalize_label(folder_name):
    """Convert folder name to a sign label, or None to skip."""
    name = folder_name.strip()
    if name.lower() in SKIP_LABELS:
        return None
    # Single letter
    if len(name) == 1 and name.isalpha():
        return name.upper()
    # Multi-letter custom signs
    if name.upper() in CUSTOM_LABELS:
        return name.upper()
    # Numbered folders like "0", "1" ... map to letters A=0, B=1, etc.
    if name.isdigit():
        idx = int(name)
        if 0 <= idx <= 25:
            return chr(ord('A') + idx)
        return None
    # Folder names like "A_train" or "letter_A"
    for part in name.replace('_', ' ').replace('-', ' ').split():
        if len(part) == 1 and part.isalpha():
            return part.upper()
    return name.upper() if len(name) <= 10 else None


def extract_landmarks(image_path):
    """Extract 21 hand landmarks from an image. Returns list of 63 floats or None."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    return [v for p in lm for v in (p.x, p.y, p.z)]


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from ASL sign images"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to root folder containing subfolders per sign (e.g., asl_images/A/, asl_images/B/)"
    )
    parser.add_argument(
        "--output", "-o", default="data/landmarks.csv",
        help="Output CSV path (default: data/landmarks.csv)"
    )
    parser.add_argument(
        "--append", "-a", action="store_true",
        help="Append to existing CSV instead of overwriting"
    )
    parser.add_argument(
        "--max-per-class", "-m", type=int, default=500,
        help="Maximum images to process per sign class (default: 500)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan folders and report what would be processed, without extracting"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_dir():
        print(f"❌ Input folder not found: {input_path}")
        return

    # Scan folders
    print(f"📂 Scanning {input_path}...")
    class_folders = {}
    for item in sorted(input_path.iterdir()):
        if not item.is_dir():
            continue
        label = normalize_label(item.name)
        if label is None:
            print(f"  ⏭️  Skipping folder: {item.name}")
            continue
        images = [f for f in item.iterdir() if f.suffix.lower() in VALID_EXTENSIONS]
        if not images:
            print(f"  ⚠️  No images in: {item.name}")
            continue
        class_folders[label] = images[:args.max_per_class]
        print(f"  ✅ {label}: {len(images)} images (using {len(class_folders[label])})")

    if not class_folders:
        print("❌ No valid sign folders found!")
        return

    total = sum(len(v) for v in class_folders.values())
    print(f"\n📊 Found {len(class_folders)} classes, {total} images to process")

    if args.dry_run:
        print("\n🔍 Dry run — no landmarks extracted.")
        return

    # Extract landmarks
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    mode = "a" if args.append else "w"
    total_extracted = 0
    total_failed = 0

    with open(args.output, mode, newline="") as f:
        writer = csv.writer(f)
        for label in sorted(class_folders.keys()):
            images = class_folders[label]
            extracted = 0
            failed = 0
            start = time.time()

            for i, img_path in enumerate(images):
                landmarks = extract_landmarks(img_path)
                if landmarks:
                    writer.writerow([label] + landmarks)
                    extracted += 1
                else:
                    failed += 1

                # Progress every 50 images
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    print(f"    {label}: {i+1}/{len(images)} "
                          f"({extracted} ok, {failed} fail) [{rate:.0f} img/s]")

            total_extracted += extracted
            total_failed += failed
            elapsed = time.time() - start
            print(f"  ✅ {label}: {extracted}/{len(images)} extracted "
                  f"({failed} no hand) [{elapsed:.1f}s]")

    print(f"\n{'='*50}")
    print(f"  📊 Total: {total_extracted} landmarks saved, {total_failed} failed")
    print(f"  💾 Saved to: {args.output}")
    print(f"\n  Next step: run 'python3 trainmodel.py' to retrain the model!")


if __name__ == "__main__":
    main()
