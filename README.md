# SignBridge v2: Real-time Sign Language Translator

## 🤟 Project Vision
SignBridge is a high-performance, real-time sign language (ASL) translation system designed to bridge the communication gap between the deaf/hard-of-hearing community and the hearing world. Originally developed for a **CIE 3 ESM project**, it has evolved from a simple gesture detector into a robust standalone kiosk system with a specialized hardware integration layer.

## 🧠 Technical Context
The system uses a **Hybrid Inference Engine**:
1.  **Vision Layer:** MediaPipe Hands extracts 21 3D-landmarks at 30fps.
2.  **ML Layer:** A Random Forest Classifier (trained on 31+ classes) performs classification with a custom temporal stabilization algorithm.
3.  **UI Layer:** A Neobrutalist-themed web dashboard powered by Flask and SocketIO.
4.  **Hardware Layer:** An RPi Pico-driven OLED "Sign Player" for reverse translation (Text-to-Sign).

## 🚀 Key Features
- **Temporal Stabilization:** Uses sliding-window majority voting to eliminate "flicker" during hand transitions.
- **Accuracy Thresholds:** Implements a "Wait & Propose" state for low-confidence detections (20%-50%).
- **Trie-based Autocomplete:** A custom prefix-tree for real-time word suggestions.
- **Neobrutalist Design:** High-contrast, aggressive styling for maximum legibility and modern aesthetic.
- **Reverse Translation:** Integrated speech-to-text dictation that pushes "signs" to an external hardware display.

## 📚 Documentation
For an exhaustive, line-by-line technical deep dive into the entire codebase, including theoretical context on MediaPipe, Random Forests, and Python concurrency, please refer to:
👉 **[Codebase_Encyclopedia.pdf](./Codebase_Encyclopedia.pdf)**

## 🛠 Project Structure
- `app.py`: Main Flask + SocketIO server & Camera thread.
- `trainmodel.py`: Training script with noise-augmentation logic.
- `collectdata.py`: Data collection utility for training new signs.
- `signbridge.py`: Standalone CLI version for desktop usage.
- `static/`: Frontend assets (Neobrutalist CSS & Vanilla JS).
- `data/`: Landmark datasets and MediaPipe task files.
