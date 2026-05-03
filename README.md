# SignBridge v2: Real-time Sign Language Translation

SignBridge is a high-fidelity sign language interpretation system that utilizes computer vision and machine learning to translate American Sign Language (ASL) into text and speech in real-time. The project bridges the gap between hardware-accelerated perception and reactive web interfaces, providing a complete end-to-end pipeline from data acquisition to standalone kiosk deployment.

## Technical Architecture

The system is built on a four-tier architecture designed for low-latency inference and thread-safe operation.

1. **Perception Layer**: Powered by MediaPipe Hands, extracting 21 high-dimensional 3D landmarks at 30+ frames per second.
2. **Classification Layer**: A Scikit-learn Random Forest ensemble trained on scale-invariant, wrist-centered coordinates.
3. **Application Layer**: A multi-threaded Flask-SocketIO backend that manages independent camera processing and WebSocket communication.
4. **Hardware Layer**: A serial interface with an RPi Pico and OLED display, enabling "reverse translation" and hardware-status visualization.

## Core Engineering Features

### Temporal Stabilization
To eliminate classification jitter, SignBridge implements a sliding-window majority voting algorithm. Confirmation of a sign requires a 75% consensus (6 out of 8 frames) before the state is updated, ensuring high-confidence output even during fast hand transitions.

### Intelligence & UI
* **Prefix-Trie Autocomplete**: A custom data structure optimized for sub-microsecond prefix searches against a 75,000-word dictionary.
* **Neobrutalist Interface**: A high-contrast, bold design system built with vanilla CSS variables, prioritized for readability and accessibility in kiosk environments.
* **Dual-Inference Modes**: Support for both real-time web dashboarding and lightweight CLI-based terminal translation.

## Documentation

For an exhaustive, line-by-line technical deep dive into the entire codebase—including theoretical context on BlazePalm joints, Random Forest impurity, and Python concurrency—please refer to the comprehensive report:

**[Codebase_Encyclopedia.pdf](./Codebase_Encyclopedia.pdf)**

## Project Structure

* `app.py`: Principal server implementation and background camera thread.
* `trainmodel.py`: Machine learning pipeline featuring noise-augmented training routines.
* `collectdata.py`: Interactive utility for capturing Ground Truth landmark datasets.
* `signbridge.py`: Standalone desktop utility with integrated Text-to-Speech (TTS).
* `diagnose.py`: Mathematical verification tool for cross-verifying normalization consistency.
* `static/`: Frontend assets implementing the Neobrutalist design system.
* `data/`: Version-controlled landmark datasets and MediaPipe task models.

## Deployment & Setup

1. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Execution**:
   ```bash
   python3 app.py
   ```
   Access the translation dashboard at `http://localhost:8000`.

3. **Training**:
   Capture new data using `collectdata.py` and execute `trainmodel.py` to regenerate the serialized model.