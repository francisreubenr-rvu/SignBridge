# SignBridge v2: Real-time Sign Language Translation
    2
    3 SignBridge is a high-fidelity sign language interpretation system that utilizes computer vision and machine learning to translate American Sign Language (ASL) into text and speech in real-time. The project bridges the gap between hardware-accelerated perception and reactive
      web interfaces, providing a complete end-to-end pipeline from data acquisition to standalone kiosk deployment.
    4
    5 ## Technical Architecture
    6
    7 The system is built on a four-tier architecture designed for low-latency inference and thread-safe operation.
    8
    9 1. **Perception Layer**: Powered by MediaPipe Hands, extracting 21 high-dimensional 3D landmarks at 30+ frames per second.
   10 2. **Classification Layer**: A Scikit-learn Random Forest ensemble trained on scale-invariant, wrist-centered coordinates.
   11 3. **Application Layer**: A multi-threaded Flask-SocketIO backend that manages independent camera processing and WebSocket communication.
   12 4. **Hardware Layer**: A serial interface with an RPi Pico and OLED display, enabling "reverse translation" and hardware-status visualization.
   13
   14 ## Core Engineering Features
   15
   16 ### Temporal Stabilization
   17 To eliminate classification jitter, SignBridge implements a sliding-window majority voting algorithm. Confirmation of a sign requires a 75% consensus (6 out of 8 frames) before the state is updated, ensuring high-confidence output even during fast hand transitions.
   18
   19 ### Intelligence & UI
   20 * **Prefix-Trie Autocomplete**: A custom data structure optimized for sub-microsecond prefix searches against a 75,000-word dictionary.
   21 * **Neobrutalist Interface**: A high-contrast, bold design system built with vanilla CSS variables, prioritized for readability and accessibility in kiosk environments.
   22 * **Dual-Inference Modes**: Support for both real-time web dashboarding and lightweight CLI-based terminal translation.
   23
   24 ## Documentation
   25
   26 For an exhaustive, line-by-line technical deep dive into the entire codebase—including theoretical context on BlazePalm joints, Random Forest impurity, and Python concurrency—please refer to the comprehensive report:
   27
   28 **[Codebase_Encyclopedia.pdf](./Codebase_Encyclopedia.pdf)**
   29
   30 ## Project Structure
   31
   32 * `app.py`: Principal server implementation and background camera thread.
   33 * `trainmodel.py`: Machine learning pipeline featuring noise-augmented training routines.
   34 * `collectdata.py`: Interactive utility for capturing Ground Truth landmark datasets.
   35 * `signbridge.py`: Standalone desktop utility with integrated Text-to-Speech (TTS).
   36 * `diagnose.py`: Mathematical verification tool for cross-verifying normalization consistency.
   37 * `static/`: Frontend assets implementing the Neobrutalist design system.
   38 * `data/`: Version-controlled landmark datasets and MediaPipe task models.
   39
   40 ## Deployment & Setup
   41
   42 1. **Installation**:
     pip install -r requirements.txt
   1 2. **Execution**:
     python3 app.py

   1    Access the translation dashboard at `http://localhost:8000`.
   2
   3 3. **Training**:
   4    Capture new data using `collectdata.py` and execute `trainmodel.py` to regenerate the serialized model.
