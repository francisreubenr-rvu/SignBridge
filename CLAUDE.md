# Project Instructions

## Tech Stack
- **Language**: Python 3.11+
- **Vision/ML**: MediaPipe (landmark extraction), scikit-learn (Random Forest Classifier), OpenCV
- **Backend**: Flask 3.x, Flask-SocketIO (real-time updates)
- **Hardware**: RPi Pico (OLED display integration via serial)
- **Frontend**: Neobrutalist design with vanilla JS, CSS, and HTML

## Code Style
- Follow PEP 8 for Python code.
- Use `snake_case` for functions, variables, and file names.
- Use `PascalCase` for classes.
- Prefer explicit type hints where beneficial.
- Frontend: Neobrutalist aesthetic (high contrast, bold borders).

## Testing
- No formal test runner configured. Manual verification via `app.py` or `signbridge.py`.

## Build & Run
- **Setup**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```
- **Run Dev Server**: `python3 app.py` (accessible at http://localhost:8000)
- **Train Model**: `python3 trainmodel.py`
- **Collect Data**: `python3 collectdata.py`
- **CLI Mode**: `python3 signbridge.py`

## Project Structure
- `app.py`: Main Flask + SocketIO entry point & detection loop.
- `signbridge.py`: CLI-based translation utility.
- `trainmodel.py`: Script to train the Random Forest model.
- `collectdata.py`: Landmark data collection tool.
- `static/`: Frontend assets (HTML, CSS, JS).
- `data/`: CSV datasets and MediaPipe model files.
- `pico_oled/`: Arduino/C++ code for RPi Pico hardware.

## Conventions
- **Inference**: Confirmed signs require >50% confidence and majority vote (6/8 frames).
- **Temporal Stabilization**: Sliding window deque in background thread.
- **Hardware**: Serial writes to Pico should be non-blocking (wrapped in try/except).
- **Communication**: Use SocketIO for low-latency state synchronization between camera thread and UI.
