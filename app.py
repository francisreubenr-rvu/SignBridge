"""
SignBridge v2 — Real-time Sign Language Translator
Flask + WebSocket backend with temporal stabilization.
"""
import os

import cv2
import mediapipe as mp
import pickle
import serial
import numpy as np
import time
import threading
import json
import re
from collections import deque, Counter

# Optional: Gemini API for sentence completion
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from flask import Flask, Response, send_from_directory, jsonify, request
from flask_socketio import SocketIO

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
import glob

def find_pico_port():
    """Attempt to auto-detect the RPi Pico serial port."""
    # Common patterns for macOS and Linux
    patterns = ["/dev/cu.usbmodem*", "/dev/ttyACM*", "/dev/tty.usbmodem*"]
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            # Prefer the known working port on this machine
            if "/dev/cu.usbmodem21101" in ports:
                return "/dev/cu.usbmodem21101"
            return ports[0]
    return "/dev/cu.usbmodem21101" # Updated fallback

PICO_PORT      = find_pico_port()
BAUD_RATE      = 115200
VOTE_WINDOW    = 8       # sliding window size for majority vote
VOTE_THRESHOLD = 6       # need ≥6/8 matching predictions to confirm
CONF_FLOOR     = 0.50    # ignore predictions below this confidence
PROPOSAL_FLOOR = 0.20    # show "WAIT" + proposal if above this but below CONF_FLOOR
COOLDOWN_SEC   = 1.2     # seconds after confirming before accepting new letter
IDLE_BREAK_SEC = 2.0     # seconds with no hand → auto word break

# ═══════════════════════════════════════════════════════════════════
# AUTO-COMPLETION TRIE
# ═══════════════════════════════════════════════════════════════════
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()
        self.words = []

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        self.words.append(word)

    def _dfs(self, node, prefix, limit, results):
        if len(results) >= limit:
            return
        if node.is_word:
            results.append(prefix)
        for char, child in sorted(node.children.items()):
            self._dfs(child, prefix + char, limit, results)

    def search_prefix(self, prefix, limit=3):
        prefix = prefix.upper()
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
        results = []
        self._dfs(node, prefix, limit, results)
        return results

word_trie = PrefixTrie()
try:
    if os.path.exists("words.txt"):
        with open("words.txt", "r") as f:
            for line in f:
                w = line.strip().upper()
                # Only insert if it's a reasonable word (no garbage, length > 2 for suggestions)
                if w and len(w) > 2 and re.match(r'^[A-Z]+$', w):
                    word_trie.insert(w)
        print(f"📚 Loaded autocomplete dictionary ({len(word_trie.words)} words)")
    else:
        # High quality fallback
        for w in ["HELLO", "NAME", "THANKS", "HELP", "PLEASE", "SORRY", "WELCOME", "GREETINGS"]:
            word_trie.insert(w)
        print("⚠️  words.txt not found. Loaded fallback words.")
except Exception as e:
    print(f"⚠️ Error loading words: {e}")

# ═══════════════════════════════════════════════════════════════════
# CAMERA
# ═══════════════════════════════════════════════════════════════════
def open_camera():
    """Try built-in camera first (index 0), then external (index 1, 2)."""
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Warmup: let auto-exposure settle
                for _ in range(20):
                    cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"📷 Camera opened on index {idx} ({w}x{h})")
                return cap
            cap.release()
    raise RuntimeError("No camera found! Check connection.")



# ═══════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
signs = sorted(model.classes_)
print(f"✅ Model loaded — {len(signs)} signs: {', '.join(signs)}")


# ═══════════════════════════════════════════════════════════════════
# MEDIAPIPE
# ═══════════════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles


# ═══════════════════════════════════════════════════════════════════
# PICO SERIAL (Raspberry Pi Pico — NOT Pico W)
# ═══════════════════════════════════════════════════════════════════
pico = None
try:
    pico = serial.Serial(PICO_PORT, BAUD_RATE, timeout=1)
    print("✅ Pico connected via USB serial")
except Exception:
    print("⚠️  Pico not found — OLED disabled")


def send_to_pico(command):
    """Send a command string to the Pico. Format: COMMAND:VALUE\\n"""
    if pico:
        try:
            pico.write(f"{command}\n".encode())
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# NORMALIZATION (must match trainmodel.py)
# ═══════════════════════════════════════════════════════════════════
def normalize_landmarks(landmarks):
    """Translate to wrist origin, scale to [-1, 1]."""
    bx, by, bz = landmarks[0].x, landmarks[0].y, landmarks[0].z
    norm = []
    for p in landmarks:
        norm.extend([p.x - bx, p.y - by, p.z - bz])
    m = max(abs(v) for v in norm)
    if m > 0:
        norm = [v / m for v in norm]
    return norm


# ═══════════════════════════════════════════════════════════════════
# SHARED STATE
# ═══════════════════════════════════════════════════════════════════
class AppState:
    """Thread-safe shared state between camera thread and web server."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frame_bytes = None       # latest JPEG for MJPEG stream

        # Detection state
        self.hand_detected = False
        self.raw_prediction = ""
        self.raw_confidence = 0.0
        self.proposal = ""            # Suggested letter if confidence is low

        # Confirmed state
        self.confirmed_letter = ""
        self.cooldown_remaining = 0.0

        # Word / sentence builder
        self.current_letters = []     # letters of current word
        self.sentence_words = []      # completed words
        self.current_word = ""
        self.sentence = ""

        # FSM state: idle / detecting / locked / waiting
        self.fsm_state = "idle"

    def update_detection(self, hand_detected, raw_pred, raw_conf, fsm_state, cooldown, proposal=""):
        with self._lock:
            self.hand_detected = hand_detected
            self.raw_prediction = raw_pred
            self.raw_confidence = raw_conf
            self.fsm_state = fsm_state
            self.cooldown_remaining = cooldown
            self.proposal = proposal

    def confirm_letter(self, letter):
        with self._lock:
            self.confirmed_letter = letter
            self.current_letters.append(letter)
            self.current_word = "".join(self.current_letters)

    def finalize_word(self):
        """Finalize current word and add to sentence."""
        with self._lock:
            if self.current_letters:
                word = "".join(self.current_letters)
                self.sentence_words.append(word)
                self.sentence = " ".join(self.sentence_words)
                self.current_letters = []
                self.current_word = ""
                return word
            return None

    def backspace(self):
        with self._lock:
            if self.current_letters:
                self.current_letters.pop()
                self.current_word = "".join(self.current_letters)

    def clear_sentence(self):
        with self._lock:
            self.sentence_words = []
            self.sentence = ""

    def clear_all(self):
        with self._lock:
            self.current_letters = []
            self.current_word = ""
            self.sentence_words = []
            self.sentence = ""
            self.confirmed_letter = ""

    def get_snapshot(self):
        with self._lock:
            return {
                "hand_detected": self.hand_detected,
                "raw_prediction": self.raw_prediction,
                "raw_confidence": self.raw_confidence,
                "proposal": self.proposal,
                "confirmed_letter": self.confirmed_letter,
                "cooldown_remaining": round(self.cooldown_remaining, 1),
                "current_word": self.current_word,
                "letters": list(self.current_letters),
                "sentence": self.sentence,
                "sentence_words": list(self.sentence_words),
                "fsm_state": self.fsm_state,
            }

    def set_frame(self, jpeg_bytes):
        with self._lock:
            self.frame_bytes = jpeg_bytes

    def get_frame(self):
        with self._lock:
            return self.frame_bytes


state = AppState()


# ═══════════════════════════════════════════════════════════════════
# CAMERA THREAD
# ═══════════════════════════════════════════════════════════════════
def camera_loop(socketio_ref):
    """Main detection loop. Runs in a background thread."""
    try:
        cap = open_camera()
    except Exception as e:
        print(f"❌ Camera Thread Error: {e}")
        return

    # Use False for real-time video stream (better performance and temporal tracking)
    hands_live = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    vote_buffer = deque(maxlen=VOTE_WINDOW)
    last_confirm_time = 0.0
    last_confirmed = ""
    last_hand_time = time.time()
    word_break_sent = False
    frame_count = 0
    hand_count = 0
    last_log = time.time()

    print("🚀 Camera loop started")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        now = time.time()

        # ── Step 1: MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_live.process(rgb)

        hand_detected = False
        raw_pred = ""
        raw_conf = 0.0
        proposal = ""

        if result.multi_hand_landmarks:
            hand_detected = True
            hand_count += 1
            last_hand_time = now
            word_break_sent = False

            lm = result.multi_hand_landmarks[0].landmark
            norm = normalize_landmarks(lm)
            features = np.array([norm])

            proba = model.predict_proba(features)[0]
            max_idx = proba.argmax()
            raw_conf = float(proba[max_idx])
            raw_pred = str(model.classes_[max_idx])

            # Proposal logic
            if raw_conf < CONF_FLOOR and raw_conf >= PROPOSAL_FLOOR:
                proposal = raw_pred

            # Only add to vote buffer if above confidence floor
            if raw_conf >= CONF_FLOOR:
                vote_buffer.append(raw_pred)
            else:
                vote_buffer.append(None)

            # ── Draw landmarks on original frame ──
            mp_draw.draw_landmarks(
                frame,
                result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style(),
            )
        else:
            vote_buffer.append(None)

            # ── Auto word break: no hand for IDLE_BREAK_SEC ──
            if (now - last_hand_time > IDLE_BREAK_SEC
                    and not word_break_sent
                    and state.current_letters):
                word = state.finalize_word()
                if word:
                    send_to_pico(f"WORD:{word}")
                    send_to_pico(f"SENTENCE:{state.sentence}")
                    word_break_sent = True
                    last_confirmed = ""
                    vote_buffer.clear()
                    try:
                        socketio_ref.emit("state_update", state.get_snapshot())
                    except Exception:
                        pass

        # ── Step 2: Majority vote ──
        cooldown_remaining = max(0, COOLDOWN_SEC - (now - last_confirm_time))
        in_cooldown = cooldown_remaining > 0

        # Determine FSM state
        if not hand_detected:
            fsm = "idle"
        elif in_cooldown:
            fsm = "locked"
        elif raw_conf < CONF_FLOOR:
            fsm = "waiting"
        else:
            fsm = "detecting"

        # Check for letter confirmation
        if hand_detected and not in_cooldown and len(vote_buffer) >= VOTE_THRESHOLD:
            valid_votes = [v for v in vote_buffer if v is not None]
            if len(valid_votes) >= VOTE_THRESHOLD:
                counter = Counter(valid_votes)
                top_sign, top_count = counter.most_common(1)[0]
                if top_count >= VOTE_THRESHOLD and top_sign != last_confirmed:
                    # ✅ CONFIRMED
                    state.confirm_letter(top_sign)
                    last_confirmed = top_sign
                    last_confirm_time = now
                    vote_buffer.clear()
                    fsm = "locked"
                    cooldown_remaining = COOLDOWN_SEC

                    send_to_pico(f"LETTER:{top_sign}")
                    send_to_pico(f"WORD:{state.current_word}")
                    print(f"  ✅ [{top_sign}] confirmed → \"{state.current_word}\"")

                    try:
                        socketio_ref.emit("letter_confirmed", {
                            "letter": top_sign,
                            "word": state.current_word,
                        })
                    except Exception:
                        pass
        
        # Reset last_confirmed if hand missing for 0.5s (allows double letters like "EE")
        if not hand_detected and (now - last_hand_time) > 0.5:
            last_confirmed = ""

        # ── Step 3: Update shared state ──
        state.update_detection(hand_detected, raw_pred, raw_conf, fsm, cooldown_remaining, proposal)

        # ── Step 4: Flip AFTER drawing → display frame ──
        display = cv2.flip(frame, 1)

        # ── Step 5: Encode to JPEG ──
        ret_enc, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret_enc:
            state.set_frame(jpeg.tobytes())
            if frame_count == 1:
                print("🖼️  First frame encoded and stored")
        else:
            if frame_count % 100 == 0:
                print("⚠️  Frame encoding failed")

        # ── Step 6: Emit state via WebSocket ──
        if frame_count % 3 == 0:  # every 3rd frame (~5 updates/sec at 15fps)
            try:
                socketio_ref.emit("state_update", state.get_snapshot())
            except Exception:
                pass

        # ── Terminal logging (every 3s) ──
        if now - last_log >= 3.0:
            det_pct = (hand_count / frame_count * 100) if frame_count else 0
            print(f"  📊 F:{frame_count} Hand:{hand_count}({det_pct:.0f}%) "
                  f"Raw:{raw_pred}({raw_conf*100:.0f}%) "
                  f"Word:\"{state.current_word}\" [{fsm}]")
            last_log = now

        # Throttle to ~15fps
        time.sleep(0.066)


# ═══════════════════════════════════════════════════════════════════
# FLASK APP
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder="static")
app.config["SECRET_KEY"] = "signbridge-v2"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


# ── Static routes ──
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


# ── MJPEG stream ──
def generate_mjpeg():
    while True:
        frame = state.get_frame()
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── REST API ──
@app.route("/state")
def get_state():
    return jsonify(state.get_snapshot())


@app.route("/action/space", methods=["POST"])
def action_space():
    word = state.finalize_word()
    if word:
        send_to_pico(f"WORD:{word}")
        send_to_pico(f"SENTENCE:{state.sentence}")
    socketio.emit("state_update", state.get_snapshot())
    return jsonify({"ok": True, "word": word})


@app.route("/action/select_word", methods=["POST"])
def action_select_word():
    data = request.json or {}
    selected_word = data.get("word", "").upper()
    if not selected_word:
        return jsonify({"ok": False})
    
    with state._lock:
        state.current_letters = []
        state.current_word = ""
        state.sentence_words.append(selected_word)
        state.sentence = " ".join(state.sentence_words)
    
    send_to_pico(f"WORD:{selected_word}")
    send_to_pico(f"SENTENCE:{state.sentence}")
    socketio.emit("state_update", state.get_snapshot())
    return jsonify({"ok": True, "word": selected_word})


@app.route("/action/backspace", methods=["POST"])
def action_backspace():
    state.backspace()
    socketio.emit("state_update", state.get_snapshot())
    return jsonify({"ok": True})


@app.route("/action/clear", methods=["POST"])
def action_clear():
    state.clear_sentence()
    send_to_pico("CLEAR")
    socketio.emit("state_update", state.get_snapshot())
    return jsonify({"ok": True})


@app.route("/action/speak", methods=["POST"])
def action_speak():
    # Priority: Full sentence, then current word
    snap = state.get_snapshot()
    text = snap["sentence"] or snap["current_word"]
    return jsonify({"ok": True, "text": text})


@app.route("/api/suggest")
def api_suggest():
    """Return top word completions for the currently building word."""
    prefix = request.args.get("q", "")
    if len(prefix) < 1:
        return jsonify({"suggestions": []})
    suggestions = word_trie.search_prefix(prefix, limit=3)
    return jsonify({"suggestions": suggestions})


@app.route("/api/sentence_suggest")
def api_sentence_suggest():
    """Return a sentence completion prediction based on current words."""
    sentence = request.args.get("q", "").strip()
    if not sentence:
        return jsonify({"suggestion": ""})
        
    # Basic mock sentence completer if no LLM is configured
    # In a real setup with Gemini:
    # model = genai.GenerativeModel('gemini-pro')
    # response = model.generate_content(f"Complete this sentence naturally: {sentence}")
    # return jsonify({"suggestion": response.text})
    
    parts = sentence.upper().split()
    suggestion = ""
    if parts[-1] == "I" or parts[-1] == "WE":
        suggestion = "ARE GOING TO"
    elif parts[-1] == "HELLO":
        suggestion = "HOW ARE YOU"
    elif parts[-1] == "WHAT":
        suggestion = "IS YOUR NAME"
    elif parts[-1] == "THANK":
        suggestion = "YOU VERY MUCH"
    elif "MY NAME" in sentence.upper():
        suggestion = "IS"
        
    return jsonify({"suggestion": suggestion})


@app.route("/action/reverse_translate", methods=["POST"])
def action_reverse_translate():
    """Phase 3: Text to Sign OLED player"""
    data = request.json or {}
    text = data.get("text", "").upper()
    if not text:
        return jsonify({"ok": False})
        
    # Send the raw text to Pico, prepended by PLAY_SIGNS
    # The Pico will scroll through each letter in large text
    send_to_pico(f"PLAY_SIGNS:{text}")
    return jsonify({"ok": True})


# ── WebSocket events ──
@socketio.on("connect")
def ws_connect():
    socketio.emit("state_update", state.get_snapshot())


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Start camera thread
    cam_thread = threading.Thread(target=camera_loop, args=(socketio,), daemon=True)
    cam_thread.start()

    print("\n🤟 SignBridge v2 running at http://localhost:8000")
    print("   Press Ctrl+C to stop.\n")

    socketio.run(app, host="0.0.0.0", port=8000, debug=False, allow_unsafe_werkzeug=True)
