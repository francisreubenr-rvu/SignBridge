/**
 * SignBridge v2 — Client-side logic
 * WebSocket state management, animations, TTS
 */

// ═══════════════════════════════════════════════════════════════
// DOM ELEMENTS
// ═══════════════════════════════════════════════════════════════
const $ = (sel) => document.querySelector(sel);
const els = {
    connectionStatus: $("#connection-status"),
    handIndicator:    $("#hand-indicator"),
    indicatorText:    $(".indicator-text"),
    fsmBadge:         $("#fsm-badge"),
    signLetter:       $("#sign-letter"),
    ringFill:         $("#ring-fill"),
    ringLabel:        $("#ring-label"),
    cooldownBar:      $("#cooldown-bar"),
    cooldownFill:     $("#cooldown-fill"),
    wordSuggestions:  $("#word-suggestions"),
    letterTiles:      $("#letter-tiles"),
    sentenceText:     $("#sentence-text"),
    sentenceGhost:    $("#sentence-ghost"),
    sentencePlaceholder: $("#sentence-placeholder"),
    debugPanel:       $("#debug-panel"),
    dHand:            $("#d-hand"),
    dRaw:             $("#d-raw"),
    dConf:            $("#d-conf"),
    dFsm:             $("#d-fsm"),
    dCooldown:        $("#d-cooldown"),
    
    // Reverse Translation Mode elements
    modeDetect:       $("#mode-detect"),
    modeReverse:      $("#mode-reverse"),
    detectSectionLeft:$("#detect-section-left"),
    translationSection:$(".translation-section"),
    reverseSection:   $("#reverse-section"),
    reverseInput:     $("#reverse-input"),
    btnMic:           $("#btn-mic"),
    btnPlaySigns:     $("#btn-play-signs"),
    videoFeed:        $("#video-feed"),
};

// Confidence ring geometry
const RING_CIRCUMFERENCE = 2 * Math.PI * 44; // r=44 in SVG

if (els.videoFeed) {
    els.videoFeed.onerror = () => {
        console.error("📹 Video feed failed to load. Check backend logs.");
        els.indicatorText.textContent = "Camera Stream Error";
        els.handIndicator.style.background = "var(--accent-alert)";
    };
}

// ═══════════════════════════════════════════════════════════════
// WEBSOCKET CONNECTION
// ═══════════════════════════════════════════════════════════════
let socket;
let fallbackInterval;

function initSocket() {
    socket = io({ transports: ["websocket", "polling"] });

    socket.on("connect", () => {
        els.connectionStatus.classList.add("connected");
        els.connectionStatus.querySelector("span:last-child").textContent = "Live";
        console.log("🟢 WebSocket connected");
        // Clear fallback polling if running
        if (fallbackInterval) {
            clearInterval(fallbackInterval);
            fallbackInterval = null;
        }
    });

    socket.on("disconnect", () => {
        els.connectionStatus.classList.remove("connected");
        els.connectionStatus.querySelector("span:last-child").textContent = "Reconnecting...";
        console.log("🔴 WebSocket disconnected");
        // Start fallback polling
        startFallbackPolling();
    });

    socket.on("state_update", (data) => {
        updateUI(data);
    });

    socket.on("letter_confirmed", (data) => {
        onLetterConfirmed(data.letter, data.word);
    });
}

function startFallbackPolling() {
    if (fallbackInterval) return;
    fallbackInterval = setInterval(() => {
        fetch("/state")
            .then(r => r.json())
            .then(data => updateUI(data))
            .catch(() => {});
    }, 300);
}

// ═══════════════════════════════════════════════════════════════
// UI UPDATE
// ═══════════════════════════════════════════════════════════════
let lastLetterCount = 0;
let lastSentence = "";

function updateUI(data) {
    // ── Hand indicator ──
    if (data.hand_detected) {
        els.handIndicator.classList.add("active");
        els.indicatorText.textContent = "Hand Detected";
    } else {
        els.handIndicator.classList.remove("active");
        els.indicatorText.textContent = "No Hand";
    }

    // ── FSM badge ──
    const fsm = data.fsm_state || "idle";
    els.fsmBadge.textContent = fsm.toUpperCase();
    els.fsmBadge.className = "fsm-badge " + fsm;

    // ── Sign letter + confidence ring ──
    const rawPred = data.raw_prediction || "—";
    const rawConf = data.raw_confidence || 0;
    const proposal = data.proposal || "";

    if (fsm === "waiting") {
        els.signLetter.textContent = proposal ? `WAIT... (${proposal}?)` : "WAITING...";
        els.signLetter.classList.add("waiting");
        els.signLetter.classList.remove("multi");
    } else if (data.hand_detected && rawPred !== "—") {
        els.signLetter.textContent = rawPred;
        els.signLetter.classList.remove("confirmed", "waiting");
        els.signLetter.classList.toggle("multi", rawPred.length > 1);
    } else if (!data.hand_detected) {
        if (!data.confirmed_letter) {
            els.signLetter.textContent = "—";
            els.signLetter.classList.remove("multi", "waiting");
        }
    }

    // Confidence ring
    const confPct = Math.round(rawConf * 100);
    const offset = RING_CIRCUMFERENCE * (1 - rawConf);
    els.ringFill.style.strokeDashoffset = offset;
    els.ringLabel.textContent = confPct + "%";

    // Ring color based on confidence
    if (rawConf >= 0.7) {
        els.ringFill.style.stroke = "var(--conf-high)";
    } else if (rawConf >= 0.5) {
        els.ringFill.style.stroke = "var(--conf-med)";
    } else {
        els.ringFill.style.stroke = "var(--conf-low)";
    }

    // ── Cooldown bar ──
    if (data.cooldown_remaining > 0) {
        els.cooldownBar.classList.add("active");
        const pct = (data.cooldown_remaining / 1.5) * 100;
        els.cooldownFill.style.width = pct + "%";
    } else {
        els.cooldownBar.classList.remove("active");
        els.cooldownFill.style.width = "0%";
    }

    // ── Letter tiles & Suggestions ──
    const letters = data.letters || [];
    if (letters.length !== lastLetterCount) {
        renderLetterTiles(letters);
        lastLetterCount = letters.length;
        fetchWordSuggestions(data.current_word);
    }

    // ── Sentence ──
    const sentence = data.sentence || "";
    if (sentence !== lastSentence) {
        if (sentence) {
            if (els.sentencePlaceholder) els.sentencePlaceholder.style.display = "none";
            if (els.sentenceText) {
                els.sentenceText.innerHTML = "";
                // Render each word as a span
                sentence.split(" ").forEach((word, i) => {
                    if (i > 0) els.sentenceText.appendChild(document.createTextNode(" "));
                    const span = document.createElement("span");
                    span.className = "sentence-word";
                    span.textContent = word;
                    els.sentenceText.appendChild(span);
                });
            }
            fetchSentenceSuggestion(sentence);
        } else {
            if (els.sentencePlaceholder) els.sentencePlaceholder.style.display = "inline";
            if (els.sentenceText) els.sentenceText.innerHTML = "";
            if (els.sentenceGhost) els.sentenceGhost.textContent = "";
        }
        lastSentence = sentence;
    }

    // ── Debug panel ──
    els.dHand.textContent = data.hand_detected ? "✅ YES" : "❌ NO";
    els.dHand.style.color = data.hand_detected ? "var(--conf-high)" : "var(--accent-alert)";
    els.dRaw.textContent = rawPred;
    els.dConf.textContent = confPct + "%";
    els.dFsm.textContent = fsm;
    els.dCooldown.textContent = data.cooldown_remaining > 0 ? data.cooldown_remaining + "s" : "—";
}

function renderLetterTiles(letters) {
    if (letters.length === 0) {
        els.letterTiles.innerHTML = '<span class="tile-placeholder">Show a sign to begin...</span>';
        return;
    }
    els.letterTiles.innerHTML = "";
    letters.forEach((letter, i) => {
        const tile = document.createElement("div");
        tile.className = "letter-tile";
        tile.textContent = letter;
        // Only animate the newest tile
        if (i === letters.length - 1) {
            tile.style.animationDelay = "0s";
        } else {
            tile.style.animation = "none";
        }
        els.letterTiles.appendChild(tile);
    });
}

function onLetterConfirmed(letter, word) {
    // Cobalt Blue high-impact flash
    document.body.style.transition = "none";
    document.body.style.backgroundColor = "var(--accent-blue)";
    
    // Slight shake to the sign display
    els.signLetter.parentElement.style.transform = "scale(1.05) rotate(1deg)";
    
    setTimeout(() => {
        document.body.style.transition = "background-color 0.3s";
        document.body.style.backgroundColor = "var(--bg)";
        els.signLetter.parentElement.style.transform = "none";
    }, 100);

    // Update letter display
    els.signLetter.textContent = letter;
    els.signLetter.classList.add("confirmed");
    setTimeout(() => els.signLetter.classList.remove("confirmed"), 500);
}

// ═══════════════════════════════════════════════════════════════
// TEXT-TO-SPEECH (Browser Web Speech API)
// ═══════════════════════════════════════════════════════════════
let systemVoices = [];
function loadVoices() {
    systemVoices = window.speechSynthesis.getVoices();
}
if ('speechSynthesis' in window) {
    loadVoices();
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = loadVoices;
    }
}

function speak(text) {
    if (!text || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    
    const normalizedText = text.toLowerCase();
    const utterance = new SpeechSynthesisUtterance(normalizedText);
    
    // 1. Target Indian Female (macOS 'Veena', 'Lekha' or Chrome 'Female')
    let selectedVoice = systemVoices.find(v => 
        (v.lang.includes('en-IN') || v.lang.includes('hi-IN')) && 
        (v.name.includes('Veena') || v.name.includes('Lekha') || v.name.toLowerCase().includes('female'))
    );
    
    // 2. If no Indian female voice is installed, strictly target ANY known female voice 
    // (macOS standard female voices: Samantha, Victoria, Karen, Moira, Tessa)
    if (!selectedVoice) {
        selectedVoice = systemVoices.find(v => 
            v.name.includes('Samantha') || 
            v.name.includes('Victoria') || 
            v.name.includes('Karen') ||
            v.name.includes('Tessa') ||
            v.name.toLowerCase().includes('female')
        );
    }
    
    if (selectedVoice) {
        utterance.voice = selectedVoice;
    }
    
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
}

// ═══════════════════════════════════════════════════════════════
// ACTIONS
// ═══════════════════════════════════════════════════════════════
function postAction(path) {
    return fetch(path, { method: "POST" })
        .then(r => r.json())
        .catch(() => ({}));
}

// Backspace
$("#btn-backspace").addEventListener("click", () => {
    postAction("/action/backspace");
});

// Next Word (space)
$("#btn-space").addEventListener("click", () => {
    postAction("/action/space").then(data => {
        if (data.word) {
            speak(data.word);
        }
    });
});

// Speak sentence
$("#btn-speak").addEventListener("click", () => {
    postAction("/action/speak").then(data => {
        if (data.text) speak(data.text);
    });
});

$("#btn-speak-sentence").addEventListener("click", () => {
    postAction("/action/speak").then(data => {
        if (data.text) speak(data.text);
    });
});

// Clear
$("#btn-clear").addEventListener("click", () => {
    postAction("/action/clear");
    lastLetterCount = 0;
    lastSentence = "";
});

// Debug toggle
$("#btn-debug-toggle").addEventListener("click", () => {
    els.debugPanel.classList.toggle("hidden");
});

// Keyboard shortcuts
window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    switch (e.code) {
        case "Space":
            e.preventDefault();
            $("#btn-space").click();
            break;
        case "Backspace":
            e.preventDefault();
            $("#btn-backspace").click();
            break;
        case "KeyD":
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                $("#btn-debug-toggle").click();
            }
            break;
        case "Escape":
            $("#btn-clear").click();
            break;
    }
});

// ═══════════════════════════════════════════════════════════════
// AUTO-COMPLETION FETCHERS
// ═══════════════════════════════════════════════════════════════
function fetchWordSuggestions(prefix) {
    if (!prefix) {
        els.wordSuggestions.innerHTML = "";
        return;
    }
    fetch(`/api/suggest?q=${encodeURIComponent(prefix)}`)
        .then(r => r.json())
        .then(data => {
            els.wordSuggestions.innerHTML = "";
            (data.suggestions || []).forEach(word => {
                const pill = document.createElement("button");
                pill.className = "suggestion-pill";
                pill.textContent = word;
                pill.onclick = () => selectWord(word);
                els.wordSuggestions.appendChild(pill);
            });
        }).catch(console.error);
}

function fetchSentenceSuggestion(sentence) {
    if (!sentence) {
        els.sentenceGhost.textContent = "";
        return;
    }
    fetch(`/api/sentence_suggest?q=${encodeURIComponent(sentence)}`)
        .then(r => r.json())
        .then(data => {
            if (data.suggestion) {
                els.sentenceGhost.textContent = " " + data.suggestion;
            } else {
                els.sentenceGhost.textContent = "";
            }
        }).catch(console.error);
}

function selectWord(word) {
    fetch("/action/select_word", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: word })
    }).then(r => r.json()).then(data => {
        if (data.word) speak(data.word);
    });
}

// ═══════════════════════════════════════════════════════════════
// MODE TOGGLE & REVERSE TRANSLATION
// ═══════════════════════════════════════════════════════════════
els.modeDetect.addEventListener("click", () => {
    els.modeDetect.classList.add("active");
    els.modeReverse.classList.remove("active");
    els.detectSectionLeft.classList.remove("hidden");
    els.translationSection.classList.remove("hidden");
    els.reverseSection.classList.add("hidden");
});

els.modeReverse.addEventListener("click", () => {
    els.modeReverse.classList.add("active");
    els.modeDetect.classList.remove("active");
    els.detectSectionLeft.classList.add("hidden");
    els.translationSection.classList.add("hidden");
    els.reverseSection.classList.remove("hidden");
});

// Play signs on OLED
els.btnPlaySigns.addEventListener("click", () => {
    const text = els.reverseInput.value.trim();
    if (!text) return;
    
    els.btnPlaySigns.disabled = true;
    els.btnPlaySigns.innerHTML = `<span class="pulse-dot"></span> Playing...`;
    
    fetch("/action/reverse_translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text })
    }).then(() => {
        setTimeout(() => {
            els.btnPlaySigns.disabled = false;
            els.btnPlaySigns.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Play on OLED Screen`;
        }, 1500);
    });
});

// Web Speech API for Dictation
let recognition = null;
if ('webkitSpeechRecognition' in window) {
    recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        els.btnMic.classList.add("listening");
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        els.reverseInput.value = (els.reverseInput.value + " " + transcript).trim();
    };
    
    recognition.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        els.btnMic.classList.remove("listening");
    };
    
    recognition.onend = () => {
        els.btnMic.classList.remove("listening");
    };
} else {
    els.btnMic.style.display = "none";
}

els.btnMic.addEventListener("click", () => {
    if (!recognition) return;
    if (els.btnMic.classList.contains("listening")) {
        recognition.stop();
    } else {
        recognition.start();
    }
});

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
initSocket();
// Also start fallback polling immediately in case WebSocket is slow
startFallbackPolling();
