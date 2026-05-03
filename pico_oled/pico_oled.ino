/*
 * SignBridge — Raspberry Pi Pico OLED Display (Optimized v3)
 * ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 * Board:   Raspberry Pi Pico (RP2040)
 * OLED:    SSD1306 128x64 I2C (GP4=SDA, GP5=SCL)
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
#define OLED_ADDR     0x3C
#define SDA_PIN       4
#define SCL_PIN       5

// Serial Buffer
#define MAX_CMD_LEN 64
char serialBuf[MAX_CMD_LEN];
int bufIdx = 0;

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// State
String currentLetter   = "";
String currentWord     = "";
String currentSentence = "";
unsigned long lastSerialTime = 0;
bool isConnected = false;
bool needsRedraw = true;

// Scrolling Word State
int16_t scrollX = 0;
uint16_t wordWidth = 0;
bool isScrolling = false;
unsigned long lastScrollTime = 0;
#define SCROLL_SPEED 20 // ms per pixel shift

void bootAnimation() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(2);
    display.setCursor(10, 10);
    display.print("SIGN");
    display.setCursor(10, 32);
    display.print("BRIDGE");
    display.drawLine(0, 55, 128, 55, SSD1306_WHITE);
    display.display();
    delay(500);
}
// Global dummy for getTextBounds
uint16_t h; 
void updateScrollState() {
    int16_t x1, y1;
    display.setTextSize(3);
    display.getTextBounds(currentWord, 0, 0, &x1, &y1, &wordWidth, &h); // h is dummy
    
    if (wordWidth > SCREEN_WIDTH) {
        isScrolling = true;
    } else {
        isScrolling = false;
        scrollX = (SCREEN_WIDTH - wordWidth) / 2;
    }
}



void drawDisplay() {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);

    // 1. Current Letter (Large, Top Left)
    if (currentLetter.length() > 0) {
        display.setTextSize(2);
        display.setCursor(2, 2);
        display.print(currentLetter);
    }
    
    // 2. Status Indicator
    if (isConnected) {
        display.fillRect(120, 4, 4, 4, SSD1306_WHITE);
    }
    
    // 3. Current Word (Main Focus)
    if (currentWord.length() > 0) {
        display.setTextSize(3);
        display.setCursor(scrollX, 22);
        display.print(currentWord);
    }
    
    // 4. Sentence (Bottom)
    if (currentSentence.length() > 0) {
        display.setTextSize(1);
        display.setCursor(0, 56);
        String toDraw = currentSentence;
        if (toDraw.length() > 21) {
            toDraw = toDraw.substring(toDraw.length() - 21);
        }
        display.print(toDraw);
    }
    
    display.display();
    needsRedraw = false;
}

void playSigns(String text) {
    for (int i = 0; i < text.length(); i++) {
        char c = text.charAt(i);
        display.clearDisplay();
        if (c != ' ') {
            display.drawRect(0, 0, 128, 64, SSD1306_WHITE);
            display.setTextSize(5);
            String s = String(c);
            int16_t x1, y1;
            uint16_t w, h2;
            display.getTextBounds(s, 0, 0, &x1, &y1, &w, &h2);
            display.setCursor((128-w)/2, 14);
            display.print(s);
        }
        display.display();
        delay(600);
    }
    needsRedraw = true;
}

void handleCommand(char* cmd) {
    lastSerialTime = millis();
    isConnected = true;

    String s = String(cmd);
    s.trim();
    if (s == "CLEAR") {
        currentLetter = ""; currentWord = ""; currentSentence = "";
        isScrolling = false; needsRedraw = true;
        return;
    }

    int colon = s.indexOf(':');
    if (colon == -1) return;

    String type = s.substring(0, colon);
    String val = s.substring(colon + 1);

    if (type == "LETTER") {
        currentLetter = val;
        needsRedraw = true;
    } else if (type == "WORD") {
        currentWord = val;
        int16_t x1, y1;
        display.setTextSize(3);
        display.getTextBounds(currentWord, 0, 0, &x1, &y1, &wordWidth, &h);
        if (wordWidth > SCREEN_WIDTH) {
            isScrolling = true;
            scrollX = 0;
        } else {
            isScrolling = false;
            scrollX = (SCREEN_WIDTH - wordWidth) / 2;
        }
        needsRedraw = true;
    } else if (type == "SENTENCE") {
        currentSentence = val;
        needsRedraw = true;
    } else if (type == "PLAY_SIGNS") {
        playSigns(val);
    }
}

void setup() {
    Serial.begin(115200);
    Wire.setSDA(SDA_PIN);
    Wire.setSCL(SCL_PIN);
    Wire.begin();
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR)) {
        for(;;);
    }
    
    bootAnimation();
    display.clearDisplay();
    display.display();
}

void loop() {
    // Non-blocking Serial
    while (Serial.available() > 0) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            serialBuf[bufIdx] = '\0';
            if (bufIdx > 0) handleCommand(serialBuf);
            bufIdx = 0;
        } else if (bufIdx < MAX_CMD_LEN - 1) {
            serialBuf[bufIdx++] = c;
        }
    }

    // Scrolling logic
    if (isScrolling && (millis() - lastScrollTime > SCROLL_SPEED)) {
        scrollX -= 2;
        if (scrollX < -(int16_t)wordWidth) scrollX = SCREEN_WIDTH;
        lastScrollTime = millis();
        needsRedraw = true;
    }

    // Heartbeat
    if (isConnected && (millis() - lastSerialTime > 10000)) {
        isConnected = false;
        currentLetter = "OFF";
        needsRedraw = true;
    }

    if (needsRedraw) {
        drawDisplay();
    }
}
