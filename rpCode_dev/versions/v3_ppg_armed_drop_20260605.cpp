#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "pico/time.h"
#include <Wire.h>
#include "MAX30105.h"

#define ECG_PIN        A0
#define MA_SIZE        25       // 50ms window at 500Hz
#define PT_WINDOW      75       // 150ms window at 500Hz
#define REFRACTORY_MS  200
#define BLOCK_MS       500

#define BEATS_TO_AVERAGE  8
#define PTT_MIN_MS      100.0f
#define PTT_MAX_MS      450.0f

MAX30105 ppgSensor;
repeating_timer_t ecgTimer;

bool sleeping  = true;
bool streaming = false;

// ── ECG state ─────────────────────────────────────────────────────────────────
float maBuffer[MA_SIZE] = {0};
int   maIndex = 0;
float maSum   = 0;

float ptBuffer[PT_WINDOW] = {0};
int   ptIndex  = 0;
float ptMWI    = 0;
float ptPeak   = 500.0f;
bool  ptRefrac = false;
uint32_t lastR_ms = 0;

volatile float    latestECG    = 0;
volatile bool     latestRmark  = false;
volatile uint32_t rWaveTime_us = 0;
volatile float    latestThresh = 0;
volatile int      ecgStartupSkip = 250;  // skip first 500ms (250 × 2ms) after wake

// ── PPG state ─────────────────────────────────────────────────────────────────
float    ppgDC        = 0;
float    ppgPrev      = 0;
float    ppgPrevDeriv = 0;
bool     dcSettled    = false;
uint32_t fingerTime   = 0;
bool     armed        = false;
float    peakAC       = 0;
uint32_t lastFire_ms  = 0;
float    printPPG     = 0;
bool     printFire    = false;

// ── PTT collection ────────────────────────────────────────────────────────────
float pttSamples[BEATS_TO_AVERAGE];
int   pttCount       = 0;
bool  skipNextSample = true;
float medianPTT      = 0;
bool  newPTT         = false;

// ── DSP helpers ───────────────────────────────────────────────────────────────

float movingAverage(float v) {
  maSum -= maBuffer[maIndex];
  maBuffer[maIndex] = v;
  maSum += v;
  maIndex = (maIndex + 1) % MA_SIZE;
  return maSum / MA_SIZE;
}

float panTompkins(float raw) {
  static float x1 = 0, x2 = 0;
  float deriv = (raw - x2) * 0.5f;
  x2 = x1; x1 = raw;
  float sq = deriv * deriv;
  ptMWI -= ptBuffer[ptIndex];
  ptBuffer[ptIndex] = sq;
  ptMWI += sq;
  ptIndex = (ptIndex + 1) % PT_WINDOW;
  return ptMWI;
}

// ── ECG interrupt — 500Hz ─────────────────────────────────────────────────────

bool ecgCallback(repeating_timer_t *rt) {
  if (ecgStartupSkip > 0) { ecgStartupSkip--; return true; }

  uint32_t now_ms = millis();
  uint32_t now_us = time_us_32();
  float raw    = (float)analogRead(ECG_PIN);
  float smooth = movingAverage(raw);
  float mwi    = panTompkins(smooth);

  latestECG = smooth;

  if (mwi > ptPeak) ptPeak = mwi;
  ptPeak      *= 0.9998f;
  float thresh  = ptPeak * 0.75f;
  latestThresh  = thresh;

  if (ptRefrac && (now_ms - lastR_ms) > REFRACTORY_MS)
    ptRefrac = false;

  if (!ptRefrac && mwi > thresh && thresh > 50.0f) {
    lastR_ms     = now_ms;
    rWaveTime_us = now_us;
    ptRefrac     = true;
    ptPeak      *= 0.6f;
    latestRmark  = true;
  }

  return true;
}

// ── Sensor power management ───────────────────────────────────────────────────

void resetState() {
  memset(maBuffer, 0, sizeof(maBuffer));
  maSum = 0; maIndex = 0;
  memset(ptBuffer, 0, sizeof(ptBuffer));
  ptMWI = 0; ptIndex = 0;
  ptPeak = 500.0f; ptRefrac = false; lastR_ms = 0;
  latestECG = 0; latestRmark = false; rWaveTime_us = 0; latestThresh = 0;
  ecgStartupSkip = 250;
  ppgDC = ppgPrev = ppgPrevDeriv = 0;
  dcSettled = false; armed = false; peakAC = 0;
  lastFire_ms = 0; fingerTime = 0;
  printPPG = 0; printFire = false;
  pttCount = 0; skipNextSample = true;
  medianPTT = 0; newPTT = false;
}

void sleepSensors() {
  cancel_repeating_timer(&ecgTimer);
  ppgSensor.shutDown();
  resetState();
  sleeping = true;
  Serial.println("STATUS:SLEEPING");
}

void wakeSensors() {
  ppgSensor.wakeUp();
  // sampleAverage=1 → 100Hz effective PPG (reduces timing quantization)
  ppgSensor.setup(60, 1, 2, 100, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);
  resetState();
  add_repeating_timer_ms(-2, ecgCallback, NULL, &ecgTimer);  // 500Hz
  sleeping = false;
  Serial.println("STATUS:AWAKE");
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(2000);

  set_sys_clock_khz(125000, true);
  analogReadResolution(12);
  adc_init();
  adc_set_clkdiv(48);   // ~96µs/sample — correct for 500Hz ISR

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(400000);

  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found.");
    while (true) delay(1000);
  }

  ppgSensor.setup(60, 1, 2, 100, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);
  ppgSensor.shutDown();
  Serial.println("STATUS:SLEEPING");
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {

  // ── Serial command handling ─────────────────────────────────────────────────
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START") {
      if (sleeping) wakeSensors();
      streaming = true;
    } else if (cmd == "STOP") {
      streaming = false;
      sleepSensors();
    }
  }

  if (sleeping) return;

  // ── PPG processing — 100Hz (sampleAverage=1) ──────────────────────────────
  ppgSensor.check();
  if (ppgSensor.available()) {
    float    ir     = (float)ppgSensor.getIR();
    uint32_t now_ms = millis();
    uint32_t now_us = time_us_32();

    if (ir > 50000.0f) {
      if (!dcSettled) {
        ppgDC      = ir;
        dcSettled  = true;
        fingerTime = now_ms;
      }

      ppgDC = 0.95f * ppgDC + 0.05f * ir;
      float irAC = ir - ppgDC;

      float deriv = irAC - ppgPrev;
      ppgPrev     = irAC;

      float scaledAC = irAC / 20.0f;

      bool settled = (now_ms - fingerTime) > 2000;
      bool blocked = (now_ms - lastFire_ms) < BLOCK_MS;

      // MAX30102 irAC is inverted: positive between beats, large negative drop
      // at systole.  peakAC tracks the inter-beat positive level; the drop to
      // the trough (≥10 units) arms detection, then fires at the local minimum.
      if (!blocked && scaledAC > peakAC) peakAC = scaledAC;
      float drop = peakAC - scaledAC;
      if (!blocked && drop > 10.0f) armed = true;

      bool fire = settled && armed && !blocked &&
                  ppgPrevDeriv < 0.0f && deriv >= 0.0f;
      ppgPrevDeriv = deriv;

      if (fire) {
        armed = false; peakAC = 0; lastFire_ms = now_ms;

        if (skipNextSample) {
          skipNextSample = false;
        } else {
          uint32_t rTime = rWaveTime_us;
          if (rTime != 0 && now_us >= rTime) {
            float ptt_ms = (float)(now_us - rTime) / 1000.0f;

            if (ptt_ms >= PTT_MIN_MS && ptt_ms <= PTT_MAX_MS) {
              if (pttCount < BEATS_TO_AVERAGE) {
                pttSamples[pttCount++] = ptt_ms;
                if (streaming) {
                  Serial.print("PTTRAW:");
                  Serial.println(ptt_ms, 1);
                }
              }
            } else if (streaming) {
              Serial.print("PTTBAD:");
              Serial.print(ptt_ms, 1);
              Serial.println(ptt_ms < PTT_MIN_MS ? " (too short)" : " (too long)");
            }
          }
        }
      }

      printPPG   = scaledAC;
      printFire |= fire;

    } else {
      ppgDC = ppgPrev = ppgPrevDeriv = 0;
      dcSettled = false; armed = false; peakAC = 0;
      printPPG = 0;
      pttCount = 0; skipNextSample = true;
    }

    ppgSensor.nextSample();
  }

  // ── Median PTT when buffer full ─────────────────────────────────────────────
  if (pttCount >= BEATS_TO_AVERAGE) {
    float sorted[BEATS_TO_AVERAGE];
    memcpy(sorted, pttSamples, sizeof(pttSamples));
    for (int i = 1; i < BEATS_TO_AVERAGE; i++) {
      float key = sorted[i]; int j = i - 1;
      while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j--]; }
      sorted[j+1] = key;
    }
    medianPTT      = sorted[BEATS_TO_AVERAGE / 2];
    newPTT         = true;
    pttCount       = 0;
    skipNextSample = true;
  }

  // ── Serial output at 100Hz ───────────────────────────────────────────────────
  static uint32_t lastPrint = 0;
  uint32_t now = millis();
  if (streaming && now - lastPrint >= 10) {
    lastPrint = now;

    bool rMark  = latestRmark;
    latestRmark = false;

    Serial.print(latestECG);           Serial.print(",");
    Serial.print(printPPG);            Serial.print(",");
    Serial.print(rMark    ? 3000 : 0); Serial.print(",");
    Serial.print(printFire ? 200 : 0); Serial.print(",");
    Serial.println(latestThresh, 1);

    printFire = false;

    if (newPTT) {
      newPTT = false;
      Serial.print("PTT:");
      Serial.println(medianPTT, 1);
    }
  }
}
