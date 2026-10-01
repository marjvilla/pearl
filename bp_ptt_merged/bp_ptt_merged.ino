#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "pico/time.h"
#include <Wire.h>
#include "MAX30105.h"

#define ECG_PIN  A0
#define MA_SIZE        5
#define PT_WINDOW     15
#define REFRACTORY_MS 200
#define BLOCK_MS      500

// PTT config — conservative
#define BEATS_TO_AVERAGE  5
#define PTT_MIN_MS      150.0f
#define PTT_MAX_MS      400.0f

MAX30105 ppgSensor;

// ECG
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

// PPG
float    ppgDC        = 0;
float    ppgPrev      = 0;
float    ppgPrevDeriv = 0;
bool     dcSettled    = false;
uint32_t fingerTime   = 0;
bool     armed        = false;
float    peakAC       = 0;
uint32_t lastFire_ms  = 0;

// For graph output
float printPPG  = 0;
bool  printFire = false;

// PTT collection
float pttSamples[BEATS_TO_AVERAGE];
int   pttCount       = 0;
bool  skipNextSample = true;
float medianPTT      = 0;
bool  newPTT         = false;

repeating_timer_t ecgTimer;

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
  x2 = x1;
  x1 = raw;
  float sq = deriv * deriv;
  ptMWI -= ptBuffer[ptIndex];
  ptBuffer[ptIndex] = sq;
  ptMWI += sq;
  ptIndex = (ptIndex + 1) % PT_WINDOW;
  return ptMWI;
}

bool ecgCallback(repeating_timer_t *rt) {
  uint32_t now_ms = millis();
  uint32_t now_us = time_us_32();
  float raw    = (float)analogRead(ECG_PIN);
  float smooth = movingAverage(raw);
  float mwi    = panTompkins(smooth);

  latestECG = smooth;

  if (mwi > ptPeak) ptPeak = mwi;
  ptPeak      *= 0.9990f;
  float thresh  = ptPeak * 0.75f;

  if (ptRefrac && (now_ms - lastR_ms) > REFRACTORY_MS) {
    ptRefrac = false;
  }

  if (!ptRefrac && mwi > thresh && thresh > 50.0f) {
    lastR_ms     = now_ms;
    rWaveTime_us = now_us;
    ptRefrac     = true;
    ptPeak      *= 0.6f;
    latestRmark  = true;
  }

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  set_sys_clock_khz(125000, true);
  analogReadResolution(12);
  adc_init();
  adc_set_clkdiv(480);

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(400000);

  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found.");
    while (true) delay(1000);
  }

  ppgSensor.setup(60, 4, 2, 100, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);

  add_repeating_timer_ms(-10, ecgCallback, NULL, &ecgTimer);
}

void loop() {

  // ── PPG processing — one sample per iteration ─────────────────────────────
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

      ppgDC      = 0.95f * ppgDC + 0.05f * ir;
      float irAC = ir - ppgDC;
      float deriv = irAC - ppgPrev;
      ppgPrev    = irAC;

      bool settled = (now_ms - fingerTime) > 2000;
      bool blocked = (now_ms - lastFire_ms) < BLOCK_MS;

      float scaledAC = irAC / 20.0f;

      if (!blocked && scaledAC > peakAC) peakAC = scaledAC;

      float drop = peakAC - scaledAC;
      if (!blocked && drop > 10.0f) armed = true;

      bool fire = settled &&
                  armed &&
                  !blocked &&
                  ppgPrevDeriv < 0 &&
                  deriv > 0;

      ppgPrevDeriv = deriv;

      if (fire) {
        armed       = false;
        peakAC      = 0;
        lastFire_ms = now_ms;

        // ── PTT calculation ──
        if (skipNextSample) {
          skipNextSample = false;
        } else {
          uint32_t rTime  = rWaveTime_us;   // read volatile once
          uint32_t age_us = now_us - rTime;
          float ptt_ms    = (float)age_us / 1000.0f;

          if (ptt_ms >= PTT_MIN_MS && ptt_ms <= PTT_MAX_MS) {
            if (pttCount < BEATS_TO_AVERAGE) {
              pttSamples[pttCount++] = ptt_ms;
            }
          }
        }
      }

      printPPG  = scaledAC;
      printFire = fire;

    } else {
      ppgDC = ppgPrev = ppgPrevDeriv = 0;
      dcSettled = false;
      armed     = false;
      peakAC    = 0;
      printPPG  = 0;
      pttCount  = 0;
      skipNextSample = true;
    }

    ppgSensor.nextSample();
  }

  // ── Median PTT when buffer full ───────────────────────────────────────────
  if (pttCount >= BEATS_TO_AVERAGE) {
    float sorted[BEATS_TO_AVERAGE];
    memcpy(sorted, pttSamples, sizeof(pttSamples));
    for (int i = 1; i < BEATS_TO_AVERAGE; i++) {
      float key = sorted[i];
      int j = i - 1;
      while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j--]; }
      sorted[j+1] = key;
    }
    medianPTT      = sorted[BEATS_TO_AVERAGE / 2];
    newPTT         = true;
    pttCount       = 0;
    skipNextSample = true;
  }

  // ── Graph output at 100Hz ─────────────────────────────────────────────────
  static uint32_t lastPrint = 0;
  uint32_t now = millis();
  if (now - lastPrint >= 10) {
    lastPrint = now;

    bool rMark  = latestRmark;
    latestRmark = false;

    Serial.print(latestECG);           Serial.print(",");
    Serial.print(printPPG);            Serial.print(",");
    Serial.print(rMark    ? 3000 : 0); Serial.print(",");
    Serial.println(printFire ? 200 : 0);

    printFire = false;

    // PTT printed on its own line — won't disrupt 4-trace plotter much,
    // but if you want pure graphs, comment this block out
    if (newPTT) {
      newPTT = false;
      Serial.print(">>> Median PTT: ");
      Serial.print(medianPTT, 1);
      Serial.println(" ms <<<");
    }
  }
}
