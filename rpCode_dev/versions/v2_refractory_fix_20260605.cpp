// =============================================================================
// PEARL — rpCode_dev.ino   (v1 — full rebuild, 2026-06-05)
// Preeclampsia Early Assessment of Risk via Longitudinal data
//
// Hardware:
//   MCU  : RP2040 (Raspberry Pi Pico) @ 125 MHz
//   ECG  : AD8232 on ADC pin A0, 12-bit, 500 Hz ISR
//   PPG  : MAX30102 IR-only on I2C SDA=GP4 SCL=GP5 @ 400 kHz
//   Host : Raspberry Pi via USB Serial @ 115200 baud
//
// Serial protocol:
//   Host → Device:  START\n  |  STOP\n
//   Device → Host (streaming, 5 columns @ 100 Hz):
//     ECG_bp,PPG_ac,Rmark,Fire,Threshold\n
//     PTTRAW:###.#\n   — individual valid beat PTT (ms)
//     PTTBAD:###.# (reason)\n — rejected PTT with reason
//     PTT:###.#\n      — trimmed-mean PTT output (7 beats, drop min+max)
//     STATUS:AWAKE\n / STATUS:SLEEPING\n
//
// Changes applied (see SUGGESTIONS.md):
//   1a  IIR bandpass (LP 15 Hz → HP 5 Hz) before Pan-Tompkins
//   1b  Two-threshold Pan-Tompkins (sigLevel / noiseLevel)
//   1c  T-wave rejection at 75% of last R amplitude
//   2a  100 Hz PPG (sampleAverage=1 instead of 4)
//   2b  Amplitude tracking on abs(irAC) — signal is inverted on MAX30102
//   2c  Systolic trough detection + parabolic sub-sample interpolation
//   2d  RR-consistency quality gate + amplitude quality gate
// =============================================================================

#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "pico/time.h"
#include <Wire.h>
#include "MAX30105.h"

// ── Hardware / algorithm constants ───────────────────────────────────────────

#define ECG_PIN        A0
#define MA_SIZE        25          // 50 ms moving-average window @ 500 Hz
#define PT_WINDOW      75          // 150 ms Pan-Tompkins MWI window @ 500 Hz
#define REFRACTORY_MS  300         // minimum ms between R-wave detections (≤200 BPM)
#define BLOCK_MS       500         // minimum ms between PPG fire events

#define BEATS_TO_AVERAGE  7        // collect 7 PTTs → drop min+max → avg 5
#define PTT_MIN_MS      150.0f
#define PTT_MAX_MS      400.0f

// 1a — Bandpass filter coefficients, cascade LP(15 Hz) then HP(5 Hz) @ 500 Hz
//   LP: α = exp(−2π × 15 / 500) ≈ 0.8282
//   HP: α = RC/(RC+dt),  RC = 1/(2π×5) ≈ 0.03183,  dt = 1/500 = 0.002
#define ECG_LP_A  0.8282f
#define ECG_LP_B  0.1718f          // 1 − ECG_LP_A
#define ECG_HP_A  0.9409f

// ── Peripheral handles ────────────────────────────────────────────────────────

MAX30105 ppgSensor;
repeating_timer_t ecgTimer;

bool sleeping  = true;
bool streaming = false;

// ── ECG state (ISR + loop share via volatile) ─────────────────────────────────

float    maBuffer[MA_SIZE] = {0};
int      maIndex = 0;
float    maSum   = 0;

float    ptBuffer[PT_WINDOW] = {0};
int      ptIndex  = 0;
float    ptMWI    = 0;
bool     ptRefrac = false;
uint32_t lastR_ms = 0;

// 1b — Two-threshold state (replaces single ptPeak)
volatile float    sigLevel   = 1500.0f;  // EWA of signal peaks → starts high to avoid false triggers
volatile float    noiseLevel = 0.0f;     // EWA of noise peaks

// 1c — T-wave rejection
volatile float    lastRAmp          = 0.0f;
volatile uint32_t prevConfirmedR_ms = 0;

// 2d — RR-interval tracking (ISR writes, loop reads)
volatile float    latestRR_ms  = 0.0f;
volatile float    rrMean       = 0.0f;
volatile uint32_t prevR_for_rr = 0;

volatile float    latestECG      = 0.0f;
volatile bool     latestRmark    = false;
volatile uint32_t rWaveTime_us   = 0;
volatile float    latestThresh   = 0.0f;
volatile int      ecgStartupSkip = 250;  // skip first 500 ms (250 × 2 ms) after wake

// ── PPG state (loop only) ─────────────────────────────────────────────────────

float    ppgDC       = 0.0f;
float    ppgPrev     = 0.0f;     // previous irAC for derivative
bool     dcSettled   = false;
uint32_t fingerTime  = 0;
uint32_t lastFire_ms = 0;
float    printPPG    = 0.0f;
bool     printFire   = false;

// 2b — Amplitude tracking on abs(irAC).
//      MAX30102 IR reading DECREASES at systole (more blood = more absorption),
//      so irAC is NEGATIVE at the systolic peak.  Track the absolute value
//      so the quality gate and amplitude comparisons are sign-independent.
float ppgAmplitude = 0.0f;

// 2c — Parabolic interpolation: store the last 3 irAC samples + timing
float    irAC_nm1       = 0.0f;  // irAC one sample ago
float    irAC_nm2       = 0.0f;  // irAC two samples ago
float    ppgDerivPrev   = 0.0f;  // derivative one sample ago
uint32_t ppgTimePrev_us = 0;     // timestamp of irAC_nm1

// ── PTT collection ────────────────────────────────────────────────────────────

float pttSamples[BEATS_TO_AVERAGE];
int   pttCount       = 0;
bool  skipNextSample = true;    // discard first fire after any reset
float trimmedPTT     = 0.0f;
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
  static float x1 = 0.0f, x2 = 0.0f;
  float deriv = (raw - x2) * 0.5f;
  x2 = x1; x1 = raw;
  float sq = deriv * deriv;
  ptMWI -= ptBuffer[ptIndex];
  ptBuffer[ptIndex] = sq;
  ptMWI += sq;
  ptIndex = (ptIndex + 1) % PT_WINDOW;
  return ptMWI;
}

// ── ECG ISR — 500 Hz ─────────────────────────────────────────────────────────

bool ecgCallback(repeating_timer_t *rt) {
  // Skip the first 500 ms to let the ADC settle and prevent ptPeak spike
  if (ecgStartupSkip > 0) { ecgStartupSkip--; return true; }

  // Use time_us_32() for ISR timing — millis() may return stale values in
  // alarm-interrupt context on RP2040 (earle-philhower library quirk).
  uint32_t now_us = time_us_32();
  uint32_t now_ms = now_us / 1000;
  float raw = (float)analogRead(ECG_PIN);

  // 1a — Bandpass: cascade LP(15 Hz) then HP(5 Hz)
  static float lp = 0.0f, lp_prev = 0.0f, hp = 0.0f;
  lp      = ECG_LP_A * lp + ECG_LP_B * raw;
  hp      = ECG_HP_A * (hp + lp - lp_prev);
  lp_prev = lp;

  float smooth = movingAverage(hp);   // further smooth the bandpass output
  float mwi    = panTompkins(smooth);
  latestECG    = smooth;

  // 1b — Two-threshold adaptive detection
  float thresh = noiseLevel + 0.25f * (sigLevel - noiseLevel);
  if (thresh < 50.0f) thresh = 50.0f;
  latestThresh = thresh;

  // Clear refractory flag once 300 ms has elapsed
  if (ptRefrac && (now_ms - lastR_ms) > REFRACTORY_MS)
    ptRefrac = false;

  if (!ptRefrac && mwi > thresh) {
    // 1c — T-wave rejection: reject if candidate arrives within 360 ms of the
    //      last confirmed R-wave AND its MWI amplitude is < 75% of that R-wave.
    //      Threshold raised to 75% (from 50%) because preeclamptic patients can
    //      have tall T-waves (hyperkalemia-related).
    bool isTWave = (lastRAmp > 0.0f &&
                    (now_ms - prevConfirmedR_ms) < 400 &&
                    mwi < 0.75f * lastRAmp);

    if (!isTWave) {
      // ── Confirmed R-wave ──────────────────────────────────────────────────
      lastR_ms          = now_ms;
      rWaveTime_us      = now_us;
      prevConfirmedR_ms = now_ms;
      ptRefrac          = true;
      latestRmark       = true;
      sigLevel          = 0.125f * mwi + 0.875f * sigLevel;
      lastRAmp          = mwi;

      // 2d — Track RR interval for PPG quality gate
      if (prevR_for_rr > 0) {
        float rr = (float)(now_ms - prevR_for_rr);
        if (rr > 300.0f && rr < 2000.0f) {           // 30–200 BPM range
          latestRR_ms = rr;
          rrMean = (rrMean == 0.0f) ? rr : 0.875f * rrMean + 0.125f * rr;
        }
      }
      prevR_for_rr = now_ms;

    } else {
      // ── T-wave: update noise level and enforce refractory ─────────────────
      noiseLevel = 0.125f * mwi + 0.875f * noiseLevel;
      ptRefrac   = true;
      lastR_ms   = now_ms;
    }

  } else if (mwi > thresh * 0.5f && mwi <= thresh) {
    // 1b — Sub-threshold peak → update noise level only
    noiseLevel = 0.125f * mwi + 0.875f * noiseLevel;
  }

  return true;
}

// ── Sensor power management ───────────────────────────────────────────────────

void resetState() {
  memset(maBuffer, 0, sizeof(maBuffer));
  maSum = 0.0f; maIndex = 0;
  memset(ptBuffer, 0, sizeof(ptBuffer));
  ptMWI = 0.0f; ptIndex = 0;

  ptRefrac = false; lastR_ms = 0;
  sigLevel = 1500.0f; noiseLevel = 0.0f;
  lastRAmp = 0.0f; prevConfirmedR_ms = 0;
  latestRR_ms = 0.0f; rrMean = 0.0f; prevR_for_rr = 0;
  latestECG = 0.0f; latestRmark = false; rWaveTime_us = 0; latestThresh = 0.0f;
  ecgStartupSkip = 250;

  ppgDC = ppgPrev = 0.0f;
  dcSettled = false;
  ppgAmplitude = 0.0f;
  irAC_nm1 = irAC_nm2 = ppgDerivPrev = 0.0f;
  ppgTimePrev_us = 0;
  lastFire_ms = 0; fingerTime = 0;
  printPPG = 0.0f; printFire = false;

  pttCount = 0; skipNextSample = true;
  trimmedPTT = 0.0f; newPTT = false;
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
  // 2a — sampleAverage=1 → 100 Hz effective (was 4 → 25 Hz)
  //      Reduces PPG timing quantization from ±40 ms to ±10 ms
  ppgSensor.setup(60, 1, 2, 100, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);
  resetState();
  add_repeating_timer_ms(-2, ecgCallback, NULL, &ecgTimer);  // negative = strict 500 Hz
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
  adc_set_clkdiv(48);     // ~96 µs/sample → adequate for 500 Hz ISR

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(400000);

  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found.");
    while (true) delay(1000);
  }

  // Initialise then immediately sleep — device waits for START from host
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

  // ── PPG processing — 100 Hz effective (sampleAverage=1) ───────────────────
  ppgSensor.check();
  if (ppgSensor.available()) {
    float    ir     = (float)ppgSensor.getIR();
    uint32_t now_ms = millis();
    uint32_t now_us = time_us_32();

    if (ir > 50000.0f) {
      // Initialise DC filter on first sample after finger placement
      if (!dcSettled) {
        ppgDC      = ir;
        dcSettled  = true;
        fingerTime = now_ms;
      }

      // DC removal: slow EWA tracks the DC baseline
      ppgDC      = 0.95f * ppgDC + 0.05f * ir;
      float irAC = ir - ppgDC;

      // Sample-to-sample derivative (raw irAC units, not divided by 20)
      float deriv = irAC - ppgPrev;
      ppgPrev     = irAC;

      float scaledAC = irAC / 20.0f;    // graph output only

      bool settled = (now_ms - fingerTime) > 2000;   // wait 2 s after placement
      bool blocked = (now_ms - lastFire_ms) < BLOCK_MS;

      // 2b — Track amplitude as the maximum abs(irAC), slow decay.
      //      Because irAC is inverted (systole = negative trough), abs() keeps
      //      the tracking sign-independent.
      float absAC = fabsf(irAC);
      if (absAC > ppgAmplitude) ppgAmplitude = absAC;
      ppgAmplitude *= 0.9990f;

      // 2c — Detect the systolic trough (local minimum of irAC).
      //      The derivative crosses from negative to positive at a trough.
      //      The amplitude gate checks that irAC_nm1 (the trough candidate)
      //      is genuinely negative and significant.
      bool troughDetected = (ppgDerivPrev < 0.0f && deriv >= 0.0f);
      bool fire = settled && !blocked && troughDetected &&
                  irAC_nm1 < -0.05f * ppgAmplitude;

      if (fire) {
        lastFire_ms = now_ms;

        // 2c — Parabolic sub-sample interpolation for the trough time.
        //   t_trough = t_n + (s[n-1] − s[n+1]) / (2×(s[n-1] − 2×s[n] + s[n+1])) × dt
        //   n = irAC_nm1 (the trough),  n-1 = irAC_nm2,  n+1 = current irAC.
        //   Formula is identical for minima and maxima.
        uint32_t trough_us = ppgTimePrev_us;
        float denom = irAC_nm2 - 2.0f * irAC_nm1 + irAC;
        if (fabsf(denom) > 0.5f && ppgTimePrev_us > 0) {
          float t_off = (irAC_nm2 - irAC) / (2.0f * denom);
          if (t_off >  0.5f) t_off =  0.5f;    // clamp to ±0.5 samples
          if (t_off < -0.5f) t_off = -0.5f;
          uint32_t dt_us = now_us - ppgTimePrev_us;
          trough_us = ppgTimePrev_us + (uint32_t)(t_off * (float)dt_us);
        }

        if (skipNextSample) {
          skipNextSample = false;   // DC filter not yet converged — discard
        } else {
          uint32_t rTime = rWaveTime_us;
          if (rTime != 0 && trough_us >= rTime) {
            float ptt_ms = (float)(trough_us - rTime) / 1000.0f;

            // 2d — Quality gate before accepting the PTT sample
            bool amplOK = (ppgAmplitude > 20.0f);
            bool rrOK   = (rrMean == 0.0f) ||
                          (latestRR_ms > rrMean * 0.8f &&
                           latestRR_ms < rrMean * 1.2f);

            if (amplOK && rrOK && ptt_ms >= PTT_MIN_MS && ptt_ms <= PTT_MAX_MS) {
              if (pttCount < BEATS_TO_AVERAGE) {
                pttSamples[pttCount++] = ptt_ms;
                if (streaming) {
                  Serial.print("PTTRAW:"); Serial.println(ptt_ms, 1);
                }
              }
            } else if (streaming) {
              Serial.print("PTTBAD:"); Serial.print(ptt_ms, 1);
              if      (!amplOK)              Serial.println(" (low amplitude)");
              else if (!rrOK)                Serial.println(" (irregular RR)");
              else if (ptt_ms < PTT_MIN_MS)  Serial.println(" (too short)");
              else                           Serial.println(" (too long)");
            }
          }
        }
      }

      // Shift sample history for the next parabolic interpolation
      irAC_nm2       = irAC_nm1;
      irAC_nm1       = irAC;
      ppgDerivPrev   = deriv;
      ppgTimePrev_us = now_us;

      printPPG   = scaledAC;
      printFire |= fire;

    } else {
      // Finger removed — reset all PPG-side state
      ppgDC = ppgPrev = 0.0f;
      dcSettled = false;
      ppgAmplitude = 0.0f;
      irAC_nm1 = irAC_nm2 = ppgDerivPrev = 0.0f;
      ppgTimePrev_us = 0;
      printPPG = 0.0f;
      pttCount = 0; skipNextSample = true;
    }

    ppgSensor.nextSample();
  }

  // ── Compute trimmed-mean PTT when buffer is full ──────────────────────────
  // Sort 7 values, drop the lowest and highest, average the middle 5
  if (pttCount >= BEATS_TO_AVERAGE) {
    float sorted[BEATS_TO_AVERAGE];
    memcpy(sorted, pttSamples, sizeof(pttSamples));
    for (int i = 1; i < BEATS_TO_AVERAGE; i++) {
      float key = sorted[i]; int j = i - 1;
      while (j >= 0 && sorted[j] > key) { sorted[j + 1] = sorted[j--]; }
      sorted[j + 1] = key;
    }
    float trimSum = 0.0f;
    for (int i = 1; i < BEATS_TO_AVERAGE - 1; i++) trimSum += sorted[i];
    trimmedPTT     = trimSum / (float)(BEATS_TO_AVERAGE - 2);  // ÷5
    newPTT         = true;
    pttCount       = 0;
    skipNextSample = true;
  }

  // ── Serial output at 100 Hz ───────────────────────────────────────────────
  // 5-column format: ECG,PPG,Rmark,Fire,Threshold
  static uint32_t lastPrint = 0;
  uint32_t now = millis();
  if (streaming && now - lastPrint >= 10) {
    lastPrint = now;

    bool rMark  = latestRmark;
    latestRmark = false;

    Serial.print(latestECG);             Serial.print(",");
    Serial.print(printPPG);              Serial.print(",");
    Serial.print(rMark     ? 3000 : 0);  Serial.print(",");
    Serial.print(printFire ? 200  : 0);  Serial.print(",");
    Serial.println(latestThresh, 1);
    printFire = false;

    if (newPTT) {
      newPTT = false;
      Serial.print("PTT:"); Serial.println(trimmedPTT, 1);
    }
  }
}
