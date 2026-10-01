// =============================================================================
// PEARL — rpCode_dev.ino   (v5 — raw streamer, 2026-06-05)
//
// All signal processing (bandpass, Pan-Tompkins, PPG trough detection, PTT)
// now runs on the Raspberry Pi host via ptt_analysis.py.
// The RP2040 only samples and timestamps raw ADC / IR readings.
//
// Serial protocol (460800 baud):
//   Host → Device:  START\n | STOP\n
//   Device → Host:
//     STATUS:AWAKE\n / STATUS:SLEEPING\n
//     E,<raw_12bit>,<timestamp_us>\n   — ECG at 500 Hz
//     P,<raw_ir_18bit>,<timestamp_us>\n — PPG at 400 Hz
// =============================================================================

#include "hardware/adc.h"
#include "hardware/clocks.h"
#include "pico/time.h"
#include <Wire.h>
#include "MAX30105.h"

#define ECG_PIN A0

MAX30105 ppgSensor;
repeating_timer_t ecgTimer;

bool sleeping  = true;
bool streaming = false;

// ── ECG ring buffer (ISR writes, loop drains) ─────────────────────────────────
struct EcgSample { uint16_t raw; uint32_t us; };
#define ECG_BUF 32
volatile EcgSample ecgBuf[ECG_BUF];
volatile uint8_t   ecgWr = 0;
uint8_t            ecgRd = 0;

volatile int ecgStartupSkip = 250;   // skip first 500 ms while ADC settles

bool ecgCallback(repeating_timer_t *rt) {
  if (ecgStartupSkip > 0) { ecgStartupSkip--; return true; }
  uint8_t next = (ecgWr + 1) % ECG_BUF;
  if (next != ecgRd) {    // drop sample if buffer full rather than block
    ecgBuf[ecgWr].raw = (uint16_t)analogRead(ECG_PIN);
    ecgBuf[ecgWr].us  = time_us_32();
    ecgWr = next;
  }
  return true;
}

// ── State management ──────────────────────────────────────────────────────────

void resetState() {
  ecgWr = ecgRd = 0;
  ecgStartupSkip = 250;
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
  // 400 Hz PPG, sampleAverage=1, IR-only, 411 µs pulse width, 4096 nA range
  ppgSensor.setup(60, 1, 2, 400, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);
  resetState();
  add_repeating_timer_ms(-2, ecgCallback, NULL, &ecgTimer);   // strict 500 Hz
  sleeping = false;
  Serial.println("STATUS:AWAKE");
}

// ── Setup ─────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(460800);
  delay(2000);

  set_sys_clock_khz(125000, true);
  analogReadResolution(12);
  adc_init();
  adc_set_clkdiv(48);    // ~96 µs/sample — correct for 500 Hz ISR

  Wire.setSDA(4);
  Wire.setSCL(5);
  Wire.begin();
  Wire.setClock(400000);

  if (!ppgSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("ERROR: MAX30102 not found.");
    while (true) delay(1000);
  }

  ppgSensor.setup(60, 1, 2, 400, 411, 4096);
  ppgSensor.setPulseAmplitudeRed(0);
  ppgSensor.setPulseAmplitudeIR(60);
  ppgSensor.shutDown();
  Serial.println("STATUS:SLEEPING");
}

// ── Main loop ─────────────────────────────────────────────────────────────────

void loop() {

  // ── Serial commands ─────────────────────────────────────────────────────────
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

  if (sleeping || !streaming) return;

  // ── Drain ECG ring buffer ───────────────────────────────────────────────────
  while (ecgRd != ecgWr) {
    uint16_t raw = ecgBuf[ecgRd].raw;
    uint32_t us  = ecgBuf[ecgRd].us;
    ecgRd = (ecgRd + 1) % ECG_BUF;
    Serial.printf("E,%u,%lu\n", raw, (unsigned long)us);
  }

  // ── Poll PPG FIFO ──────────────────────────────────────────────────────────
  ppgSensor.check();
  if (ppgSensor.available()) {
    uint32_t ir = ppgSensor.getIR();
    uint32_t us = time_us_32();
    Serial.printf("P,%lu,%lu\n", (unsigned long)ir, (unsigned long)us);
    ppgSensor.nextSample();
  }
}
