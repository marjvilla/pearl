# PEARL — Project Overview

**PEARL**: Preeclampsia Early Assessment of Risk via Longitudinal data.  
A clinical Raspberry Pi app that estimates blood pressure non-invasively using a RP2040 wristband, by measuring **Pulse Transit Time (PTT)** — the time delay between the ECG R-wave and the PPG pulse arrival.

---

## Hardware

| Component | Part | Interface |
|---|---|---|
| Microcontroller | RP2040 (Raspberry Pi Pico) | USB Serial @ 115200 baud |
| ECG | AD8232 analog front-end | ADC pin A0 (12-bit, 500Hz) |
| PPG | MAX30102 optical sensor | I2C @ 400kHz, SDA=GP4, SCL=GP5 |
| Host | Raspberry Pi (runs Python UI) | USB Serial |

**MAX30102 config**: `setup(ledBrightness=60, sampleAverage=4, ledMode=2_IR_only, sampleRate=100, pulseWidth=411, adcRange=4096)` → 25 effective samples/sec. IR LED amplitude = 60 (Red LED disabled).

---

## How PTT Blood Pressure Works

```
PTT (ms) = time from ECG R-wave peak → PPG pulse arrival at fingertip

SBP ≈ SBP_A - SBP_B × ln(PTT_ms)     [Moens-Korteweg equation]
DBP ≈ DBP_A - DBP_B × ln(PTT_ms)

Population defaults: SBP_A=517.7, SBP_B=71.4, DBP_A=340.0, DBP_B=46.0
Per-patient calibration: two-point fit using reference cuff readings.
```

Valid PTT range: **150–400 ms**. Shorter = too short (noise), longer = stale R-wave.

---

## File Structure

```
/Volumes/PiShare/
├── rpCode/rpCode.ino              ← STABLE production firmware (never modify)
├── rpCode_stable/rpCode_stable.ino← FROZEN backup (never modify)
├── rpCode_dev/rpCode_dev.ino      ← EXPERIMENTAL firmware (all changes go here)
├── bp_ptt_merged/bp_ptt_merged.ino← Original reference code (never modify)
└── uixml2/
    ├── main.py                    ← PySide6 clinical UI
    ├── ptt_serial.py              ← Serial comms + PTT→BP conversion
    ├── database.py                ← SQLite patient database
    ├── risk_predictor.py          ← XGBoost preeclampsia risk model
    ├── new.ui                     ← Qt Designer UI layout
    ├── preeclampsia_final_xgb.pkl ← Trained model
    └── serial_log_*.txt           ← Test logs from live sessions
```

**Rule**: `rpCode_dev/rpCode_dev.ino` is the only firmware file to edit. The others are reference/stable.

---

## Firmware — rpCode_dev.ino

### Serial Protocol

**Host → Device:**
- `START\n` — wake sensors, begin streaming
- `STOP\n` — stop streaming, sleep sensors

**Device → Host (streaming):**
- Data line (100Hz): `ECG,PPG,Rmark,Fire,Threshold\n`
  - ECG: smoothed ADC value (~2000 at rest)
  - PPG: raw irAC/20 (scaled AC component, ~±3 at rest)
  - Rmark: `3000` on R-wave beat, else `0`
  - Fire: `200` on PPG trigger event, else `0`
  - Threshold: Pan-Tompkins detection threshold
- `PTTRAW:242.1\n` — individual valid PTT sample (ms)
- `PTTBAD:578.5 (too long)\n` — rejected sample with reason
- `PTT:315.2\n` — trimmed-mean PTT output (7 samples, drop min+max, average 5)
- `STATUS:AWAKE\n` / `STATUS:SLEEPING\n`

### ECG Processing (500Hz ISR)

```
analogRead(A0)
  → movingAverage(MA_SIZE=25, 50ms window)
  → panTompkins(PT_WINDOW=75, 150ms window): deriv→square→MWI
  → adaptive threshold: ptPeak × 0.75, decay 0.9998/sample
  → R-wave detection: mwi > thresh && thresh > 50 && !refractory(200ms)
  → records rWaveTime_us (microsecond timestamp)
```

Startup: `ecgStartupSkip=250` skips first 500ms after wake to avoid ADC transient corrupting ptPeak.

Timer: `add_repeating_timer_ms(-2, ecgCallback, ...)` → negative = strict 2ms period.

ADC clock: `adc_set_clkdiv(48)` → ~96μs/sample, fast enough for 500Hz ISR.

### PPG Processing (25Hz, main loop)

```
MAX30102 IR reading
  → DC filter: ppgDC = 0.95×ppgDC + 0.05×ir  (EWA, α=0.95)
  → irAC = ir - ppgDC
  → deriv = irAC - ppgPrev  (sample-to-sample difference, irAC units)
  → finger detect: ir > 50000 (raw counts)
  → settled: finger present > 2000ms
  → blocked: lastFire < 500ms ago
  → FIRE when: settled && !blocked && deriv < -30.0
```

**Key design decision — why fire on downslope, not trough:**  
The PPG signal at 25Hz is quantized to ±40ms timing resolution. The trough (zero-crossing) occurs in a flat, noisy region where noise randomly flips the derivative sign, causing ±20–40ms PTT jitter. The steep downslope (`deriv < -30 irAC/sample`) is a sharp, consistent impulse: measured at ~−55 irAC/sample at the pulse, noise floor is ~±10 irAC/sample. Firing at the downslope onset is deterministic and gives consistent PTT timing. The 500ms block period prevents double-firing within the same pulse.

### PTT Collection

- `skipNextSample=true` after each reset — first PPG event after wake is discarded (DC hasn't converged)
- Collects `BEATS_TO_AVERAGE=7` valid PTTRAW values
- Trimmed mean: sort 7, drop lowest and highest, average middle 5
- Resets collection after each PTT output

---

## Python UI — uixml2/

### main.py

PySide6 app targeting a **1024×600 touchscreen** (Raspberry Pi 7" or ROADOM 10.1" display).

**Pages** (Qt stacked widget):
- `welcome_page` — home, patient login, staff login, test mode buttons
- `patient_login_page` — code + DOB entry
- `registration_info_page` / `_page_2` / `_page_3` — multi-step registration
- `registration_confirm_page` — shows generated patient code
- `patient_home_page` — BP reading display, measure button, calibrate, risk model
- `calibration_page` — two-point PTT calibration against reference cuff
- `staff_dashboard_page` — patient search
- `staff_patient_detail_page` — past readings view

**SerialPlotterDialog**: Live pyqtgraph window shown during bracelet reads.
- ECG plot (top): blue line + orange dashed threshold + yellow dots for R-marks
- PPG plot (bottom): green line, y-axis locked to [-10, 10], red triangles for fire events
- PPG display applies EWA smoother (α=0.5) for visual clarity only — doesn't affect detection
- Log pane: shows PTTRAW/PTTBAD/PTT lines in real time
- Saves session to `serial_log_YYYYMMDD_HHMMSS.txt` on close

**Timeouts**: 2-minute inactivity timer returns to home page.

**Test mode**: `TEST0000` patient code, bypasses login.

### ptt_serial.py

- Auto-detects RP2040 by USB VID (`0x2E8A`), falls back to `usbmodem`/`ttyACM`
- Sends `START`, reads `PTT:` lines until `NUM_SAMPLES=3` collected, sends `STOP`
- `read_ptt_once_blocking()` → returns `BPResult(systolic, diastolic, heart_rate=0)`
- `read_ptt_raw_blocking()` → returns median PTT in ms (for calibration screen)
- PTT→BP: logarithmic model, per-patient calibration coefficients override population defaults

### database.py / patients.db

SQLite. Tables: patients, bp_readings, risk_factors, ptt_calibration_points.

### risk_predictor.py

XGBoost model (`preeclampsia_final_xgb.pkl`) returning Low/Moderate/High risk with probability. Requires at least 1 BP reading; improves with 2–3 visits.

---

## Known Issues / Recent History

- **ECG startup transient**: First 500ms of ADC samples spike ptPeak to 800K+. Fixed with `ecgStartupSkip=250` counter.
- **PPG DC convergence spike**: First ~2s after finger placement, irAC shows large transient (~5000 scaledAC units). Suppressed by `settled = fingerTime > 2000ms` — no fires during this period.
- **Stale rTime PTT "too long"**: PPG could fire before any R-wave detected (rWaveTime_us=0), giving enormous age_us. Fixed with `if (rTime != 0)` guard.
- **PPG amplitude too small for drop detection**: Stable signal is ~±3 scaledAC units. Previous `drop > 10.0f` arming never triggered. Replaced with derivative threshold.
- **PPG trough jitter**: Zero-crossing detection at the noisy trough caused ±20–40ms PTT scatter. Fixed by firing at the steep downslope onset instead.

---

## What to Know Before Modifying

1. **Only edit `rpCode_dev/rpCode_dev.ino`** for firmware experiments. `rpCode/` and `rpCode_stable/` are untouched references.
2. ECG runs in a hardware timer ISR at 500Hz. Variables shared with `loop()` are `volatile`.
3. PPG runs in `loop()` at 25Hz effective. The `deriv` is in raw irAC units (not divided by 20).
4. The trimmed-mean PTT requires exactly `BEATS_TO_AVERAGE=7` valid samples before outputting.
5. The UI expects exactly 5 comma-separated values per data line. Non-data lines (`PTT:`, `PTTRAW:`, `STATUS:`) are parsed separately.
6. BP estimation coefficients are population-level defaults — accuracy improves significantly with per-patient calibration using a reference cuff.
