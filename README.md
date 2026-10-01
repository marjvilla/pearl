# PEARL — Preeclampsia Early Assessment of Risk via Longitudinal data

PEARL is a wearable wristband that estimates blood pressure non-invasively using **pulse transit time (PTT)** — the time it takes a pressure wave to travel from the heart to the wrist — derived from simultaneous ECG and PPG acquisition. The goal is longitudinal, cuffless BP tracking to support early detection of preeclampsia risk during pregnancy.

## How it works

1. **ECG** — An AD8232 analog front-end conditions the electrical signal from skin electrodes; an RP2040 samples it at 500 Hz. R-waves are detected with a Pan-Tompkins-derived algorithm (LP filter → moving average → derivative → squaring → moving-window integration → adaptive threshold).
2. **PPG** — A MAX30102 optical sensor shines IR light into the wrist and measures reflected intensity at 400 Hz. Blood volume changes with each heartbeat change how much light is absorbed, producing a pulsatile waveform. The **foot** (onset) of each pulse — not the peak — is detected as the PTT fiducial point.
3. **PTT** — Computed as the time between the ECG R-wave and the following PPG pulse foot. Samples are quality-gated (PPG amplitude, RR-interval consistency, physiological PTT range) before being accepted. Seven valid samples are collected per reading and combined with a trimmed mean.
4. **BP estimation** — PTT is converted to systolic/diastolic BP via the Moens-Korteweg relationship, using either population-default coefficients or per-patient calibration.
5. **Risk scoring** — Patient demographics, risk factors, and BP readings feed into an XGBoost model (`preeclampsia_final_xgb.pkl`) to flag elevated preeclampsia risk.

## Repository layout

| Path | Description |
|---|---|
| `main.py` | PySide6 desktop app — patient registration, live ECG/PPG plotter, BP measurement, risk display |
| `ptt_analysis.py` | Core signal processing: ECG R-wave detection, PPG foot detection, PTT quality gating |
| `ptt_serial.py` | Serial communication with the RP2040 |
| `risk_predictor.py` | Loads the trained model and scores preeclampsia risk |
| `database.py` | SQLite patient/session storage |
| `Train_model_updated.py` | Training script for the risk model |
| `rpCode_dev/` | RP2040 firmware (`.ino`) and version history of the signal-processing algorithm |
| `ptt_sessions/` | Recorded session data (raw waveform CSVs + session JSON summaries) |
| `f33945152.c` | Reference ECG detection algorithm the current Python implementation is ported from |
| `new.ui` / `f22816256.xml` | Qt Designer UI definitions |
| `*.pkl` | Trained risk model artifacts (XGBoost model, feature list, decision threshold) |

## Hardware

- **RP2040** microcontroller — samples ECG at 500 Hz and PPG at 400 Hz, streams raw samples over serial at 460800 baud
- **AD8232** — single-lead ECG analog front-end (instrumentation amp + filtering + gain)
- **MAX30102** — integrated pulse oximetry/heart-rate sensor (IR + red LEDs, photodetector, I2C output)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

Requires Python 3.9+. Connect the RP2040 over USB before starting a live session.

## Session data format

Each recorded session produces:
- `raw_<timestamp>.csv` — raw ECG/PPG samples with microsecond timestamps
- `session_<timestamp>.json` — patient info, calibration coefficients, PTT values, estimated BP, and signal quality metrics (fire count, acceptance rate, rejection breakdown)
