# PEARL Algorithm Improvement Suggestions

Target file for all firmware changes: `rpCode_dev/rpCode_dev.ino`

---

## 1. R-Wave Detection

### 1a. Add bandpass filtering before Pan-Tompkins differentiation
- Current pipeline skips the bandpass stage
- Implement a 5–15 Hz IIR bandpass in the ECG ISR before the derivative step
- Removes baseline wander (< 0.5 Hz) and muscle noise (> 40 Hz)
- Can be done as two cascaded IIR filters cheaply at 500 Hz on the RP2040

### 1b. Two-threshold detection (Hamilton-Tompkins refinement)
- Current code uses a single adaptive threshold: `ptPeak × 0.75`
- Maintain two separate thresholds: one updated on signal peaks, one on noise peaks
- Prevents a noise burst from raising the threshold high enough to miss real R-waves
- Particularly important for wrist-worn ECG with motion artifact

### 1c. T-wave rejection
- Currently not implemented
- Preeclamptic patients can exhibit tall T-waves (hyperkalemia-related)
- Rule: if a candidate peak occurs within 360 ms of the previous R-wave AND its amplitude is less than half the previous R-wave amplitude, classify as T-wave and skip
- Requires tracking the last confirmed R-wave amplitude alongside `rWaveTime_us`

### 1d. Searchback for missed beats
- If `1.66 × average_RR_interval` passes with no detection, re-scan the buffered signal at half the current threshold
- `rWaveTime_us` already uses microsecond timestamps, so RR interval tracking is straightforward
- Requires a small circular buffer of recent ECG samples in the ISR

---

## 2. PPG Sensing

### 2a. Increase effective sample rate to 100 Hz
- Current config: `sampleAverage=4` → 25 effective samples/sec → ±40 ms quantization error
- Change to `sampleAverage=1` and read MAX30102 at 100 Hz from `loop()`
- Reduces fundamental timing quantization from ±40 ms to ±10 ms
- Tradeoff: more per-sample noise, but the existing DC filter (`α=0.95`) handles this
- **Highest-leverage single change available without hardware modification**

### 2b. Adaptive derivative threshold
- Current threshold `deriv < -30.0` is hardcoded
- PPG amplitude varies with perfusion, finger pressure, and skin tone across patients
- Track a short-term peak-to-peak amplitude using an EWA of `|irAC|` peaks
- Set threshold as a fraction: `deriv < -0.15 × ppgAmplitude`
- Makes firing robust across patients rather than tuned to a single session

### 2c. Parabolic interpolation for sub-sample peak timing
- Firing on the downslope is consistent but the systolic peak is the physiologically correct PTT endpoint (pressure wavefront arrival)
- At 100 Hz (after change 2a), fit a parabola to the 3 samples around the local irAC maximum to interpolate the true peak time
- Reduces PTT timing error from ±10 ms (sample grid) to ±3–5 ms without hardware changes
- Formula: `t_peak = t_n + (s[n-1] - s[n+1]) / (2 × (s[n-1] - 2×s[n] + s[n+1])) × dt`

### 2d. Signal quality gating per beat
- Before accepting a `PTTRAW` sample, validate:
  - PPG amplitude: `ppgPeakToPeak > 20 irAC units` (catches poor finger contact)
  - RR consistency: reject if current RR interval deviates > 20% from recent RR mean (ectopic beat or motion artifact)
  - PTT range already enforced (150–400 ms) — keep this
- Emit `PTTBAD` with reason string for each rejection (already supported by serial protocol)

### 2e. Dicrotic notch false-trigger prevention
- At 100 Hz the dicrotic notch becomes detectable and can cause a spurious downslope trigger in hypertensive patients with high diastolic reflection
- The existing 500 ms block period helps, but add an amplitude gate: only fire if `irAC > midpoint of current pulse` at the time of the downslope event
- Midpoint can be tracked as `0.5 × ppgPeakToPeak` from the preceding cycle

---

## 3. PTT Variance Budget (Current vs. Improved)

| Error Source | Current | After Changes |
|---|---|---|
| PPG quantization | ±40 ms | ±10 ms (change 2a) |
| ECG R-wave jitter | ±2–4 ms | ±2 ms (change 1a) |
| Downslope vs. peak offset | ~10–20 ms variable | ±3–5 ms (change 2c) |
| Noise threshold variance | ±5–10 ms | ±2 ms (changes 1b + 2b) |
| **Total (RSS)** | **~±45 ms** | **~±12 ms** |

Reducing PTT jitter from ~±45 ms to ~±12 ms translates to roughly **±3–5 mmHg BP error** vs. the current **~±15 mmHg** — clinically significant for a preeclampsia screening tool where the threshold is 140/90 mmHg.

---

## Implementation Priority

1. **Change 2a** — increase PPG to 100 Hz (highest impact, low risk)
2. **Change 1a** — add IIR bandpass to ECG ISR
3. **Change 2b** — adaptive derivative threshold
4. **Change 2d** — signal quality gating
5. **Change 1b** — two-threshold Pan-Tompkins
6. **Change 2c** — parabolic interpolation
7. **Changes 1c, 1d, 2e** — edge case robustness

---

## Notes for Claude Code

- All firmware changes go in `rpCode_dev/rpCode_dev.ino` only
- `rpCode/rpCode.ino` and `rpCode_stable/rpCode_stable.ino` are never modified
- ECG ISR variables shared with `loop()` must remain `volatile`
- PPG `deriv` is in raw irAC units (not divided by 20) — keep consistent
- Serial protocol must continue emitting exactly 5 comma-separated values per data line; `PTT:`, `PTTRAW:`, `PTTBAD:`, `STATUS:` lines are parsed separately and are unaffected by these changes
