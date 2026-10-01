"""
ptt_analysis.py — ECG + PPG signal processing on the Raspberry Pi.

ECG detection is a direct Python port of f33945152.c:
  raw ADC → LP (IIR 15 Hz, equivalent to f33945152's 25-sample MA) →
  40-sample MA (wider than f33945152's 25-sample MA to compensate for the
  v5 firmware's noisier ADC clock) → Pan-Tompkins MWI →
  single-threshold with ptPeak × 0.6 drop after each detection.

No HP filter — the HP stage created a biphasic QRS output which produced
two equal-amplitude MWI peaks per beat.  Pan-Tompkins differentiates
internally so it is DC-immune; the HP was never needed.
"""

from collections import deque


class AnalysisEngine:
    """
    Stateful single-channel ECG + PPG processor.

    Typical usage::

        engine = AnalysisEngine()
        ecg_val, thresh, r_det = engine.feed_ecg(raw_adc, timestamp_us)
        scaled_ac, fire, ptt_raw, ptt_bad, ptt_result = engine.feed_ppg(raw_ir, timestamp_us)
    """

    # ── ECG constants — direct port of f33945152.c ────────────────────────────
    _LP_A  = 0.8282;  _LP_B = 0.1718  # IIR LP at 15 Hz
    _MA_N  = 40                        # MA window: 80 ms
    _PT_N  = 75                        # Pan-Tompkins MWI window: 150 ms
    _PT_PEAK_INIT  = 500.0             # initial ptPeak
    _PT_PEAK_DECAY = 0.9998            # per-sample slow decay
    _PT_PEAK_DROP  = 0.6               # drop 40 % after each R detection
    _PT_THRESH_RATIO = 0.75            # threshold = ptPeak × 0.75
    _PT_THRESH_MIN   = 50.0            # minimum threshold guard
    _REFRACTORY_US   = 200_000         # 200 ms refractory
    _STARTUP_SKIP    = 250             # skip first 500 ms while ADC settles

    # ── DC removal constants ──────────────────────────────────────────────────
    # The LP-only signal sits at the raw ADC DC level (~2000–4095 counts).
    # At startup the ADC can saturate, driving ptPeak to ~350 000 which takes
    # ~42 s to decay — blocking all R-wave detection during that window.
    # A two-speed EWA baseline tracks DC quickly during the startup skip, then
    # switches to a very slow rate that removes drift without distorting the QRS.
    _DC_A_FAST = 0.980   # fast tracking during startup skip  (τ ≈ 50 samples = 100 ms)
    _DC_A_SLOW = 0.9975  # slow tracking during operation      (τ ≈ 400 samples = 800 ms)

    # ── PPG detection constants ───────────────────────────────────────────────
    _BLOCK_US   = 500_000              # 500 ms between fires
    _SETTLE_US  = 2_000_000            # 2 s finger-on settle time
    _DROP_ARM   = 7.0                  # drop from peak to arm (scaled units)
    _DROP_FIRE  = 10.0                 # drop from peak required at fire time

    # ── PTT collection ────────────────────────────────────────────────────────
    _PTT_COLLECT = 7
    _PTT_MIN_MS  = 100.0
    _PTT_MAX_MS  = 450.0

    def __init__(self):
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def feed_ecg(self, raw: int, timestamp_us: int):
        """
        Process one raw 12-bit ECG sample (500 Hz).

        Returns
        -------
        filtered_ecg : float   LP + MA output (display value)
        threshold    : float   current detection threshold
        r_detected   : bool    True on a confirmed R-wave
        """
        # ── LP + MA always run (even during startup skip) ────────────────────
        # This lets the DC baseline warm up during the skip so the first real
        # sample after the skip sees a near-zero AC signal rather than the full
        # ADC DC offset, preventing ptPeak from being set to astronomic values.
        self._lp      = self._LP_A * self._lp + self._LP_B * raw
        self._ma_sum += self._lp - self._ma_buf[0]
        self._ma_buf.append(self._lp)
        smooth        = self._ma_sum / self._MA_N

        # DC baseline: fast EWA during startup skip, slow EWA thereafter
        dc_a = self._DC_A_SLOW if self._skip_count == 0 else self._DC_A_FAST
        self._dc_baseline = dc_a * self._dc_baseline + (1 - dc_a) * smooth
        signal = smooth - self._dc_baseline   # AC-only, near zero between beats

        if self._skip_count > 0:
            self._skip_count -= 1
            return 0.0, 0.0, False

        # ── Pan-Tompkins on DC-removed signal ────────────────────────────────
        deriv         = (signal - self._pt_x2) * 0.5
        self._pt_x2   = self._pt_x1
        self._pt_x1   = signal
        sq            = deriv * deriv
        self._pt_mwi += sq - self._pt_buf[0]
        self._pt_buf.append(sq)

        # ── f33945152.c threshold: running max + slow decay ──────────────────
        if self._pt_mwi > self._pt_peak:
            self._pt_peak = self._pt_mwi
        self._pt_peak *= self._PT_PEAK_DECAY

        thresh     = self._pt_peak * self._PT_THRESH_RATIO
        thresh     = max(thresh, self._PT_THRESH_MIN)
        in_refrac  = (timestamp_us - self._last_r_us) < self._REFRACTORY_US
        r_detected = False

        if not in_refrac and self._pt_mwi > thresh:
            self._last_r_us = timestamp_us
            self._r_time_us = timestamp_us
            self._pt_peak  *= self._PT_PEAK_DROP
            r_detected = True

            if self._prev_r_for_rr:
                rr = timestamp_us - self._prev_r_for_rr
                if 300_000 < rr < 2_000_000:
                    self._latest_rr_us = rr
                    self._rr_mean = (rr if self._rr_mean == 0 else
                                     0.875 * self._rr_mean + 0.125 * rr)
            self._prev_r_for_rr = timestamp_us

        return signal, thresh, r_detected  # signal is DC-removed → clean display

    def feed_ppg(self, raw_ir: int, timestamp_us: int):
        """
        Process one raw PPG IR sample (400 Hz).

        Returns
        -------
        scaled_ac  : float        irAC / 20 for display
        fire       : bool         True when a systolic trough is detected
        ptt_raw    : float | None accepted PTT sample (ms)
        ptt_bad    : tuple | None (ptt_ms, reason_str) for a rejected sample
        ptt_result : float | None trimmed-mean PTT when the 7-sample buffer fills
        """
        if raw_ir < 50_000:
            self._ppg_dc = self._ppg_prev = self._ppg_prev_deriv = 0.0
            self._dc_settled = False
            self._armed = False
            self._peak_ac = 0.0
            return 0.0, False, None, None, None

        if not self._dc_settled:
            self._ppg_dc     = float(raw_ir)
            self._dc_settled = True
            self._finger_us  = timestamp_us

        self._ppg_dc  = 0.95 * self._ppg_dc + 0.05 * raw_ir
        ir_ac         = raw_ir - self._ppg_dc
        deriv         = ir_ac - self._ppg_prev
        self._ppg_prev = ir_ac
        scaled_ac     = ir_ac / 20.0

        settled = (timestamp_us - self._finger_us)    > self._SETTLE_US
        blocked = (timestamp_us - self._last_fire_us) < self._BLOCK_US

        if not blocked and scaled_ac > self._peak_ac:
            self._peak_ac = scaled_ac
        drop = self._peak_ac - scaled_ac
        if not blocked and drop > self._DROP_ARM:
            self._armed = True

        # Pulse-foot detection: fire on the descending slope once the drop is
        # large enough.  This is the clinical "onset" method — fires at the
        # foot of the systolic wave rather than the trough, which is earlier,
        # more reproducible, and avoids the dicrotic-notch confusion that
        # plagued the local-minimum (prev_deriv<0, deriv>=0) approach.
        fire = (settled and self._armed and not blocked and
                deriv < 0.0 and
                drop > self._DROP_FIRE)
        self._ppg_prev_deriv = deriv

        ptt_raw = ptt_bad = ptt_result = None

        if fire:
            self._armed = False
            self._peak_ac = 0.0
            self._last_fire_us = timestamp_us

            if self._skip_next:
                self._skip_next = False
            elif self._r_time_us and timestamp_us >= self._r_time_us:
                ptt_ms = (timestamp_us - self._r_time_us) / 1000.0
                rr_ok  = (self._rr_mean == 0 or
                          (self._latest_rr_us > self._rr_mean * 0.8 and
                           self._latest_rr_us < self._rr_mean * 1.2))

                if rr_ok and self._PTT_MIN_MS <= ptt_ms <= self._PTT_MAX_MS:
                    ptt_raw = ptt_ms
                    self._ptt_samples.append(ptt_ms)
                    if len(self._ptt_samples) >= self._PTT_COLLECT:
                        ptt_result = self._trimmed_mean()
                else:
                    reason = ("irregular RR" if not rr_ok else
                              ("too short"   if ptt_ms < self._PTT_MIN_MS else
                               "too long"))
                    ptt_bad = (ptt_ms, reason)

        return scaled_ac, fire, ptt_raw, ptt_bad, ptt_result

    def reset(self):
        """Reset all state (called on new session / finger removed)."""
        # ECG filter state
        self._lp      = 0.0
        self._ma_buf  = deque([0.0] * self._MA_N, maxlen=self._MA_N)
        self._ma_sum  = 0.0
        self._pt_buf  = deque([0.0] * self._PT_N, maxlen=self._PT_N)
        self._pt_mwi  = 0.0
        self._pt_x1   = self._pt_x2 = 0.0
        # DC removal state
        self._dc_baseline = 0.0          # warmed up during startup skip
        # f33945152.c-style detection state
        self._pt_peak   = self._PT_PEAK_INIT
        self._last_r_us = 0
        self._r_time_us = 0
        # RR interval tracking (for PTT quality gating)
        self._rr_mean       = 0.0
        self._latest_rr_us  = 0
        self._prev_r_for_rr = 0
        # Startup skip
        self._skip_count = self._STARTUP_SKIP
        # PPG state
        self._ppg_dc         = 0.0
        self._ppg_prev       = 0.0
        self._ppg_prev_deriv = 0.0
        self._dc_settled     = False
        self._finger_us      = 0
        self._armed          = False
        self._peak_ac        = 0.0
        self._last_fire_us   = 0
        # PTT collection
        self._ptt_samples: list = []
        self._skip_next = True

    # ── Internal ─────────────────────────────────────────────────────────────

    def _trimmed_mean(self) -> float:
        s    = sorted(self._ptt_samples)
        mean = sum(s[1:-1]) / (len(s) - 2)   # drop min and max
        self._ptt_samples = []
        self._skip_next   = True
        return mean
