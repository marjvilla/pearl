"""
ptt_serial.py
-------------
Serial communication with the PEARL RP2040 wristband.

Auto-detects the device by USB VID 0x2E8A, falls back to usbmodem/ttyACM.

Public API
----------
find_rp2040_port()                        → port device string
ptt_to_bp(ptt_ms, ...)                    → (systolic, diastolic)
compute_calibration_coefficients(...)     → (sbp_a, sbp_b, dbp_a, dbp_b) | None
read_ptt_raw_blocking(timeout_seconds)    → float  (single PTT in ms)
read_ptt_once_blocking(timeout_seconds, sbp_a, sbp_b, dbp_a, dbp_b)
                                          → BPResult(systolic, diastolic, heart_rate=0)
"""

import math
import time
import serial
import serial.tools.list_ports
from dataclasses import dataclass

# ── Hardware constants ────────────────────────────────────────────────────────
RP2040_VID  = 0x2E8A
BAUD        = 460800
NUM_SAMPLES = 3       # PTT: lines to collect per BP measurement

# ── Population-level Moens-Korteweg coefficients ─────────────────────────────
# BP = A - B × ln(PTT_ms)
SBP_A = 517.7
SBP_B = 71.4
DBP_A = 340.0
DBP_B = 46.0


@dataclass
class BPResult:
    systolic:   int
    diastolic:  int
    heart_rate: int   = 0
    ptt_ms:     float = 0.0


# ── Port detection ────────────────────────────────────────────────────────────

def find_rp2040_port() -> str:
    """Return the first RP2040 serial port device string, or raise RuntimeError."""
    for p in serial.tools.list_ports.comports():
        if p.vid == RP2040_VID:
            return p.device
    for p in serial.tools.list_ports.comports():
        d = p.device.lower()
        if "usbmodem" in d or "ttyacm" in d:
            return p.device
    raise RuntimeError(
        "PEARL bracelet not found.\n"
        "Check that the RP2040 wristband is connected via USB."
    )


# ── BP conversion ─────────────────────────────────────────────────────────────

def ptt_to_bp(ptt_ms: float,
              sbp_a: float = SBP_A, sbp_b: float = SBP_B,
              dbp_a: float = DBP_A, dbp_b: float = DBP_B) -> tuple:
    """Moens-Korteweg model: BP = A - B × ln(PTT_ms)."""
    ln_ptt = math.log(ptt_ms)
    return round(sbp_a - sbp_b * ln_ptt), round(dbp_a - dbp_b * ln_ptt)


def compute_calibration_coefficients(sbp1: int, dbp1: int, ptt1: float,
                                     sbp2: int, dbp2: int, ptt2: float):
    """
    Two-point fit of BP = A − B × ln(PTT).

    SBP_B = (SBP1 − SBP2) / (ln(PTT2) − ln(PTT1))
    SBP_A = SBP1 + SBP_B × ln(PTT1)

    Returns (sbp_a, sbp_b, dbp_a, dbp_b) or None if points are too close.
    """
    try:
        ln1, ln2 = math.log(ptt1), math.log(ptt2)
        d_ln = ln2 - ln1
        if abs(d_ln) < 0.01:
            return None
        sbp_b = (sbp1 - sbp2) / d_ln
        sbp_a = sbp1 + sbp_b * ln1
        dbp_b = (dbp1 - dbp2) / d_ln
        dbp_a = dbp1 + dbp_b * ln1
        return float(sbp_a), float(sbp_b), float(dbp_a), float(dbp_b)
    except (ValueError, ZeroDivisionError):
        return None


# ── Blocking read helpers ─────────────────────────────────────────────────────

def _open_port(port_name: str) -> serial.Serial:
    return serial.Serial(port_name, BAUD, timeout=1.0)


def _stream_until_ptt(ser, deadline: float, n_results: int = 1) -> list[float]:
    """
    Internal helper: drive AnalysisEngine on the raw E/P stream until
    n_results trimmed-mean PTT values have been produced, or deadline passes.
    """
    from ptt_analysis import AnalysisEngine
    engine  = AnalysisEngine()
    results = []
    while time.time() < deadline and len(results) < n_results:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("E,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    engine.feed_ecg(int(parts[1]), int(parts[2]))
                except ValueError:
                    pass
        elif line.startswith("P,"):
            parts = line.split(",")
            if len(parts) == 3:
                try:
                    _, _, _, _, ptt_result = engine.feed_ppg(int(parts[1]), int(parts[2]))
                    if ptt_result is not None:
                        results.append(ptt_result)
                except ValueError:
                    pass
    return results


def read_ptt_raw_blocking(timeout_seconds: float = 90.0) -> float:
    """
    Send START, wait for the first trimmed-mean PTT, send STOP.
    Returns PTT in ms.  Used for calibration.
    """
    port     = find_rp2040_port()
    deadline = time.time() + timeout_seconds
    with _open_port(port) as ser:
        ser.write(b"START\n")
        results = _stream_until_ptt(ser, deadline, n_results=1)
        ser.write(b"STOP\n")
    if not results:
        raise TimeoutError("No PTT reading received within timeout.")
    return results[0]


def read_ptt_once_blocking(timeout_seconds: float = 90.0,
                           sbp_a: float = SBP_A, sbp_b: float = SBP_B,
                           dbp_a: float = DBP_A, dbp_b: float = DBP_B) -> BPResult:
    """
    Send START, collect NUM_SAMPLES trimmed-mean PTTs, send STOP.
    Returns BPResult with systolic/diastolic from the median of collected PTTs.
    """
    port     = find_rp2040_port()
    deadline = time.time() + timeout_seconds
    with _open_port(port) as ser:
        ser.write(b"START\n")
        ptt_values = _stream_until_ptt(ser, deadline, n_results=NUM_SAMPLES)
        ser.write(b"STOP\n")

    if not ptt_values:
        raise TimeoutError("No PTT readings received within timeout.")

    median_ptt = sorted(ptt_values)[len(ptt_values) // 2]
    sbp, dbp = ptt_to_bp(median_ptt, sbp_a, sbp_b, dbp_a, dbp_b)
    return BPResult(systolic=sbp, diastolic=dbp, heart_rate=0, ptt_ms=median_ptt)
