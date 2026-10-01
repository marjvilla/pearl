import os
import sys
import random
from collections import deque
from datetime import datetime
from risk_predictor import predict_risk

from PySide6.QtCore import QFile, QObject, QTimer, Signal, QThread, QDate, Qt
from PySide6.QtWidgets import (
    QApplication, QLabel, QMessageBox, QPushButton, QLineEdit, QWidget,
    QVBoxLayout, QScrollArea, QDialog, QTextEdit, QHBoxLayout, QSizePolicy,
)
from PySide6.QtUiTools import QUiLoader
from qt_material import apply_stylesheet
from PySide6.QtGui import QIntValidator
from database import DatabaseManager
from ptt_serial import (
    read_ptt_once_blocking, read_ptt_raw_blocking, ptt_to_bp,
    compute_calibration_coefficients, find_rp2040_port,
    SBP_A, SBP_B, DBP_A, DBP_B,
)

try:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    import pyqtgraph as pg
    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

STAFF_CODE        = "4321"
TEST_CODE         = "TEST0000"
PTT_TIMEOUT_SECS  = 90
NUM_PLOT_SAMPLES  = 500          # 5 s @ 100 Hz live window
HISTORY_SAMPLES   = 6000         # 60 s @ 100 Hz pause-and-scroll buffer
PTT_SESSIONS_DIR  = "ptt_sessions"


# ── Serial reader thread (used by SerialPlotterDialog) ───────────────────────

class SerialReaderThread(QThread):
    csv_data  = Signal(float, float, float, float, float)  # ECG,PPG,Rmark,Fire,Thresh
    log_line  = Signal(str)
    ptt_ready = Signal(float)
    error_sig = Signal(str)

    BAUD         = 460800
    ECG_DECIMATE = 5   # emit one csv_data row per 5 ECG samples → 100 Hz output

    def __init__(self, port_name: str):
        super().__init__()
        self._port      = port_name
        self._stop_flag = False

    def run(self):
        import serial
        from ptt_analysis import AnalysisEngine

        engine = AnalysisEngine()

        # Pending state accumulated between 100 Hz output ticks
        ecg_counter   = 0
        pending_rmark = False
        pending_fire  = False
        latest_ppg    = 0.0
        latest_ecg    = 0.0
        latest_thresh = 0.0

        try:
            with serial.Serial(self._port, self.BAUD, timeout=0.1) as ser:
                ser.write(b"START\n")

                while not self._stop_flag:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue

                    # ── STATUS / error pass-through ───────────────────────────
                    if line.startswith("STATUS:") or line.startswith("ERROR:"):
                        self.log_line.emit(line)
                        continue

                    # ── ECG sample: "E,<raw>,<us>" ────────────────────────────
                    if line.startswith("E,"):
                        parts = line.split(",")
                        if len(parts) != 3:
                            continue
                        try:
                            ecg_raw = int(parts[1])
                            ecg_us  = int(parts[2])
                        except ValueError:
                            continue

                        ecg_f, thresh, r_det = engine.feed_ecg(ecg_raw, ecg_us)
                        latest_ecg    = ecg_f
                        latest_thresh = thresh
                        if r_det:
                            pending_rmark = True

                        ecg_counter += 1
                        if ecg_counter >= self.ECG_DECIMATE:
                            ecg_counter = 0
                            rmark = 3000.0 if pending_rmark else 0.0
                            fire  = 200.0  if pending_fire  else 0.0
                            pending_rmark = False
                            pending_fire  = False
                            self.csv_data.emit(latest_ecg, latest_ppg,
                                               rmark, fire, latest_thresh)
                        continue

                    # ── PPG sample: "P,<ir>,<us>" ─────────────────────────────
                    if line.startswith("P,"):
                        parts = line.split(",")
                        if len(parts) != 3:
                            continue
                        try:
                            ir_raw = int(parts[1])
                            ppg_us = int(parts[2])
                        except ValueError:
                            continue

                        scaled_ac, fire, ptt_raw, ptt_bad, ptt_result = \
                            engine.feed_ppg(ir_raw, ppg_us)
                        latest_ppg = scaled_ac
                        if fire:
                            pending_fire = True
                        if ptt_raw is not None:
                            self.log_line.emit(f"PTTRAW:{ptt_raw:.1f}")
                        if ptt_bad is not None:
                            ptt_ms, reason = ptt_bad
                            self.log_line.emit(f"PTTBAD:{ptt_ms:.1f} ({reason})")
                        if ptt_result is not None:
                            self.log_line.emit(f"PTT:{ptt_result:.1f}")
                            self.ptt_ready.emit(ptt_result)
                        continue

                ser.write(b"STOP\n")
        except Exception as exc:
            self.error_sig.emit(str(exc))

    def stop(self):
        self._stop_flag = True


# ── Live signal plotter dialog ────────────────────────────────────────────────

class SerialPlotterDialog(QDialog):
    bp_ready      = Signal(int, int, float)  # (systolic, diastolic, median_ptt_ms) — normal mode
    ptt_captured  = Signal(float)            # raw PTT ms — calibration mode
    calib_failed  = Signal(str)              # error string — calibration mode

    def __init__(self, parent, port_name,           # port_name may be None
                 calib_mode=False, is_test=False,
                 sbp_a=SBP_A, sbp_b=SBP_B, dbp_a=DBP_A, dbp_b=DBP_B):
        super().__init__(parent)
        self._calib_mode = calib_mode
        self._is_test    = is_test
        title = "PEARL — PTT Calibration Read" if calib_mode else "PEARL — Live Signal Monitor"
        self.setWindowTitle(title)
        self.resize(920, 620)
        self._port   = port_name   # None = no device connected
        self._sbp_a  = sbp_a; self._sbp_b = sbp_b
        self._dbp_a  = dbp_a; self._dbp_b = dbp_b

        self._ptt_values:  list[float] = []
        self._log_lines:   list[str]   = []
        self._waveform_log: list[str]  = []  # CSV rows only  → raw_*.csv
        self._event_log:    list[str]  = []  # PTT/STATUS lines → included in JSON
        self._ppg_ewa   = 0.0
        self._csv_count = 0                  # elapsed_ms = csv_count × 10
        self._sample_idx = 0
        self._stopped   = False              # guard against double-stop
        self._session_ts: str | None = None  # set at flush time

        # Quality tracking for JSON output
        self._fire_count    = 0
        self._pttraw_all:   list[float] = []
        self._pttbad_reasons: dict      = {}

        # Rolling data buffers (60 s history for pause-and-scroll)
        self._ecg_buf    = deque([0.0] * HISTORY_SAMPLES, maxlen=HISTORY_SAMPLES)
        self._ppg_buf    = deque([0.0] * HISTORY_SAMPLES, maxlen=HISTORY_SAMPLES)
        self._thresh_buf = deque([0.0] * HISTORY_SAMPLES, maxlen=HISTORY_SAMPLES)
        self._paused     = False

        # Marker lists: (absolute_sample_index, y_value)
        self._rmarks: list[tuple[int, float]] = []
        self._fires:  list[tuple[int, float]] = []

        # Manual annotation marks placed by the user during pause
        # Each entry: (elapsed_ms, y_data_value, snap_x_index)
        self._manual_ecg_marks: list[tuple[int, float, int]] = []
        self._manual_ppg_marks: list[tuple[int, float, int]] = []
        self._mark_order:       list[str] = []   # 'ecg'|'ppg' in insertion order → undo stack

        # Frozen snapshot captured at the moment Pause is pressed
        self._snap_ecg:       list = []
        self._snap_ppg:       list = []
        self._snap_thresh:    list = []
        self._snap_rmarks:    list = []
        self._snap_fires:     list = []
        self._snap_buf_start: int  = 0

        self._build_ui()
        self._start_reader()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        # ECG plot
        self._ecg_plot = pg.PlotWidget(title="ECG  (bandpass 5–15 Hz)")
        self._ecg_plot.setMinimumHeight(170)
        self._ecg_plot.setXRange(0, NUM_PLOT_SAMPLES, padding=0)
        self._ecg_plot.enableAutoRange("y", True)
        self._ecg_curve     = self._ecg_plot.plot(pen=pg.mkPen("b", width=1))
        self._thresh_curve  = self._ecg_plot.plot(
            pen=pg.mkPen("orange", width=1, style=Qt.PenStyle.DashLine))
        self._rmark_scatter = pg.ScatterPlotItem(pen=None, brush=pg.mkBrush("y"), size=9)
        self._ecg_plot.addItem(self._rmark_scatter)
        # Green diamonds = user-placed manual R-wave marks
        self._man_ecg_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("g", width=2), brush=pg.mkBrush(None), size=14, symbol="d")
        self._ecg_plot.addItem(self._man_ecg_scatter)

        # PPG plot
        self._ppg_plot = pg.PlotWidget(title="PPG  (EWA α=0.5, y-axis ±10 irAC/20)")
        self._ppg_plot.setMinimumHeight(170)
        self._ppg_plot.setXRange(0, NUM_PLOT_SAMPLES, padding=0)
        self._ppg_plot.setYRange(-10, 10, padding=0)
        self._ppg_curve    = self._ppg_plot.plot(pen=pg.mkPen("g", width=1))
        self._fire_scatter = pg.ScatterPlotItem(
            pen=None, brush=pg.mkBrush("r"), size=12, symbol="t")
        self._ppg_plot.addItem(self._fire_scatter)
        # Green upward triangles = user-placed manual PPG marks
        self._man_ppg_scatter = pg.ScatterPlotItem(
            pen=pg.mkPen("g", width=2), brush=pg.mkBrush(None), size=14, symbol="t1")
        self._ppg_plot.addItem(self._man_ppg_scatter)

        # Log pane
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._log_edit.setMaximumHeight(110)
        self._log_edit.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: #1a1a1a; color: #cccccc;"
        )

        # Status
        init_msg = ("Waiting for PTT pulse… (place finger on sensor)"
                    if self._calib_mode else "Waiting for bracelet signal…")
        self._status_label = QLabel(init_msg)
        self._status_label.setAlignment(Qt.AlignCenter)
        self._status_label.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #1e88e5; padding: 4px;")

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.clicked.connect(self._toggle_pause)
        self._pause_btn.setFixedWidth(110)

        self._clear_marks_btn = QPushButton("Clear Marks")
        self._clear_marks_btn.clicked.connect(self._clear_marks)
        self._clear_marks_btn.setVisible(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._pause_btn)
        btn_row.addWidget(self._clear_marks_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)

        layout.addWidget(self._ecg_plot)
        layout.addWidget(self._ppg_plot)
        layout.addWidget(self._log_edit)
        layout.addWidget(self._status_label)
        layout.addLayout(btn_row)

        # Connect click signals — used for manual annotation during pause
        self._ecg_plot.scene().sigMouseClicked.connect(self._on_ecg_scene_click)
        self._ppg_plot.scene().sigMouseClicked.connect(self._on_ppg_scene_click)

    def _start_reader(self):
        if self._port is None:
            self._status_label.setText(
                "No PEARL bracelet found. Connect the RP2040 via USB and reopen.")
            # Still start the plot-refresh timer so the blank axes are visible
            self._plot_timer = QTimer(self)
            self._plot_timer.setInterval(50)
            self._plot_timer.timeout.connect(self._refresh_plots)
            self._plot_timer.start()
            return

        self._thread = SerialReaderThread(self._port)
        self._thread.csv_data.connect(self._on_csv)
        self._thread.log_line.connect(self._on_log)
        self._thread.ptt_ready.connect(self._on_ptt)
        self._thread.error_sig.connect(self._on_error)
        self._thread.start()

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(50)   # 20 Hz refresh
        self._plot_timer.timeout.connect(self._refresh_plots)
        self._plot_timer.start()

    def _on_csv(self, ecg, ppg, rmark, fire, thresh):
        elapsed_ms = self._csv_count * 10
        self._csv_count += 1
        self._waveform_log.append(
            f"{elapsed_ms},{ecg:.2f},{ppg:.2f},{rmark:.0f},{fire:.0f},{thresh:.1f}")
        if fire > 0:
            self._fire_count += 1
        # PPG display: EWA smoother for visual clarity (doesn't affect firmware detection)
        self._ppg_ewa = 0.5 * self._ppg_ewa + 0.5 * ppg

        self._ecg_buf.append(ecg)
        self._ppg_buf.append(self._ppg_ewa)
        self._thresh_buf.append(thresh)

        if rmark > 0:
            self._rmarks.append((self._sample_idx, ecg))
        if fire > 0:
            self._fires.append((self._sample_idx, self._ppg_ewa))

        self._sample_idx += 1

        # Trim markers beyond the history buffer
        cutoff = self._sample_idx - HISTORY_SAMPLES
        self._rmarks = [(i, y) for (i, y) in self._rmarks if i > cutoff]
        self._fires  = [(i, y) for (i, y) in self._fires  if i > cutoff]

    def _refresh_plots(self):
        if self._paused:
            return   # frozen — user is scrolling through history

        # Slice the last NUM_PLOT_SAMPLES for the live scrolling window
        ecg_sl    = list(self._ecg_buf)[-NUM_PLOT_SAMPLES:]
        thresh_sl = list(self._thresh_buf)[-NUM_PLOT_SAMPLES:]
        ppg_sl    = list(self._ppg_buf)[-NUM_PLOT_SAMPLES:]
        xs = list(range(len(ecg_sl)))

        self._ecg_curve.setData(xs, ecg_sl)
        self._thresh_curve.setData(xs, thresh_sl)
        self._ppg_curve.setData(xs, ppg_sl)
        self._ecg_plot.setXRange(0, NUM_PLOT_SAMPLES, padding=0)
        self._ppg_plot.setXRange(0, NUM_PLOT_SAMPLES, padding=0)

        start = max(0, self._sample_idx - NUM_PLOT_SAMPLES)
        vis_r = [(i, y) for (i, y) in self._rmarks if i > start]
        vis_f = [(i, y) for (i, y) in self._fires  if i > start]
        self._rmark_scatter.setData(
            x=[i - start for (i, _) in vis_r],
            y=[y          for (_, y) in vis_r])
        self._fire_scatter.setData(
            x=[i - start for (i, _) in vis_f],
            y=[y          for (_, y) in vis_f])

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_btn.setText("▶  Resume")
            self._clear_marks_btn.setVisible(True)
            # Freeze a snapshot so annotation coordinates stay stable
            # even as the live buffers keep rolling.
            self._snap_ecg       = list(self._ecg_buf)
            self._snap_ppg       = list(self._ppg_buf)
            self._snap_thresh    = list(self._thresh_buf)
            self._snap_rmarks    = list(self._rmarks)
            self._snap_fires     = list(self._fires)
            self._snap_buf_start = self._sample_idx - HISTORY_SAMPLES
            self._show_history()
            self._status_label.setText(
                "Paused — click ECG to mark R-wave  ·  click PPG to mark pulse  ·  Close to export marks")
        else:
            self._pause_btn.setText("⏸  Pause")
            self._clear_marks_btn.setVisible(False)
            self._ppg_plot.setYRange(-10, 10, padding=0)
            self._ecg_plot.enableAutoRange("y", True)

    def _show_history(self):
        """Render the frozen snapshot for pan/zoom and annotation.
        Must be called after self._snap_* fields are populated."""
        ecg    = self._snap_ecg
        thresh = self._snap_thresh
        ppg    = self._snap_ppg
        n      = len(ecg)
        xs     = list(range(n))

        self._ecg_curve.setData(xs, ecg)
        self._thresh_curve.setData(xs, thresh)
        self._ppg_curve.setData(xs, ppg)
        self._ecg_plot.enableAutoRange("y", True)
        self._ppg_plot.enableAutoRange("y", True)

        # Start at the most-recent 500 samples; user scrolls left
        self._ecg_plot.setXRange(max(0, n - NUM_PLOT_SAMPLES), n, padding=0)
        self._ppg_plot.setXRange(max(0, n - NUM_PLOT_SAMPLES), n, padding=0)

        # Auto-detected markers (from snapshot — stable even as live buffer rolls)
        buf = self._snap_buf_start
        self._rmark_scatter.setData(
            x=[i - buf for (i, _) in self._snap_rmarks],
            y=[y        for (_, y) in self._snap_rmarks])
        self._fire_scatter.setData(
            x=[i - buf for (i, _) in self._snap_fires],
            y=[y        for (_, y) in self._snap_fires])

        self._refresh_manual_markers()

    # ── Pause-mode annotation ─────────────────────────────────────────────────

    def _on_ecg_scene_click(self, event):
        """Left-click on ECG panel during pause → drop a manual R-wave mark."""
        if not self._paused or not self._snap_ecg:
            return
        pos = event.scenePos()
        if not self._ecg_plot.sceneBoundingRect().contains(pos):
            return
        pt  = self._ecg_plot.plotItem.vb.mapSceneToView(pos)
        x   = int(round(pt.x()))
        if not (0 <= x < len(self._snap_ecg)):
            return
        elapsed_ms = (self._snap_buf_start + x) * 10
        self._manual_ecg_marks.append((elapsed_ms, pt.y(), x))
        self._mark_order.append('ecg')
        self._refresh_manual_markers()
        self._status_label.setText(
            f"Paused — {len(self._manual_ecg_marks)} R-mark(s) · "
            f"{len(self._manual_ppg_marks)} PPG mark(s)  (⌫ undo · Close to export)")

    def _on_ppg_scene_click(self, event):
        """Left-click on PPG panel during pause → drop a manual PPG-fire mark."""
        if not self._paused or not self._snap_ppg:
            return
        pos = event.scenePos()
        if not self._ppg_plot.sceneBoundingRect().contains(pos):
            return
        pt  = self._ppg_plot.plotItem.vb.mapSceneToView(pos)
        x   = int(round(pt.x()))
        if not (0 <= x < len(self._snap_ppg)):
            return
        elapsed_ms = (self._snap_buf_start + x) * 10
        self._manual_ppg_marks.append((elapsed_ms, pt.y(), x))
        self._mark_order.append('ppg')
        self._refresh_manual_markers()
        self._status_label.setText(
            f"Paused — {len(self._manual_ecg_marks)} R-mark(s) · "
            f"{len(self._manual_ppg_marks)} PPG mark(s)  (⌫ undo · Close to export)")

    def _refresh_manual_markers(self):
        """Redraw the green annotation scatter items."""
        self._man_ecg_scatter.setData(
            x=[x for _, _, x in self._manual_ecg_marks],
            y=[y for _, y, _ in self._manual_ecg_marks])
        self._man_ppg_scatter.setData(
            x=[x for _, _, x in self._manual_ppg_marks],
            y=[y for _, y, _ in self._manual_ppg_marks])

    def _clear_marks(self):
        self._manual_ecg_marks.clear()
        self._manual_ppg_marks.clear()
        self._mark_order.clear()
        self._refresh_manual_markers()
        self._status_label.setText(
            "Paused — click ECG to mark R-wave  ·  click PPG to mark pulse  ·  Close to export marks")

    def keyPressEvent(self, event):
        """Backspace / Delete removes the last annotation mark while paused."""
        if self._paused and event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self._undo_last_mark()
        else:
            super().keyPressEvent(event)

    def _undo_last_mark(self):
        if not self._mark_order:
            return
        kind = self._mark_order.pop()
        if kind == 'ecg' and self._manual_ecg_marks:
            self._manual_ecg_marks.pop()
        elif kind == 'ppg' and self._manual_ppg_marks:
            self._manual_ppg_marks.pop()
        self._refresh_manual_markers()
        cnt_e, cnt_p = len(self._manual_ecg_marks), len(self._manual_ppg_marks)
        if cnt_e == 0 and cnt_p == 0:
            self._status_label.setText(
                "Paused — click ECG to mark R-wave  ·  click PPG to mark pulse  ·  Close to export marks")
        else:
            self._status_label.setText(
                f"Paused — {cnt_e} R-mark(s) · {cnt_p} PPG mark(s)  (⌫ undo · Close to export)")

    def _on_log(self, line: str):
        self._log_lines.append(line)
        self._event_log.append(line)
        self._log_edit.append(line)
        # Parse PTT events for quality stats
        if line.startswith("PTTRAW:"):
            try:
                self._pttraw_all.append(float(line[7:]))
            except ValueError:
                pass
        elif line.startswith("PTTBAD:"):
            reason = "other"
            if "(irregular RR)" in line:  reason = "irregular_rr"
            elif "(too short)"   in line:  reason = "too_short"
            elif "(too long)"    in line:  reason = "too_long"
            elif "(low amplitude)" in line: reason = "low_amplitude"
            self._pttbad_reasons[reason] = self._pttbad_reasons.get(reason, 0) + 1

    def _on_ptt(self, ptt_ms: float):
        self._ptt_values.append(ptt_ms)
        n = len(self._ptt_values)

        if self._calib_mode:
            # Calibration: one reading is enough
            self._status_label.setText(f"Captured: {ptt_ms:.1f} ms — close window to confirm.")
            self._stop_reader()
            self._flush_to_disk()
            self.ptt_captured.emit(ptt_ms)
        else:
            # Normal BP mode: collect 3 then convert
            self._status_label.setText(f"PTT reading {n}/3: {ptt_ms:.1f} ms")
            if n >= 3:
                median_ptt = sorted(self._ptt_values)[len(self._ptt_values) // 2]
                sbp, dbp = ptt_to_bp(median_ptt, self._sbp_a, self._sbp_b,
                                      self._dbp_a, self._dbp_b)
                self._status_label.setText(
                    f"Done!  BP = {sbp}/{dbp}  (median PTT {median_ptt:.1f} ms)")
                self._stop_reader()
                self._flush_to_disk()
                self.bp_ready.emit(sbp, dbp, median_ptt)

    def _on_error(self, msg: str):
        self._status_label.setText(f"Error: {msg}")
        if self._calib_mode:
            self.calib_failed.emit(msg)

    def _stop_reader(self):
        if self._stopped:
            return
        self._stopped = True
        if hasattr(self, "_plot_timer"):
            self._plot_timer.stop()
        if hasattr(self, "_thread") and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(2000)

    def _on_close(self):
        self._stop_reader()
        self._flush_to_disk()
        self.accept()

    def closeEvent(self, event):
        self._stop_reader()
        self._flush_to_disk()
        super().closeEvent(event)

    def get_quality_stats(self) -> dict:
        """Return signal quality stats for inclusion in the session JSON."""
        total = len(self._pttraw_all) + sum(self._pttbad_reasons.values())
        accepted = len(self._pttraw_all)
        return {
            "fire_count":          self._fire_count,
            "pttraw_values_ms":    [round(v, 1) for v in self._pttraw_all],
            "ptt_accepted":        accepted,
            "ptt_rejected":        total - accepted,
            "acceptance_rate_pct": round(accepted / total * 100, 1) if total > 0 else 0.0,
            "rejection_breakdown": dict(self._pttbad_reasons),
            "waveform_csv":        (
                os.path.join("test" if self._is_test else "", f"raw_{self._session_ts}.csv")
                if self._session_ts else None
            ),
        }

    def _flush_to_disk(self):
        """Write waveform CSV. Called on any exit — early close or normal completion."""
        if not self._waveform_log:
            return
        if self._session_ts:      # already flushed this session
            return
        try:
            out_dir = os.path.join(PTT_SESSIONS_DIR, "test" if self._is_test else "")
            os.makedirs(out_dir, exist_ok=True)
            self._session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(out_dir, f"raw_{self._session_ts}.csv")
            with open(csv_path, "w") as f:
                f.write("elapsed_ms,ecg_bandpass,ppg_scaled_ac,"
                        "ecg_rmark,ppg_fire,ecg_threshold\n")
                f.write("\n".join(self._waveform_log))

            # Export manual annotations (if any were placed during pause)
            if self._manual_ecg_marks or self._manual_ppg_marks:
                ann_rows = (
                    [(ms, "ECG_RMARK") for ms, _, _ in self._manual_ecg_marks] +
                    [(ms, "PPG_FIRE")  for ms, _, _ in self._manual_ppg_marks]
                )
                ann_rows.sort()
                ann_path = os.path.join(out_dir, f"annotations_{self._session_ts}.csv")
                with open(ann_path, "w") as f:
                    f.write("elapsed_ms,type\n")
                    for ms, t in ann_rows:
                        f.write(f"{ms},{t}\n")
        except Exception:
            pass


# ── Background workers ────────────────────────────────────────────────────────

class PTTWorker(QThread):
    """Headless PTT read — used when pyqtgraph is unavailable."""
    success = Signal(int, int, float)  # (systolic, diastolic, median_ptt_ms)
    error   = Signal(str)

    def __init__(self, sbp_a=SBP_A, sbp_b=SBP_B, dbp_a=DBP_A, dbp_b=DBP_B):
        super().__init__()
        self._sbp_a, self._sbp_b = sbp_a, sbp_b
        self._dbp_a, self._dbp_b = dbp_a, dbp_b

    def run(self):
        try:
            result = read_ptt_once_blocking(
                timeout_seconds=PTT_TIMEOUT_SECS,
                sbp_a=self._sbp_a, sbp_b=self._sbp_b,
                dbp_a=self._dbp_a, dbp_b=self._dbp_b,
            )
            self.success.emit(result.systolic, result.diastolic, result.ptt_ms)
        except Exception as exc:
            self.error.emit(str(exc))


class CalibPTTWorker(QThread):
    """Reads a single raw PTT value for two-point calibration."""
    success = Signal(float)
    error   = Signal(str)

    def run(self):
        try:
            ptt = read_ptt_raw_blocking(timeout_seconds=PTT_TIMEOUT_SECS)
            self.success.emit(ptt)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main application ──────────────────────────────────────────────────────────

class App:
    # ── Navigation helpers ────────────────────────────────────────────────────
    def go_to_page_2(self): self.go_to("registration_info_page_2")
    def go_to_page_3(self): self.go_to("registration_info_page_3")
    def go_to_page_4(self): self.go_to("registration_risk_page")

    def __init__(self, qt_app):
        self.app = qt_app

        loader  = QUiLoader()
        ui_file = QFile("new.ui")
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError("Could not open new.ui")
        self.window = loader.load(ui_file)
        ui_file.close()
        if self.window is None:
            raise RuntimeError("Failed to load UI")

        self.db                = DatabaseManager("patients.db")
        self.current_patient_id = None
        self.temp_registration  = {}
        self._ptt_worker        = None
        self._calib_worker      = None
        self._plotter           = None
        self._calib_ptt1        = None
        self._calib_ptt2        = None

        self.window.page1to2.clicked.connect(self.handle_registration_info_next)
        self._setup_ui()

    def _setup_ui(self):
        # Welcome page
        self.window.bigPatientButton.clicked.connect(self.open_patient_login_page)
        self.window.bigStaffButton.clicked.connect(self.open_staff_login_page)
        self.window.testModeButton.clicked.connect(self.handle_test_mode)
        self.window.terminateProgram.clicked.connect(self.app.quit)

        if hasattr(self.window, "deviceManagerButton"):
            self.window.deviceManagerButton.hide()

        # Patient login
        self.window.patientCodeInput.clear()
        self.window.patientCodeInput.setPlaceholderText("Enter Patient Code")
        self.window.patientFirstNameInput.clear()
        self.window.patientFirstNameInput.setPlaceholderText("Enter First Name")
        self.window.patientLastNameInput.clear()
        self.window.patientLastNameInput.setPlaceholderText("Enter Last Name")
        self.window.staffCodeInput.clear()
        self.window.staffCodeInput.setPlaceholderText("Enter Staff Code")
        self.window.patientSearchInput.clear()
        self.window.patientSearchInput.setPlaceholderText("Enter Patient Name")
        self.window.patientAge.setPlaceholderText("Enter Age")
        self.window.patientAge.setValidator(QIntValidator(0, 120))

        self.window.backFromPatientLoginButton.clicked.connect(self.back_from_patient_login)
        self.window.patientLoginButton.clicked.connect(self.handle_patient_login)
        self.window.dateLoginInput.setCalendarPopup(True)
        self.window.dateLoginInput.lineEdit().setReadOnly(True)

        # Registration
        self.window.page2to3.clicked.connect(self.handle_registration_page_2_next)
        self.window.page3to4.clicked.connect(self.go_to_page_4)
        self.window.cancelPatientRegistration.clicked.connect(self.cancel_registration_to_home)
        self.window.backToPatientInfoButton.clicked.connect(
            lambda: self.go_to("registration_info_page"))
        self.window.continueToRegistrationConfirmationButton.clicked.connect(
            self.complete_registration)
        self.window.continueToLoginButton.clicked.connect(self.after_registration_go_to_login)
        self.window.homeFromPatientRegistrationConfirmButton.clicked.connect(
            self.return_home_after_registration)
        self.window.registrationDOBInput.setCalendarPopup(True)
        self.window.registrationDOBInput.lineEdit().setReadOnly(True)

        # Patient home
        self.window.backFromPatientHomeButton.clicked.connect(self.logout_to_home)
        self.window.savePatientBPReadingButton.clicked.connect(self.save_patient_bp)
        self.window.startModelFromPatientButton.clicked.connect(self.run_patient_risk_assess)
        self.window.bpReadingsBluetoothButton.clicked.connect(self.start_ptt_read)
        self.window.calibrateBraceletButton.clicked.connect(self.open_calibration_page)

        # Calibration page
        self.window.calibReadPTT1Button.clicked.connect(lambda: self.start_calib_ptt(1))
        self.window.calibReadPTT2Button.clicked.connect(lambda: self.start_calib_ptt(2))
        self.window.saveCalibrationButton.clicked.connect(self.save_calibration)
        self.window.backFromCalibrationButton.clicked.connect(
            lambda: self.go_to("patient_home_page"))

        # Staff
        self.window.backFromStaffLoginButton.clicked.connect(self.open_welcome_page)
        self.window.continueToStaffPageButton.clicked.connect(self.handle_staff_login)
        self.window.pushButton.clicked.connect(self.open_welcome_page)
        self.window.patientSearchInput.returnPressed.connect(self.run_staff_search)
        self.window.backFromPatientPastInfoButton.clicked.connect(
            lambda: self.go_to("staff_dashboard_page"))
        self.window.toNewPatientRegistration.clicked.connect(self.open_registration_info_page)

        self.open_welcome_page()

    def go_to(self, page_name: str):
        page = getattr(self.window, page_name, None)
        if page is None:
            QMessageBox.warning(self.window, "Page Error",
                                f"Could not find page: {page_name}")
            return
        self.window.stackedWidget.setCurrentWidget(page)

    def generate_unique_code(self):
        while True:
            code = f"{random.randint(1000, 9999)}"
            if self.db.get_patient_by_code(code) is None:
                return code

    # ── Page navigation helpers ───────────────────────────────────────────────

    def open_welcome_page(self):      self.go_to("welcome_page")
    def open_patient_login_page(self):
        self.clear_patient_login_fields(); self.go_to("patient_login_page")
    def open_registration_info_page(self):
        self.clear_registration_fields(); self.go_to("registration_info_page")
    def open_staff_login_page(self):
        self.clear_staff_login_fields(); self.go_to("staff_login_page")
    def back_from_patient_login(self):
        self.clear_patient_login_fields(); self.go_to("welcome_page")
    def cancel_registration_to_home(self):
        self.clear_registration_fields(); self.go_to("welcome_page")
    def after_registration_go_to_login(self):
        self.clear_patient_login_fields(); self.go_to("patient_login_page")
    def return_home_after_registration(self):
        self.clear_registration_fields(); self.go_to("welcome_page")

    def logout_to_home(self):
        self.current_patient_id = None
        self.clear_patient_login_fields()
        self.go_to("welcome_page")

    # ── Field clearing ────────────────────────────────────────────────────────

    def clear_patient_login_fields(self):
        self.window.patientCodeInput.clear()
        self.window.dateLoginInput.setDate(QDate.currentDate())

    def clear_registration_fields(self):
        self.temp_registration = {}
        self.window.patientFirstNameInput.clear()
        self.window.patientLastNameInput.clear()
        self.window.patientAge.clear()
        self.window.registrationDOBInput.setDate(QDate.currentDate())
        self.window.gestationalAgeWeeksInput.setValue(0)
        self.window.gestationalAgeDaysInput.setValue(0)
        self.window.raceComboBox.setCurrentIndex(0)
        self.window.patientHeightFeet.clear()
        self.window.patientHeightInches.clear()
        for name in ["checkBox", "checkBox_2", "checkBox_3", "checkBox_4",
                     "checkBox_8", "hispanicCheck"]:
            box = getattr(self.window, name, None)
            if box:
                box.setChecked(False)

    def clear_staff_login_fields(self):
        self.window.staffCodeInput.clear()

    def clear_staff_search_fields(self):
        self.window.patientSearchInput.clear()
        self.clear_staff_search_results()

    # ── Test mode ─────────────────────────────────────────────────────────────

    def handle_test_mode(self):
        """Bypass login using the TEST0000 patient (created if absent)."""
        patient = self.db.get_patient_by_code(TEST_CODE)
        if patient is None:
            self.db.add_patient({
                "code": TEST_CODE, "first_name": "Test", "last_name": "Patient",
                "dob": "2000-01-01", "gestational_weeks": 20, "gestational_days": 0,
                "age": 25, "race": "Unknown", "height_inches": 65, "is_hispanic": 0,
            }, [])
            patient = self.db.get_patient_by_code(TEST_CODE)
        self.current_patient_id = patient["id"]
        self.load_patient_home()
        self.go_to("patient_home_page")

    # ── Registration ──────────────────────────────────────────────────────────

    def handle_registration_info_next(self):
        self.window.registrationDOBInput.clearFocus()
        first = self.window.patientFirstNameInput.text().strip()
        last  = self.window.patientLastNameInput.text().strip()
        age   = self.window.patientAge.text().strip()
        if not first or not last:
            QMessageBox.warning(self.window, "Missing Info",
                                "Please enter first and last name.")
            return
        if not age:
            QMessageBox.warning(self.window, "Missing Info", "Please enter age.")
            return
        self.temp_registration = {
            "first_name": first, "last_name": last, "age": int(age),
            "race": self.window.raceComboBox.currentText(),
        }
        self.go_to("registration_info_page_2")

    def handle_registration_page_2_next(self):
        self.temp_registration["dob"] = (
            self.window.registrationDOBInput.date().toString("yyyy-MM-dd"))
        self.temp_registration["gestational_weeks"] = (
            self.window.gestationalAgeWeeksInput.value())
        self.temp_registration["gestational_days"] = (
            self.window.gestationalAgeDaysInput.value())
        self.go_to("registration_info_page_3")

    def complete_registration(self):
        risk_checkbox_map = {
            "checkBox_2": "Pregestational Diabetes",
            "checkBox_8": "Chronic Hypertension",
            "checkBox":   "Gestational / Other Diabetes",
            "checkBox_3": "Ever Used Tobacco",
            "checkBox_4": "Smoked Tobacco in the 3 Months Prior to Pregnancy",
        }
        risk_factors = [
            label for name, label in risk_checkbox_map.items()
            if getattr(self.window, name, None) and getattr(self.window, name).isChecked()
        ]
        try:
            feet   = int(self.window.patientHeightFeet.text().strip() or 0)
            inches = int(self.window.patientHeightInches.text().strip() or 0)
        except ValueError:
            feet = inches = 0

        code = self.generate_unique_code()
        patient_dict = {
            "code":               code,
            "first_name":         self.temp_registration["first_name"],
            "last_name":          self.temp_registration["last_name"],
            "dob":                self.temp_registration["dob"],
            "gestational_weeks":  self.temp_registration["gestational_weeks"],
            "gestational_days":   self.temp_registration["gestational_days"],
            "age":                self.temp_registration["age"],
            "race":               self.temp_registration["race"],
            "height_inches":      (feet * 12) + inches,
            "is_hispanic":        1 if self.window.hispanicCheck.isChecked() else 0,
        }
        self.db.add_patient(patient_dict, risk_factors)
        self.window.registrationThanksLabel.setText(
            f"Thank you for registering {patient_dict['first_name']}!")
        self.window.giveCodeLabel.setText(f"Your Code is: {code}")
        self.clear_registration_fields()
        self.go_to("registration_confirm_page")

    # ── Patient login ─────────────────────────────────────────────────────────

    def handle_patient_login(self):
        code    = self.window.patientCodeInput.text().strip()
        dob     = self.window.dateLoginInput.date().toString("yyyy-MM-dd")
        patient = self.db.get_patient_by_code_and_dob(code, dob)
        if patient is None:
            QMessageBox.warning(self.window, "Login Failed",
                                "Patient code or date of birth is incorrect.")
            return
        self.current_patient_id = patient["id"]
        self.clear_patient_login_fields()
        self.load_patient_home()
        self.go_to("patient_home_page")

    def load_patient_home(self):
        patient  = self.db.get_patient_by_id(self.current_patient_id)
        readings = self.db.get_bp_readings(self.current_patient_id)
        self.window.patientHomeScreenNameLabel.setText(f"Hello {patient['first_name']}!")
        if readings:
            latest = readings[0]
            self.window.systolicSpinBox.setValue(latest["systolic"])
            self.window.diastolicSpinBox.setValue(latest["diastolic"])
            self.window.heartRateSpinBox.setValue(latest["gest_age_at_visit"])
            if hasattr(self.window, "inputWeight"):
                self.window.inputWeight.setValue(latest["weight"])
            self.window.patientBPRecentReadingLabel.setText(
                f"Latest: {latest['systolic']}/{latest['diastolic']}  "
                f"recorded {latest['timestamp']}"
            )
        else:
            self.window.patientBPRecentReadingLabel.setText("No BP readings yet.")

    # ── Save BP reading ───────────────────────────────────────────────────────

    def save_patient_bp(self):
        if self.current_patient_id is None:
            return
        gest_age = self.window.heartRateSpinBox.value()
        systolic = self.window.systolicSpinBox.value()
        diastolic = self.window.diastolicSpinBox.value()
        weight   = self.window.inputWeight.value()
        map_val  = round((systolic + 2 * diastolic) / 3, 1)
        now      = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        patient   = self.db.get_patient_by_id(self.current_patient_id)
        first_ga  = self.db.get_first_gest_age(self.current_patient_id)
        dob       = datetime.strptime(patient["dob"], "%Y-%m-%d")

        this_year_bday   = dob.replace(year=now.year)
        last_bday        = this_year_bday if this_year_bday <= now else dob.replace(year=now.year - 1)
        weeks_since_bday = (now - last_bday).days / 7.0
        weeks_elapsed    = (gest_age - first_ga) if first_ga is not None else 0
        age_at_last_bday = now.year - dob.year - (1 if this_year_bday > now else 0)
        visit_age        = age_at_last_bday + int((weeks_since_bday + weeks_elapsed) // 52)

        self.db.add_bp_reading(self.current_patient_id, systolic, diastolic,
                               gest_age, weight, map_val, visit_age, timestamp)
        self.load_patient_home()
        QMessageBox.information(self.window, "Success", "Data recorded!")

    # ── PTT bracelet read ─────────────────────────────────────────────────────

    def _get_calib_coeffs(self):
        """Return per-patient calibration coefficients or population defaults."""
        calib = self.db.get_calibration(self.current_patient_id)
        if calib:
            return calib
        return SBP_A, SBP_B, DBP_A, DBP_B

    def start_ptt_read(self):
        sbp_a, sbp_b, dbp_a, dbp_b = self._get_calib_coeffs()

        if HAS_PYQTGRAPH:
            # Try to find the device — pass None if not found so dialog shows error state
            port = None
            try:
                port = find_rp2040_port()
            except RuntimeError:
                pass
            is_test = (self.current_patient_id is not None and
                       self.db.get_patient_by_id(self.current_patient_id)["patient_code"] == TEST_CODE)
            self._plotter = SerialPlotterDialog(
                self.window, port,
                calib_mode=False, is_test=is_test,
                sbp_a=sbp_a, sbp_b=sbp_b, dbp_a=dbp_a, dbp_b=dbp_b)
            self._plotter.bp_ready.connect(self._on_ptt_bp_ready)
            self._plotter.show()
        else:
            # pyqtgraph not installed — run headless and warn the user
            QMessageBox.information(
                self.window, "Live Plotter Unavailable",
                "pyqtgraph is not installed — running in background mode.\n\n"
                "Install it with:  pip install pyqtgraph"
            )
            self.window.patientBPRecentReadingLabel.setText("Measuring PTT…")
            self._ptt_worker = PTTWorker(sbp_a, sbp_b, dbp_a, dbp_b)
            self._ptt_worker.success.connect(self._on_ptt_bp_ready)
            self._ptt_worker.error.connect(self._on_ptt_error)
            self._ptt_worker.start()

    def _on_ptt_bp_ready(self, sbp: int, dbp: int, median_ptt_ms: float = 0.0):
        self.window.systolicSpinBox.setValue(sbp)
        self.window.diastolicSpinBox.setValue(dbp)
        self.window.patientBPRecentReadingLabel.setText(
            f"Received: {sbp}/{dbp}. Press Save.")
        self._save_ptt_session(sbp, dbp, median_ptt_ms)

    def _on_ptt_error(self, msg: str):
        QMessageBox.warning(self.window, "PTT Read Failed", msg)

    def _save_ptt_session(self, sbp: int, dbp: int, median_ptt_ms: float):
        """Write a structured JSON session file to ptt_sessions/."""
        if self.current_patient_id is None:
            return
        try:
            import json

            patient      = self.db.get_patient_by_id(self.current_patient_id)
            is_test      = (patient["patient_code"] == TEST_CODE)
            out_dir      = os.path.join(PTT_SESSIONS_DIR, "test") if is_test else PTT_SESSIONS_DIR
            os.makedirs(out_dir, exist_ok=True)

            risk_factors = self.db.get_risk_factors(self.current_patient_id)
            calib        = self.db.get_calibration(self.current_patient_id)
            calib_pts    = self.db.get_calibration_points(self.current_patient_id)

            ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
            dt_iso = (f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                      f"T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}")

            # Quality stats from the live plotter (if available)
            quality  = {}
            plotter  = getattr(self, '_plotter', None) or getattr(self, '_calib_plotter', None)
            if plotter is not None:
                quality = plotter.get_quality_stats()

            # Calibration block
            if calib:
                sbp_a, sbp_b, dbp_a, dbp_b = calib
                calib_block = {
                    "source": "per_patient",
                    "sbp_a": round(sbp_a, 2), "sbp_b": round(sbp_b, 2),
                    "dbp_a": round(dbp_a, 2), "dbp_b": round(dbp_b, 2),
                    "points": [
                        {"point": dict(p)["point_number"],
                         "ref_sbp": dict(p)["systolic_ref"],
                         "ref_dbp": dict(p)["diastolic_ref"],
                         "ptt_ms":  round(dict(p)["ptt_ms"], 1)}
                        for p in calib_pts
                    ],
                }
            else:
                calib_block = {
                    "source": "population_defaults",
                    "sbp_a": SBP_A, "sbp_b": SBP_B,
                    "dbp_a": DBP_A, "dbp_b": DBP_B,
                }

            doc = {
                "schema_version": 1,
                "session_id":     ts,
                "is_test_session": is_test,
                "waveform_csv":   quality.pop("waveform_csv", None),
                # ── Fill this in after taking a cuff reading ───────────────
                "reference_bp": {
                    "systolic":      None,   # mmHg  ← add cuff value here
                    "diastolic":     None,
                    "measured_with": "manual_cuff"
                },
                "patient": {
                    "code":               patient["patient_code"],
                    "first_name":         patient["first_name"],
                    "last_name":          patient["last_name"],
                    "dob":                patient["dob"],
                    "age":                patient["age"],
                    "race":               patient["race"],
                    "is_hispanic":        bool(patient["is_hispanic"]),
                    "height_inches":      patient["height_inches"],
                    "gestational_weeks":  patient["gestational_weeks"],
                    "gestational_days":   patient["gestational_days"],
                    "risk_factors":       risk_factors,
                },
                "calibration": calib_block,
                "measurement": {
                    "timestamp":    dt_iso,
                    "mode":         "bp_measurement",
                    "median_ptt_ms": round(median_ptt_ms, 1),
                    "estimated_sbp": sbp,
                    "estimated_dbp": dbp,
                    "pttraw_values_ms": quality.pop("pttraw_values_ms", []),
                },
                "quality": quality,
            }

            fname = os.path.join(out_dir, f"session_{ts}.json")
            with open(fname, "w") as f:
                json.dump(doc, f, indent=2)
        except Exception:
            pass   # never crash the clinical UI over a log write

    # ── Calibration page ──────────────────────────────────────────────────────

    def open_calibration_page(self):
        self._calib_ptt1 = None
        self._calib_ptt2 = None
        self.window.calibPTT1ResultLabel.setText("PTT: ---")
        self.window.calibPTT2ResultLabel.setText("PTT: ---")
        self.window.saveCalibrationButton.setEnabled(False)
        self.window.calibStatusLabel.setText(
            "Enter the BP from a reference cuff, then press Take PTT. Do this twice.")
        self.go_to("calibration_page")

    def start_calib_ptt(self, point: int):
        if getattr(self, '_calib_plotter', None) and self._calib_plotter.isVisible():
            QMessageBox.information(self.window, "Busy",
                                    "A PTT measurement is already in progress.")
            return

        btn = (self.window.calibReadPTT1Button if point == 1
               else self.window.calibReadPTT2Button)
        lbl = (self.window.calibPTT1ResultLabel if point == 1
               else self.window.calibPTT2ResultLabel)
        btn.setEnabled(False)
        lbl.setText("PTT: measuring…")
        self.window.calibStatusLabel.setText(
            f"Reading {point} — place finger on sensor, then watch the signal.")

        if HAS_PYQTGRAPH:
            port = None
            try:
                port = find_rp2040_port()
            except RuntimeError:
                pass
            is_test = (self.current_patient_id is not None and
                       self.db.get_patient_by_id(self.current_patient_id)["patient_code"] == TEST_CODE)
            self._calib_plotter = SerialPlotterDialog(
                self.window, port, calib_mode=True, is_test=is_test)
            self._calib_plotter.ptt_captured.connect(
                lambda ptt, p=point, b=btn, l=lbl: self._on_calib_success(ptt, p, b, l))
            self._calib_plotter.calib_failed.connect(
                lambda msg, b=btn, l=lbl: self._on_calib_error(msg, b, l))
            self._calib_plotter.show()
        else:
            # Headless fallback when pyqtgraph is not installed
            self._calib_worker = CalibPTTWorker()
            self._calib_worker.success.connect(
                lambda ptt, p=point, b=btn, l=lbl: self._on_calib_success(ptt, p, b, l))
            self._calib_worker.error.connect(
                lambda msg, b=btn, l=lbl: self._on_calib_error(msg, b, l))
            self._calib_worker.start()

    def _on_calib_success(self, ptt: float, point: int, btn, lbl):
        if point == 1:
            self._calib_ptt1 = ptt
        else:
            self._calib_ptt2 = ptt
        lbl.setText(f"PTT: {ptt:.1f} ms")
        btn.setEnabled(True)
        self.window.calibStatusLabel.setText(
            f"Reading {point} captured: {ptt:.1f} ms")
        if self._calib_ptt1 is not None and self._calib_ptt2 is not None:
            self.window.saveCalibrationButton.setEnabled(True)

    def _on_calib_error(self, msg: str, btn, lbl):
        lbl.setText("PTT: failed")
        btn.setEnabled(True)
        QMessageBox.warning(self.window, "PTT Calibration Error", msg)

    def save_calibration(self):
        sbp1 = self.window.calibSBP1.value()
        dbp1 = self.window.calibDBP1.value()
        sbp2 = self.window.calibSBP2.value()
        dbp2 = self.window.calibDBP2.value()

        coeffs = compute_calibration_coefficients(
            sbp1, dbp1, self._calib_ptt1,
            sbp2, dbp2, self._calib_ptt2,
        )
        if coeffs is None:
            QMessageBox.warning(
                self.window, "Calibration Error",
                "The two readings are too similar.\n"
                "Please retake with a greater difference in blood pressure."
            )
            return

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.db.save_calibration_point(
            self.current_patient_id, 1, sbp1, dbp1, self._calib_ptt1, now)
        self.db.save_calibration_point(
            self.current_patient_id, 2, sbp2, dbp2, self._calib_ptt2, now)

        QMessageBox.information(self.window, "Calibration Saved",
                                "PTT calibration saved successfully.")
        self.go_to("patient_home_page")

    # ── Risk assessment ───────────────────────────────────────────────────────

    def run_patient_risk_assess(self):
        if self.current_patient_id is None:
            QMessageBox.warning(self.window, "Error", "No patient loaded.")
            return
        patient     = self.db.get_patient_by_id(self.current_patient_id)
        risk_factors = self.db.get_risk_factors(self.current_patient_id)
        bp_readings  = self.db.get_bp_readings(self.current_patient_id)

        if not bp_readings:
            QMessageBox.warning(self.window, "No BP Data",
                                "Please save at least one BP reading before running "
                                "the risk assessment.")
            return

        try:
            result = predict_risk(patient, risk_factors, bp_readings)
        except FileNotFoundError as exc:
            QMessageBox.critical(self.window, "Model Files Missing",
                                 f"Could not find a required model file:\n{exc}")
            return
        except Exception as exc:
            QMessageBox.critical(self.window, "Prediction Error", str(exc))
            return

        visits = result["visits_used"]
        prob   = result["probability"]
        label  = result["label"]
        top_features = result.get("top_3_features", [])

        icon   = QMessageBox.Warning if label == "High Risk" else QMessageBox.Information
        detail = {
            "High Risk":     ("This patient's readings are consistent with elevated "
                              "preeclampsia risk.\nPlease escalate to clinical review."),
            "Moderate Risk": ("This patient's readings suggest moderate preeclampsia risk.\n"
                              "Continue close monitoring and consider follow-up."),
        }.get(label, "Current readings do not indicate elevated risk.\n"
                      "Continue routine monitoring.")

        data_note = ""
        if visits == 1:
            data_note = "\n\n⚠️  Only 1 visit recorded. Results improve with 2–3 visits."
        elif visits == 2:
            data_note = "\n\nℹ️  2 visits recorded. A 3rd reading will improve accuracy."

        top_text = ""
        if top_features:
            top_text = "\n\nTop factors in this prediction:\n"
            for item in top_features:
                top_text += f"- {item.get('feature', '?')} ({item.get('direction', '')})\n"

        msg = QMessageBox(self.window)
        msg.setIcon(icon)
        msg.setWindowTitle("Preeclampsia Risk Assessment")
        msg.setText(f"Result: {label}")
        msg.setInformativeText(
            f"Probability: {prob:.1f}%\nVisits used: {visits}\n\n"
            f"{detail}{data_note}{top_text}"
        )
        msg.exec()

    # ── Staff ─────────────────────────────────────────────────────────────────

    def handle_staff_login(self):
        if self.window.staffCodeInput.text().strip() != STAFF_CODE:
            QMessageBox.warning(self.window, "Login Failed", "Incorrect staff code.")
            return
        self.clear_staff_login_fields()
        self.go_to("staff_dashboard_page")

    def clear_staff_search_results(self):
        layout = self.window.widgetForStaffSearch.layout()
        if layout:
            while layout.count():
                w = layout.takeAt(0).widget()
                if w:
                    w.deleteLater()

    def run_staff_search(self):
        query = self.window.patientSearchInput.text().strip()
        self.clear_staff_search_results()
        layout = self.window.widgetForStaffSearch.layout()
        if not query or not layout:
            return
        matches = self.db.search_patients_by_name(query)
        if not matches:
            layout.addWidget(QLabel("No patients found."))
            return
        for patient in matches:
            btn = QPushButton(f"{patient['first_name']} {patient['last_name']}")
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 32px; font-weight: bold; color: #1e88e5;
                    background-color: white; border: 2px solid #1e88e5;
                    border-radius: 10px; padding: 20px; margin: 5px;
                }
            """)
            btn.setMinimumHeight(100)
            btn.clicked.connect(
                lambda checked=False, pid=patient["id"]: self.open_staff_patient_detail(pid))
            layout.addWidget(btn)

    def open_staff_patient_detail(self, patient_id: int):
        patient  = self.db.get_patient_by_id(patient_id)
        readings = self.db.get_bp_readings(patient_id)
        self.window.patientDetailTitleLabel.setText(
            f"{patient['first_name']} {patient['last_name']} Info")
        self.window.patientGeneralInfoLabel.setText(
            f"Code: {patient['patient_code']}\nDOB: {patient['dob']}")
        history = (
            "\n".join(
                f"{r['timestamp']}: {r['systolic']}/{r['diastolic']} "
                f"(MAP: {r['map_value']})"
                for r in readings
            ) if readings else "No readings."
        )
        self.window.pastBPReadingsLabel.setText(history)
        self.go_to("staff_patient_detail_page")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.environ["QT_SCALE_FACTOR"]            = "1.0"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"

    qt_app = QApplication(sys.argv)
    apply_stylesheet(qt_app, theme="light_blue.xml", extra={"density_selection": "-3"})

    custom_styles = """
        QPushButton#staffSearchButton, QPushButton#goBackFromStaffSearchButton {
            height: 30px; max-height: 30px; min-height: 30px;
            padding: 2px 10px; margin: 2px;
            border-radius: 6px; font-size: 10pt; font-weight: bold;
        }
        QPushButton#staffSearchButton      { background-color: #2c3e50; color: white; }
        QPushButton#goBackFromStaffSearchButton { background-color: #7f8c8d; color: white; }
        * { font-size: 10pt; }
    """
    qt_app.setStyleSheet(qt_app.styleSheet() + custom_styles)

    app = App(qt_app)
    app.window.setFixedSize(1024, 600)
    app.window.show()
    sys.exit(qt_app.exec())
