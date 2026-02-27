"""Live training metrics logging and dashboard generation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _coerce_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return str(value)


class LiveMetricsCallback(TrainerCallback):
    """Persist live metrics for dashboard refresh during training."""

    def __init__(
        self,
        output_dir: str | Path,
        metrics_filename: str = "training_live_metrics.json",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.metrics_path = self.output_dir / metrics_filename
        self.status = "idle"
        self.started_at: str | None = None
        self.ended_at: str | None = None
        self._train_start_monotonic: float | None = None
        self._current_epoch: int | None = None
        self._epoch_start_monotonic: float | None = None
        self.log_history: list[dict[str, Any]] = []
        self.epoch_timings: list[dict[str, Any]] = []

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.status = "running"
        self.started_at = _utc_now_iso()
        self.ended_at = None
        self._train_start_monotonic = time.perf_counter()
        self.log_history = []
        self.epoch_timings = []
        self._write_payload(state)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._current_epoch = int(float(state.epoch or 0.0)) + 1
        self._epoch_start_monotonic = time.perf_counter()
        self._upsert_epoch(
            epoch=self._current_epoch,
            started_at=_utc_now_iso(),
        )
        self._write_payload(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        entry = {
            "timestamp": _utc_now_iso(),
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
        }
        for key, value in logs.items():
            entry[key] = _coerce_scalar(value)
        self.log_history.append(entry)
        # Keep the file compact for frequent polling.
        if len(self.log_history) > 2000:
            self.log_history = self.log_history[-2000:]
        self._write_payload(state)

    def on_epoch_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        if self._epoch_start_monotonic is None:
            return
        epoch = int(round(float(state.epoch or self._current_epoch or 0.0)))
        if epoch < 1:
            epoch = self._current_epoch or 1
        duration = time.perf_counter() - self._epoch_start_monotonic
        self._upsert_epoch(
            epoch=epoch,
            ended_at=_utc_now_iso(),
            duration_seconds=round(duration, 3),
        )
        self._epoch_start_monotonic = None
        self._write_payload(state)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self.status = "completed"
        self.ended_at = _utc_now_iso()
        self._write_payload(state)

    def _upsert_epoch(self, epoch: int, **updates: Any) -> None:
        for record in self.epoch_timings:
            if int(record.get("epoch", -1)) == epoch:
                record.update(updates)
                return
        record = {"epoch": epoch}
        record.update(updates)
        self.epoch_timings.append(record)
        self.epoch_timings.sort(key=lambda item: int(item["epoch"]))

    def _write_payload(self, state) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = _utc_now_iso()
        runtime = None
        if self._train_start_monotonic is not None:
            runtime = round(time.perf_counter() - self._train_start_monotonic, 3)
        current_epoch_elapsed = None
        if self._epoch_start_monotonic is not None:
            current_epoch_elapsed = round(time.perf_counter() - self._epoch_start_monotonic, 3)

        payload = {
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "updated_at": now,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "train_runtime_seconds": runtime,
            "current_epoch_elapsed_seconds": current_epoch_elapsed,
            "epoch_timings": self.epoch_timings,
            "log_history": self.log_history,
        }
        tmp_path = self.metrics_path.with_suffix(self.metrics_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(self.metrics_path)


def write_live_dashboard(
    output_dir: str | Path,
    dashboard_filename: str = "training_dashboard.html",
    metrics_filename: str = "training_live_metrics.json",
) -> Path:
    """Write a no-dependency auto-refresh dashboard page."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_path / dashboard_filename
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DetectAnyLLM Training Dashboard</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --card: #ffffff;
      --text: #111827;
      --muted: #6b7280;
      --line: #d1d5db;
      --accent: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--text);
      background: radial-gradient(circle at top left, #eef2ff 0%, var(--bg) 45%);
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .muted {{ color: var(--muted); font-size: 14px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin: 16px 0 20px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 3px 14px rgba(0, 0, 0, 0.05);
    }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .value {{ font-size: 22px; margin-top: 4px; }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      margin-top: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid #e5e7eb;
      font-size: 14px;
    }}
    canvas {{
      width: 100%;
      max-width: 100%;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fcfdff;
      margin-top: 8px;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>DetectAnyLLM Training Dashboard</h1>
  <div class="muted">Auto-refresh every 2s · metrics source: <code>{metrics_filename}</code></div>

  <div class="grid">
    <div class="card"><div class="label">Status</div><div class="value" id="status">-</div></div>
    <div class="card"><div class="label">Current Epoch</div><div class="value" id="epoch">-</div></div>
    <div class="card"><div class="label">Global Step</div><div class="value" id="step">-</div></div>
    <div class="card"><div class="label">Train Runtime (s)</div><div class="value" id="runtime">-</div></div>
  </div>

  <div class="panel">
    <div class="label">Epoch Timings</div>
    <table>
      <thead><tr><th>Epoch</th><th>Duration (s)</th><th>Started</th><th>Ended</th></tr></thead>
      <tbody id="epochRows"></tbody>
    </table>
  </div>

  <div class="panel">
    <div class="label">Loss vs Step</div>
    <canvas id="lossChart" width="980" height="280"></canvas>
  </div>

  <div class="panel">
    <div class="label">Grad Norm vs Step</div>
    <canvas id="gradChart" width="980" height="280"></canvas>
  </div>

  <div class="panel">
    <div class="label">DDL Metrics (mean_d_h / mean_d_m)</div>
    <canvas id="ddlChart" width="980" height="280"></canvas>
  </div>

  <script>
    const METRICS_URL = "{metrics_filename}";
    const REFRESH_MS = 2000;

    function toFixed(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(3);
    }}

    function text(id, value) {{
      document.getElementById(id).textContent = value;
    }}

    function drawLine(canvasId, series) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const allPoints = series.flatMap(s => s.points);
      if (allPoints.length === 0) {{
        ctx.fillStyle = "#6b7280";
        ctx.font = "14px sans-serif";
        ctx.fillText("No data yet", 18, 30);
        return;
      }}

      let minX = Math.min(...allPoints.map(p => p.x));
      let maxX = Math.max(...allPoints.map(p => p.x));
      let minY = Math.min(...allPoints.map(p => p.y));
      let maxY = Math.max(...allPoints.map(p => p.y));
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) {{
        minY -= 1;
        maxY += 1;
      }}

      const padL = 52, padR = 20, padT = 18, padB = 34;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;
      const sx = x => padL + ((x - minX) / (maxX - minX)) * plotW;
      const sy = y => padT + (1 - (y - minY) / (maxY - minY)) * plotH;

      ctx.strokeStyle = "#e5e7eb";
      ctx.lineWidth = 1;
      ctx.strokeRect(padL, padT, plotW, plotH);

      ctx.fillStyle = "#6b7280";
      ctx.font = "12px sans-serif";
      ctx.fillText(minY.toFixed(3), 8, sy(minY) + 4);
      ctx.fillText(maxY.toFixed(3), 8, sy(maxY) + 4);
      ctx.fillText(String(minX), sx(minX) - 6, H - 10);
      ctx.fillText(String(maxX), sx(maxX) - 6, H - 10);

      series.forEach((line, idx) => {{
        if (line.points.length === 0) return;
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        line.points.forEach((p, i) => {{
          const x = sx(p.x);
          const y = sy(p.y);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }});
        ctx.stroke();

        const legendX = padL + idx * 180;
        ctx.fillStyle = line.color;
        ctx.fillRect(legendX, 4, 12, 12);
        ctx.fillStyle = "#374151";
        ctx.fillText(line.label, legendX + 18, 14);
      }});
    }}

    function renderEpochRows(rows) {{
      const tbody = document.getElementById("epochRows");
      tbody.innerHTML = "";
      if (!rows || rows.length === 0) {{
        tbody.innerHTML = "<tr><td colspan='4' class='muted'>No epoch data yet</td></tr>";
        return;
      }}
      for (const row of rows) {{
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + row.epoch + "</td>" +
          "<td>" + (row.duration_seconds !== undefined ? Number(row.duration_seconds).toFixed(3) : "-") + "</td>" +
          "<td>" + (row.started_at || "-") + "</td>" +
          "<td>" + (row.ended_at || "-") + "</td>";
        tbody.appendChild(tr);
      }}
    }}

    function points(history, key) {{
      return history
        .filter(h => Number.isFinite(Number(h[key])) && Number.isFinite(Number(h.global_step)))
        .map(h => ({{ x: Number(h.global_step), y: Number(h[key]) }}));
    }}

    async function refresh() {{
      try {{
        const res = await fetch(METRICS_URL + "?t=" + Date.now(), {{ cache: "no-store" }});
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();

        text("status", data.status || "-");
        text("epoch", data.epoch !== null && data.epoch !== undefined ? Number(data.epoch).toFixed(3) : "-");
        text("step", data.global_step !== undefined ? String(data.global_step) : "-");
        text("runtime", toFixed(data.train_runtime_seconds));
        renderEpochRows(data.epoch_timings || []);

        const history = data.log_history || [];
        drawLine("lossChart", [
          {{ label: "loss", color: "#d9480f", points: points(history, "loss") }},
          {{ label: "train_loss", color: "#c2410c", points: points(history, "train_loss") }},
        ]);
        drawLine("gradChart", [
          {{ label: "grad_norm", color: "#0284c7", points: points(history, "grad_norm") }},
        ]);
        drawLine("ddlChart", [
          {{ label: "mean_d_h", color: "#0f766e", points: points(history, "mean_d_h") }},
          {{ label: "mean_d_m", color: "#7c3aed", points: points(history, "mean_d_m") }},
        ]);
      }} catch (err) {{
        text("status", "load error");
      }}
    }}

    refresh();
    setInterval(refresh, REFRESH_MS);
  </script>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")
    return dashboard_path

