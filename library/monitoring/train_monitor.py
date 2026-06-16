"""
训练监控服务器
实时显示 loss 曲线和采样图片
"""

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
from urllib.parse import urlparse, parse_qs

# 全局状态
MONITOR_STATE = {
    "losses": [],
    "val_losses": [],  # validation passes (CMMD or FM-MSE average) — sparse, plotted as a 2nd curve
    "lr_history": [],
    "epoch": 0,
    "step": 0,
    "total_steps": 0,
    "speed": 0.0,
    "samples": [],
    "start_time": None,
    "config": {},
}

MONITOR_DIR = Path(__file__).resolve().parent / "monitor_data"
MONITOR_DIR.mkdir(exist_ok=True)

# MONITOR_STATE is mutated by the training thread (update_monitor) and read by the
# HTTP daemon thread (get_state). Guard every access with a re-entrant lock so a
# concurrent read can't observe a half-updated list (or trip "list changed size
# during iteration" while the server downsamples). RLock because update_monitor /
# restore_monitor_state call save_state() while already holding the lock.
_LOCK = threading.RLock()

# Disk-write throttle: update_monitor() fires every logged step, and save_state()
# json.dumps()-es the WHOLE state (up to 50k loss + 50k lr points) under the lock.
# At every-step cadence that grows O(n) and the long lock-hold starves the HTTP
# poll thread → the dashboard flips to "Offline" mid-run. Throttle the per-step
# disk writes to ~1/s (loss/lr only); samples/config/resume force an immediate
# write so nothing important is lost. Bounded by the in-memory 50k cap either way.
_SAVE_MIN_INTERVAL = 1.0
_LAST_SAVE_T = 0.0
# Cap the points handed to /api/state so the per-poll copy + json.dumps stays O(1)
# in wall-clock regardless of run length (the chart can't resolve more anyway).
_STATE_MAX_POINTS = 5000


def update_monitor(
    loss=None,
    lr=None,
    epoch=None,
    step=None,
    total_steps=None,
    speed=None,
    sample_path=None,
    config=None,
    val_loss=None,
):
    """更新监控状态"""
    with _LOCK:
        # 先更新 step/epoch 等，使本次写入的 loss/lr 点位正确
        if epoch is not None:
            MONITOR_STATE["epoch"] = epoch
        if step is not None:
            MONITOR_STATE["step"] = step
        if total_steps is not None:
            MONITOR_STATE["total_steps"] = total_steps
        if speed is not None:
            MONITOR_STATE["speed"] = speed

        if loss is not None:
            MONITOR_STATE["losses"].append(
                {"step": MONITOR_STATE["step"], "loss": loss, "time": time.time()}
            )
            # 保留最近 50000 个点（支持长时间训练）
            if len(MONITOR_STATE["losses"]) > 50000:
                MONITOR_STATE["losses"] = MONITOR_STATE["losses"][-50000:]

        # Validation pass (CMMD or FM-MSE average) — sparse points, own series so the
        # dashboard can overlay it on the loss chart. The step used is whatever the
        # validation log carried (the global_step at the val pass).
        if val_loss is not None:
            MONITOR_STATE["val_losses"].append(
                {"step": MONITOR_STATE["step"], "loss": val_loss, "time": time.time()}
            )
            if len(MONITOR_STATE["val_losses"]) > 50000:
                MONITOR_STATE["val_losses"] = MONITOR_STATE["val_losses"][-50000:]

        if lr is not None:
            MONITOR_STATE["lr_history"].append(
                {"step": MONITOR_STATE["step"], "lr": lr}
            )
            if len(MONITOR_STATE["lr_history"]) > 50000:
                MONITOR_STATE["lr_history"] = MONITOR_STATE["lr_history"][-50000:]
        if sample_path is not None:
            MONITOR_STATE["samples"].append(
                {
                    "path": str(sample_path),
                    "step": MONITOR_STATE["step"],
                    "time": time.time(),
                }
            )
            # 只保留最近 50 张
            if len(MONITOR_STATE["samples"]) > 50:
                MONITOR_STATE["samples"] = MONITOR_STATE["samples"][-50:]
        if config is not None:
            MONITOR_STATE["config"] = config

        if MONITOR_STATE["start_time"] is None:
            MONITOR_STATE["start_time"] = time.time()

        # 写入 JSON 文件 — per-step writes are throttled; a new sample or config
        # change forces an immediate persist (infrequent + worth not losing).
        save_state(force=(sample_path is not None or config is not None))


def save_state(force=False):
    """保存状态到 JSON. ``force`` bypasses the per-step throttle (resume / sample /
    config writes); the hot per-step path is throttled to ~1/s so json.dumps of the
    growing history doesn't run every step under the lock."""
    global _LAST_SAVE_T
    state_file = MONITOR_DIR / "state.json"
    try:
        # Snapshot under the lock for consistency (and to guard _LAST_SAVE_T); the
        # file write happens outside it to keep the lock-hold time short. When the
        # throttle skips a write we never reach json.dumps at all — that's the win.
        with _LOCK:
            now = time.time()
            if not force and (now - _LAST_SAVE_T) < _SAVE_MIN_INTERVAL:
                return
            _LAST_SAVE_T = now
            payload = json.dumps(MONITOR_STATE)
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(payload)
    except Exception:
        pass


def save_run_snapshot(output_dir, run_name=None):
    """Archive the finished run as a portable bundle under ``<output_dir>/runs/
    <run>_<timestamp>/`` — ``state.json`` (full history), ``meta.json`` (summary),
    and a copy of the referenced ``sample/*.png`` — so it can be re-opened offline
    and compared against other runs even after ``<output_dir>/sample`` is cleared.
    Returns the archive path (or None on failure)."""
    try:
        state = get_state(0)  # full, un-downsampled history
        cfg = state.get("config") or {}
        run = run_name or cfg.get("run") or "run"
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(run))
        dest = Path(output_dir) / "runs" / f"{safe}_{ts}"
        sdir = dest / "samples"
        sdir.mkdir(parents=True, exist_ok=True)
        src_sample = Path(output_dir) / "sample"
        for s in state.get("samples") or []:
            fn = Path(str(s.get("path", ""))).name
            src = src_sample / fn
            if fn and src.exists():
                try:
                    shutil.copy2(src, sdir / fn)
                except OSError:
                    pass
        losses = state.get("losses") or []
        meta = {
            "run": run,
            "saved_at": ts,
            "step": state.get("step"),
            "epoch": state.get("epoch"),
            "total_steps": state.get("total_steps"),
            "final_loss": losses[-1].get("loss") if losses else None,
            "config": cfg,
        }
        (dest / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (dest / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        print(f"[Monitor] saved run snapshot → {dest}")
        return str(dest)
    except Exception as exc:  # never break training teardown
        print(f"[Monitor] snapshot failed: {exc}")
        return None


def get_state(max_points: int = _STATE_MAX_POINTS):
    """获取当前状态（线程安全快照：复制顶层列表，避免读取时被并发追加）.

    Copies the top-level lists under the lock (cheap ref-copy), then downsamples
    the loss/lr curves to ``max_points`` OUTSIDE the lock so a /api/state poll on a
    long run can't hold the lock through a 50k-element json.dumps. Resume reads the
    full on-disk state.json instead, so capping the live view loses no history."""
    with _LOCK:
        snapshot = MONITOR_STATE.copy()
        losses = list(MONITOR_STATE["losses"])
        lr_history = list(MONITOR_STATE["lr_history"])
        val_losses = list(MONITOR_STATE.get("val_losses", []))
        snapshot["samples"] = list(MONITOR_STATE["samples"])
    if max_points and len(losses) > max_points:
        losses = _downsample_uniform(losses, max_points)
    if max_points and len(lr_history) > max_points:
        lr_history = _downsample_uniform(lr_history, max_points)
    if max_points and len(val_losses) > max_points:
        val_losses = _downsample_uniform(val_losses, max_points)
    snapshot["losses"] = losses
    snapshot["lr_history"] = lr_history
    snapshot["val_losses"] = val_losses
    return snapshot


def restore_monitor_state(
    losses=None,
    lr_history=None,
    epoch=None,
    step=None,
    total_steps=None,
    start_time=None,
    config=None,
    val_losses=None,
):
    """恢复监控状态（用于断点续训）

    Args:
        losses: 历史 loss 列表，格式 [{"step": int, "loss": float, "time": float}, ...]
        lr_history: 历史 lr 列表，格式 [{"step": int, "lr": float}, ...]
        epoch, step, total_steps: 训练进度
        start_time: 训练开始时间
        config: 配置字典
    """
    with _LOCK:
        if losses is not None:
            MONITOR_STATE["losses"] = losses
        if val_losses is not None:
            MONITOR_STATE["val_losses"] = val_losses
        if lr_history is not None:
            MONITOR_STATE["lr_history"] = lr_history
        if epoch is not None:
            MONITOR_STATE["epoch"] = epoch
        if step is not None:
            MONITOR_STATE["step"] = step
        if total_steps is not None:
            MONITOR_STATE["total_steps"] = total_steps
        if start_time is not None:
            MONITOR_STATE["start_time"] = start_time
        if config is not None:
            MONITOR_STATE["config"] = config
        save_state(force=True)  # resume rehydrate — persist immediately


def load_persisted_state(run_name=None):
    """Rehydrate MONITOR_STATE from the on-disk state.json (for resume).

    The donor persists the whole state to ``monitor_data/state.json`` on every
    update, so a resumed run can pick the loss/lr curve back up. When ``run_name``
    is given, only restore if the persisted ``config['run']`` matches it — so
    resuming run A doesn't graft an unrelated run B's curve. Fully guarded: any
    failure leaves the (fresh) state untouched. Returns the number of loss points
    restored (0 if nothing applicable).
    """
    state_file = MONITOR_DIR / "state.json"
    if not state_file.exists():
        return 0
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except Exception:
        return 0
    if run_name is not None:
        prev_run = (prev.get("config") or {}).get("run")
        if prev_run is not None and prev_run != run_name:
            return 0
    losses = prev.get("losses") or []
    restore_monitor_state(
        losses=losses,
        val_losses=prev.get("val_losses") or [],
        lr_history=prev.get("lr_history") or [],
        start_time=prev.get("start_time"),
    )
    return len(losses)


def _downsample_uniform(points, target_points: int):
    """均匀降采样到 target_points（保留首尾，适合 loss/lr 长序列）"""
    if not isinstance(target_points, int) or target_points <= 0:
        return points
    n = len(points)
    if n <= target_points:
        return points
    if target_points == 1:
        return [points[-1]]
    step = (n - 1) / (target_points - 1)
    out = []
    for i in range(target_points):
        idx = round(i * step)
        out.append(points[idx])
    return out


# HTML 页面
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anima Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { 
            text-align: center; 
            margin-bottom: 20px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2em;
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .card h2 { 
            font-size: 1.1em; 
            margin-bottom: 15px; 
            color: #00d4ff;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }
        .stat-item {
            background: rgba(0,212,255,0.1);
            border-radius: 12px;
            padding: 15px;
            text-align: center;
        }
        .stat-value { font-size: 1.8em; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 0.85em; color: #888; margin-top: 5px; }
        .chart-container { height: 300px; position: relative; }
        .samples-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
            gap: 15px;
        }
        .sample-item {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s;
        }
        .sample-item:hover { transform: scale(1.02); }
        .sample-item img { 
            width: 100%; 
            height: 200px; 
            object-fit: cover;
        }
        .sample-info {
            padding: 10px;
            font-size: 0.85em;
            color: #888;
        }
        .progress-bar {
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            transition: width 0.3s;
        }
        .config-list {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            font-size: 0.9em;
        }
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }
        .config-key { color: #888; }
        .config-value { color: #00d4ff; font-weight: 500; }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .full-width { grid-column: 1 / -1; }
        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Anima Training Monitor</h1>
        
        <div class="stats-grid" style="margin-bottom: 20px;">
            <div class="stat-item">
                <div class="stat-value" id="epoch">-</div>
                <div class="stat-label">Epoch</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="step">-</div>
                <div class="stat-label">Step</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="loss">-</div>
                <div class="stat-label">Loss</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="speed">-</div>
                <div class="stat-label">Speed (it/s)</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress" style="width: 0%"></div>
        </div>
        <p style="text-align: center; margin: 10px 0; color: #888;" id="progress-text">等待训练开始...</p>
        
        <div class="grid">
            <div class="card">
                <h2><span class="status-dot"></span> Loss 曲线 <span style="font-size:0.7em;color:#00ff88;margin-left:10px">绿色=平滑趋势</span></h2>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Learning Rate</h2>
                <div class="chart-container">
                    <canvas id="lrChart"></canvas>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>🖼️ 采样预览</h2>
                <div class="samples-grid" id="samples">
                    <p style="color: #666;">等待采样...</p>
                </div>
            </div>
            
            <div class="card full-width">
                <h2>⚙️ 训练配置</h2>
                <div class="config-list" id="config">
                    <p style="color: #666;">加载中...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 图表配置
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { 
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#888' }
                },
                y: { 
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#888' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        };
        
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '原始',
                        data: [],
                        borderColor: 'rgba(0,212,255,0.3)',
                        backgroundColor: 'rgba(0,212,255,0.05)',
                        fill: true,
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 1
                    },
                    {
                        label: '平滑 (EMA)',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Validation',
                        data: [],
                        borderColor: '#36d1ff',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0,
                        spanGaps: true,
                        pointRadius: 4,
                        pointStyle: 'rectRot',
                        borderWidth: 1.5
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    legend: { 
                        display: true,
                        labels: { color: '#888', boxWidth: 12 }
                    }
                }
            }
        });
        
        // 计算 EMA 平滑
        function calcEMA(data, alpha = 0.05) {
            if (data.length === 0) return [];
            const ema = [data[0]];
            for (let i = 1; i < data.length; i++) {
                ema.push(alpha * data[i] + (1 - alpha) * ema[i - 1]);
            }
            return ema;
        }
        
        const lrChart = new Chart(document.getElementById('lrChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124,58,237,0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 0
                }]
            },
            options: chartOptions
        });
        
        // 更新函数
        async function updateData() {
            try {
                const resp = await fetch('/api/state?' + Date.now());
                const data = await resp.json();
                
                // 更新统计
                document.getElementById('epoch').textContent = data.epoch || 0;
                document.getElementById('step').textContent = data.step || 0;
                document.getElementById('speed').textContent = (data.speed || 0).toFixed(2);
                
                // Loss
                if (data.losses && data.losses.length > 0) {
                    const lastLoss = data.losses[data.losses.length - 1].loss;
                    document.getElementById('loss').textContent = lastLoss.toFixed(4);
                    
                    // 更新图表（最多显示 500 个点）
                    const displayLosses = data.losses.slice(-500);
                    const rawLosses = displayLosses.map(l => l.loss);
                    const smoothLosses = calcEMA(rawLosses, 0.02);  // alpha=0.02 更平滑
                    
                    lossChart.data.labels = displayLosses.map(l => l.step);
                    lossChart.data.datasets[0].data = rawLosses;      // 原始曲线
                    lossChart.data.datasets[1].data = smoothLosses;   // 平滑曲线
                    // Validation (sparse): align each val point onto the nearest
                    // training-step label so it overlays the loss curve.
                    const _vl = data.val_losses || [];
                    if (_vl.length) {
                        const _steps = displayLosses.map(l => l.step);
                        const _arr = new Array(_steps.length).fill(null);
                        _vl.forEach(v => { let bi=0, bd=Infinity; for (let i=0;i<_steps.length;i++){ const d=Math.abs(_steps[i]-v.step); if (d<bd){bd=d;bi=i;} } _arr[bi]=v.loss; });
                        lossChart.data.datasets[2].data = _arr;
                        const _bv = _vl.reduce((a,b)=> b.loss<a.loss ? b : a);
                        const _vlEl = document.getElementById('val-loss');
                        if (_vlEl) _vlEl.textContent = Number(_vl[_vl.length-1].loss).toFixed(4) + ' (best ' + Number(_bv.loss).toFixed(4) + ')';
                    } else {
                        lossChart.data.datasets[2].data = [];
                    }
                    lossChart.update('none');
                    
                    // 显示平滑后的趋势（最近 100 步 vs 之前 100 步）
                    if (smoothLosses.length >= 200) {
                        const recent = smoothLosses.slice(-100).reduce((a,b) => a+b, 0) / 100;
                        const before = smoothLosses.slice(-200, -100).reduce((a,b) => a+b, 0) / 100;
                        const trend = ((recent - before) / before * 100).toFixed(2);
                        const trendText = trend < 0 ? `↓${Math.abs(trend)}%` : `↑${trend}%`;
                        const trendColor = trend < 0 ? '#00ff88' : '#ff6b6b';
                        document.getElementById('loss').innerHTML = 
                            `${lastLoss.toFixed(4)} <span style="font-size:0.5em;color:${trendColor}">${trendText}</span>`;
                    }
                }
                
                // LR
                if (data.lr_history && data.lr_history.length > 0) {
                    const displayLr = data.lr_history.slice(-500);
                    lrChart.data.labels = displayLr.map(l => l.step);
                    lrChart.data.datasets[0].data = displayLr.map(l => l.lr);
                    lrChart.update('none');
                }
                
                // 进度
                if (data.total_steps > 0) {
                    const pct = Math.min(100, (data.step / data.total_steps) * 100);
                    document.getElementById('progress').style.width = pct + '%';
                    
                    const elapsed = data.start_time ? (Date.now()/1000 - data.start_time) : 0;
                    const eta = data.speed > 0 ? (data.total_steps - data.step) / data.speed : 0;
                    document.getElementById('progress-text').textContent = 
                        `${pct.toFixed(1)}% | 已用: ${formatTime(elapsed)} | 预计剩余: ${formatTime(eta)}`;
                }
                
                // 采样图片
                if (data.samples && data.samples.length > 0) {
                    const samplesHtml = data.samples.slice(-6).reverse().map(s => `
                        <div class="sample-item">
                            <img src="/samples/${s.path.split(/[\\\\/]/).pop()}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22/>'">
                            <div class="sample-info">Step ${s.step}</div>
                        </div>
                    `).join('');
                    document.getElementById('samples').innerHTML = samplesHtml;
                }
                
                // 配置
                if (data.config && Object.keys(data.config).length > 0) {
                    const configHtml = Object.entries(data.config).map(([k, v]) => `
                        <div class="config-item">
                            <span class="config-key">${k}</span>
                            <span class="config-value">${v}</span>
                        </div>
                    `).join('');
                    document.getElementById('config').innerHTML = configHtml;
                }
                
            } catch (e) {
                console.log('Update failed:', e);
            }
        }
        
        function formatTime(seconds) {
            if (!seconds || seconds < 0) return '--:--';
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            if (h > 0) return `${h}h ${m}m`;
            return `${m}m ${s}s`;
        }
        
        // 每秒更新
        setInterval(updateData, 1000);
        updateData();
    </script>
</body>
</html>
"""


class MonitorHandler(SimpleHTTPRequestHandler):
    """监控服务器 Handler"""

    def __init__(self, *args, output_dir=None, **kwargs):
        self.output_dir = output_dir or Path("./output")
        super().__init__(*args, **kwargs)

    def _mutation_allowed(self) -> bool:
        """State-changing routes (live LR control, open-folder) are localhost-only +
        CSRF-guarded. Read routes stay open so a LAN host bound via --monitor_host
        0.0.0.0 can still WATCH; but it cannot STEER a live run. Two gates:
          1. client must be loopback — a LAN/remote host is read-only even on 0.0.0.0
             (the monitor was read-only before runtime LR control existed; control
             must not silently inherit the documented "expose on the LAN" bind).
          2. Sec-Fetch-Site must be same-origin/none — blocks a cross-site page in the
             operator's own browser from forging a control GET (CSRF), since the live
             LR mutation lands even though the response is opaque to the attacker."""
        host = (self.client_address[0] if self.client_address else "") or ""
        if host not in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
            return False
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in ("same-origin", "none"):
            return False
        return True

    def _deny_control(self):
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {"ok": False, "error": "control restricted to a local same-origin client"}
            ).encode("utf-8")
        )

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

            # 优先使用 monitor_smooth.html
            smooth_html = Path(__file__).resolve().parent / "monitor_smooth.html"
            if smooth_html.exists():
                with open(smooth_html, "r", encoding="utf-8") as f:
                    content = f.read()
                self.wfile.write(content.encode("utf-8"))
            else:
                print(
                    f"[Monitor] Warning: smooth UI not found at {smooth_html}, using fallback."
                )
                self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path.startswith("/api/state"):
            # 支持 query 参数：max_points（对 losses/lr_history 降采样，降低传输和前端渲染压力）
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query or "")
            try:
                max_points = int(qs.get("max_points", ["0"])[0] or 0)
            except Exception:
                max_points = 0

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            state = get_state()
            if max_points > 0:
                # 不修改全局状态，只对返回值裁剪
                if "losses" in state:
                    state["losses"] = _downsample_uniform(state["losses"], max_points)
                if "lr_history" in state:
                    state["lr_history"] = _downsample_uniform(
                        state["lr_history"], max_points
                    )
            self.wfile.write(json.dumps(state).encode("utf-8"))
        elif self.path.startswith("/samples/"):
            # 提供采样图片. anima-lora 把样图存到 <output_dir>/sample/（单数），
            # 而本监控器原本用 samples/（复数）；优先 sample/，回退 samples/.
            filename = self.path.split("/")[-1]
            sample_path = self.output_dir / "sample" / filename
            if not sample_path.exists():
                sample_path = self.output_dir / "samples" / filename
            if sample_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(sample_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif self.path.startswith("/open-samples"):
            # Local dashboard → open the sample dir in the OS file browser. Launches
            # an OS process, so gate it like the control routes (loopback + same-origin)
            # — a remote/cross-site GET must not pop a file-manager on the training host.
            if not self._mutation_allowed():
                return self._deny_control()
            d = self.output_dir / "sample"
            if not d.exists():
                d = self.output_dir / "samples"
            ok = False
            try:
                d.mkdir(parents=True, exist_ok=True)
                if sys.platform.startswith("win"):
                    os.startfile(str(d))  # noqa: S606
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(d)])
                else:
                    subprocess.Popen(["xdg-open", str(d)])
                ok = True
            except Exception:
                ok = False
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "path": str(d)}).encode("utf-8"))
        elif self.path.startswith("/api/runs"):
            # List saved run archives (newest first) for the Runs browser / compare.
            runs = []
            runs_dir = self.output_dir / "runs"
            if runs_dir.is_dir():
                for sub in sorted(runs_dir.iterdir(), reverse=True):
                    mf = sub / "meta.json"
                    if mf.is_file():
                        try:
                            m = json.loads(mf.read_text(encoding="utf-8"))
                            m["id"] = sub.name
                            runs.append(m)
                        except Exception:
                            pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(runs).encode("utf-8"))
        elif self.path.startswith("/api/run/"):
            # /api/run/<id>/state  |  /api/run/<id>/sample/<file>
            parts = self.path.split("?", 1)[0].strip("/").split("/")
            runs_root = (self.output_dir / "runs").resolve()
            served = False
            if len(parts) >= 4 and ".." not in parts:
                run_dir = runs_root / parts[2]
                try:
                    inside = str(run_dir.resolve()).startswith(str(runs_root))
                except Exception:
                    inside = False
                if inside and parts[3] == "state":
                    sf = run_dir / "state.json"
                    if sf.is_file():
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(sf.read_bytes())
                        served = True
                elif inside and parts[3] == "sample" and len(parts) >= 5:
                    img = run_dir / "samples" / parts[4]
                    if img.is_file():
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.end_headers()
                        self.wfile.write(img.read_bytes())
                        served = True
            if not served:
                self.send_error(404)
        elif self.path.startswith("/api/notes"):
            # AI-Analysis notes (written by the MCP server's post_analysis tool).
            try:
                from . import mcp_data

                notes = mcp_data.read_notes()
            except Exception:
                notes = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(notes).encode("utf-8"))
        elif self.path.startswith("/api/control"):
            # Runtime LR control. Query sets it; bare GET returns the current state.
            # ?lr_scale=X  ·  ?reset=1  ·  ?decay=1&k=K&floor=F (from the current step)
            from . import mcp_data

            qs = parse_qs(urlparse(self.path).query or "")
            step = int(MONITOR_STATE.get("step") or 0)
            # A bare GET reads the current control state (open to all, incl. a LAN
            # dashboard); only the state-CHANGING variants are gated (loopback +
            # same-origin) so a remote/cross-site request can't steer the live LR.
            mutating = any(k in qs for k in ("lr_scale", "reset", "decay"))
            if mutating and not self._mutation_allowed():
                return self._deny_control()
            try:
                if "lr_scale" in qs:
                    mcp_data.set_lr_scale(float(qs["lr_scale"][0]))
                elif "reset" in qs:
                    mcp_data.reset_control()
                elif "decay" in qs:
                    k = int(qs.get("k", ["500"])[0])
                    floor = float(qs.get("floor", ["0"])[0])
                    mcp_data.start_lr_decay(step, k, floor)
            except (ValueError, KeyError):
                pass
            ctrl = mcp_data.read_control()
            out = {
                "control": ctrl,
                "step": step,
                "effective_scale": mcp_data.effective_lr_scale(ctrl, step),
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # 静默日志


def start_monitor_server(
    port=8766, host="127.0.0.1", output_dir=None, open_browser=True
):
    """启动监控服务器"""
    output_dir = Path(output_dir) if output_dir else Path("./output")

    def handler(*args, **kwargs):
        return MonitorHandler(*args, output_dir=output_dir, **kwargs)

    # ThreadingHTTPServer (not plain HTTPServer): the dashboard holds a keep-alive
    # connection and polls /api/state every 1s + fetches /samples/*. A single-
    # threaded server blocks on one connection at a time, so a lingering keep-alive
    # (or a slow sample read) stalls every poll → the dashboard shows
    # "Disconnected/OFFLINE" and a stale (lagging) step while training runs fine.
    # daemon_threads so handler threads never block process exit.
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True

    def run():
        shown_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
        print(f"📊 训练监控面板: http://{shown_host}:{port}")
        server.serve_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    if open_browser:
        time.sleep(0.5)
        webbrowser.open(
            f"http://{('localhost' if host in ('0.0.0.0', '127.0.0.1') else host)}:{port}"
        )

    return server


if __name__ == "__main__":
    # 测试模式
    import random

    server = start_monitor_server(port=8766)

    print("测试模式：模拟训练数据...")
    for i in range(1000):
        update_monitor(
            loss=0.5 * (0.95 ** (i / 10)) + random.random() * 0.05,
            lr=1e-4 * (0.99 ** (i / 50)),
            epoch=i // 100 + 1,
            step=i,
            total_steps=1000,
            speed=2.5 + random.random() * 0.5,
            config={
                "model": "Anima LoKr",
                "rank": 64,
                "epochs": 10,
                "batch_size": 4,
            },
        )
        time.sleep(0.1)
