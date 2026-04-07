"""
Ptychography GUI Dashboard
==========================
Serve standalone:
    panel serve ptycho_dashboard.py --show
Or via CLI:
    ptyxplore --allow-websocket-origin '*'
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import panel as pn
import numpy as np
from datetime import datetime
import h5py
import os
import traceback

pn.extension(
    "tabulator", "codeeditor",
    sizing_mode="stretch_width",
    notifications=True,
)

# ── Colour tokens ──────────────────────────────────────────────────────────────
ACCENT    = "#0FCFAF"
ACCENT2   = "#845EC2"
WARN      = "#FF6B6B"
BG_DARK   = "#0D1117"
BG_MID    = "#161B22"
BG_CARD   = "#1C2333"
BORDER    = "#30363D"
TXT       = "#E6EDF3"
TXT_MUTED = "#8B949E"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&display=swap');

:root {{
  --accent:   {ACCENT};
  --accent2:  {ACCENT2};
  --warn:     {WARN};
  --bg-dark:  {BG_DARK};
  --bg-mid:   {BG_MID};
  --bg-card:  {BG_CARD};
  --border:   {BORDER};
  --txt:      {TXT};
  --txt-muted:{TXT_MUTED};
}}

body, .bk-root {{
  background: var(--bg-dark) !important;
  color: var(--txt) !important;
  font-family: 'Syne', sans-serif !important;
}}

.ptycho-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px 28px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}}
.ptycho-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}}

.dash-header {{
  background: linear-gradient(135deg, {BG_MID} 0%, #0A1628 100%);
  border-bottom: 1px solid var(--border);
  padding: 20px 32px;
  display: flex; align-items: center; gap: 16px;
}}
.dash-title {{
  font-size: 22px; font-weight: 800; letter-spacing: -0.5px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0;
}}
.dash-sub {{
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--txt-muted); margin: 2px 0 0 0;
}}

.section-label {{
  font-size: 11px; font-weight: 600; letter-spacing: 2px;
  text-transform: uppercase; color: var(--accent);
  margin: 0 0 14px 0; font-family: 'JetBrains Mono', monospace;
}}

.badge {{
  display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-family: 'JetBrains Mono', monospace;
  font-weight: 600; letter-spacing: 0.5px;
}}
.badge-ok   {{ background: rgba(0,201,167,0.15);  color: var(--accent);    border: 1px solid rgba(0,201,167,0.3); }}
.badge-err  {{ background: rgba(255,107,107,0.15); color: var(--warn);     border: 1px solid rgba(255,107,107,0.3); }}
.badge-idle {{ background: rgba(139,148,158,0.15); color: var(--txt-muted); border: 1px solid rgba(139,148,158,0.3); }}
.badge-run  {{ background: rgba(121,192,255,0.15); color: #79C0FF;         border: 1px solid rgba(121,192,255,0.3); }}

.stat-grid {{
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px; margin-top: 12px;
}}
.stat-chip {{
  background: var(--bg-mid); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px;
}}
.stat-chip .val {{
  font-family: 'JetBrains Mono', monospace; font-size: 18px;
  font-weight: 600; color: var(--accent); line-height: 1;
}}
.stat-chip .lbl {{
  font-size: 11px; color: var(--txt-muted); margin-top: 4px;
  text-transform: uppercase; letter-spacing: 1px;
}}

.log-box {{
  background: var(--bg-dark); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
  line-height: 1.7; color: var(--txt-muted);
  max-height: 280px; overflow-y: auto;
}}
.log-ok   {{ color: {ACCENT}; }}
.log-err  {{ color: {WARN}; }}
.log-info {{ color: #79C0FF; }}
.log-warn {{ color: #E3B341; }}

.queue-row {{
  display: flex; align-items: center; gap: 12px;
  background: var(--bg-mid); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
  font-family: 'JetBrains Mono', monospace; font-size: 12px;
}}
.q-idx  {{ color: var(--txt-muted); min-width: 24px; }}
.q-op   {{ color: var(--txt); flex: 1; }}
.q-roi  {{ color: var(--accent2); min-width: 100px; }}
.q-par  {{ color: var(--txt-muted); font-size: 11px; flex: 1; }}

.bk-input, .bk-select {{
  background: var(--bg-mid) !important; border: 1px solid var(--border) !important;
  color: var(--txt) !important; border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important; font-size: 13px !important;
}}
.bk-input:focus {{ border-color: var(--accent) !important; outline: none !important; }}
.bk-btn-primary {{
  background: linear-gradient(135deg, var(--accent), #00A88A) !important;
  border: none !important; color: #000 !important; font-weight: 700 !important;
  font-family: 'Syne', sans-serif !important; letter-spacing: 0.5px !important;
  border-radius: 8px !important; transition: transform 0.15s, box-shadow 0.15s !important;
}}
.bk-btn-primary:hover {{
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(0,201,167,0.35) !important;
}}
.bk-btn-default {{
  background: var(--bg-mid) !important; border: 1px solid var(--border) !important;
  color: var(--txt) !important; font-family: 'Syne', sans-serif !important;
  border-radius: 8px !important;
}}
.bk-widget-box {{ background: transparent !important; }}
label.bk {{ color: var(--txt-muted) !important; font-size: 12px !important; letter-spacing: 0.3px; }}
.divider {{ border: none; border-top: 1px solid var(--border); margin: 18px 0; }}
"""

pn.config.raw_css.append(CSS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Global state
# ═══════════════════════════════════════════════════════════════════════════════

log_lines: list[str] = []
state = {
    "exp":             None,
    "scan":            None,
    "current_frame":   None,
    "processed_frame": None,
    "rois":            {},    # name → ROI object
    "pupil_name":      None,  # name of the ROI designated as pupil
    "roi_history":     {},    # name → list[str]
}
_prev_figs: dict = {}
pipeline_queue: list[dict] = []
plot_gallery:   list[dict] = []


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def log(msg: str, kind: str = "info") -> None:
    log_lines.append(f'<span class="log-{kind}">[{ts()}] {msg}</span>')
    _refresh_log()


def _refresh_log():
    log_pane.object = '<br>'.join(log_lines[-80:])


def _set_pane_fig(pane, fig, key: str):
    if key in _prev_figs and _prev_figs[key] is not None:
        try:
            plt.close(_prev_figs[key])
        except Exception:
            pass
    _prev_figs[key] = fig
    pane.object = fig


def _make_image_fig(data: np.ndarray, title: str, log_scale: bool,
                    cmap: str, vmin, vmax, figsize=(7, 5.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TXT_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    disp = np.log1p(np.abs(data)) if log_scale else data
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, color=TXT, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log(1+|I|)" if log_scale else "Intensity", color=TXT_MUTED, fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT_MUTED)
    fig.patch.set_facecolor(BG_DARK)
    plt.tight_layout()
    return fig


def _placeholder_fig(msg: str):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            color=TXT_MUTED, fontsize=12, transform=ax.transAxes)
    ax.axis("off")
    fig.patch.set_facecolor(BG_DARK)
    plt.tight_layout()
    return fig


def _make_kcoord_fig(det_image, k_coords, coords, cmap, title):
    import matplotlib.patches as mpatches
    sx = coords[0] if coords else 0
    sy = coords[2] if len(coords) > 2 else 0
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.tick_params(colors=TXT_MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.imshow(det_image, cmap=cmap, origin='lower')
    for idx in k_coords:
        rect = mpatches.Rectangle(
            (idx[1] - 1 - sy, idx[0] - 1 - sx), 1, 1,
            linewidth=1, edgecolor=ACCENT, facecolor=ACCENT,
        )
        ax.add_patch(rect)
    ax.set_title(title, color=TXT, fontsize=11)
    fig.patch.set_facecolor(BG_DARK)
    plt.tight_layout()
    return fig


def _add_roi_history(name: str, entry: str):
    if name and name in state["roi_history"]:
        state["roi_history"][name].append(f"[{ts()}] {entry}")


def _add_to_gallery(title: str, fig) -> None:
    pane = pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width", height=460)
    plot_gallery.append({"title": title, "pane": pane, "fig": fig})
    _rebuild_gallery()


def _rebuild_gallery():
    gallery_col.clear()
    if not plot_gallery:
        gallery_col.append(pn.pane.HTML(
            "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
            "No plots yet — run pipeline steps that include Plot operations.</span>"))
        return
    for i, item in enumerate(reversed(plot_gallery)):
        actual_idx = len(plot_gallery) - 1 - i

        def _make_rm(idx=actual_idx):
            btn = pn.widgets.Button(name="✕", button_type="default", width=40)
            def _rm(event):
                if idx < len(plot_gallery):
                    removed = plot_gallery.pop(idx)
                    try:
                        plt.close(removed["fig"])
                    except Exception:
                        pass
                    _rebuild_gallery()
            btn.on_click(_rm)
            return btn

        gallery_col.append(pn.Column(
            pn.Row(
                pn.pane.HTML(
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                    f"color:var(--accent)'>{len(plot_gallery)-i}. {item['title']}</span>"),
                _make_rm(),
                align="center",
            ),
            item["pane"],
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ))


# ── Shared widgets ─────────────────────────────────────────────────────────────
status_badge = pn.pane.HTML("<span class='badge badge-idle'>IDLE</span>", width=120)
log_pane = pn.pane.HTML(
    "<span class='log-info'>[--:--:--] Dashboard initialised.</span>",
    sizing_mode="stretch_width",
)
card_log = pn.Column(
    pn.pane.HTML("<p class='section-label'>Activity Log</p>", sizing_mode="stretch_width"),
    pn.Column(log_pane, css_classes=["log-box"], sizing_mode="stretch_width"),
    sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 0, 0),
)
gallery_col = pn.Column(sizing_mode="stretch_width")


def _set_status(ok):
    if ok is None:
        status_badge.object = "<span class='badge badge-idle'>IDLE</span>"
    elif ok:
        status_badge.object = "<span class='badge badge-ok'>READY</span>"
    else:
        status_badge.object = "<span class='badge badge-err'>ERROR</span>"


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LOADING
# ═══════════════════════════════════════════════════════════════════════════════

w_year     = pn.widgets.IntInput(name="Year", value=datetime.now().year, step=1, width=120)
w_btid     = pn.widgets.TextInput(name="Beamtime ID", placeholder="e.g. 11018232", width=200)
w_beamline = pn.widgets.Select(name="Beamline", options=["p11","p06","p10","p23"], value="p11", width=140)
btn_create_exp = pn.widgets.Button(name="Create Experiment", button_type="primary", width=200)

exp_path_md = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
    "Fill fields above to preview path…</span>", sizing_mode="stretch_width",
)


def _update_path_preview(event=None):
    y  = w_year.value or "YYYY"
    bt = w_btid.value.strip() or "<beamtime_id>"
    bl = w_beamline.value
    path = f"/asap3/petra3/gpfs/{bl}/{y}/data/{bt}/raw"
    exp_path_md.object = (
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;color:var(--txt-muted)'>"
        f"<span style='color:var(--accent)'>scan_frames:</span> {path}/scan_frames<br>"
        f"<span style='color:var(--accent)'>logs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</span> {path}/server_log/Scan_logs"
        f"</div>"
    )


for _w in [w_year, w_btid, w_beamline]:
    _w.param.watch(_update_path_preview, "value")
_update_path_preview()

w_scan_num  = pn.widgets.IntInput(name="Scan Number", value=1, step=1, width=140)
w_detector  = pn.widgets.Select(name="Detector", options=["Eiger","Lambda"], value="Eiger", width=140)
w_sample    = pn.widgets.TextInput(name="Sample Name", placeholder="e.g. SampleA_001", width=220)
w_det_dist  = pn.widgets.FloatInput(name="Detector Distance (m)", value=2.0, step=0.05, width=200)
btn_create_scan = pn.widgets.Button(name="Load Scan", button_type="primary", width=180, disabled=True)
btn_clear_scan  = pn.widgets.Button(name="Clear", button_type="default", width=100)

scan_info_html = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;"
    "font-size:12px'>No scan loaded yet.</span>", sizing_mode="stretch_width",
)

file_table = pn.widgets.Tabulator(
    value=None, pagination="local", page_size=12,
    sizing_mode="stretch_width", height=260, theme="midnight",
    stylesheets=[""":host .tabulator { background: var(--bg-dark); color: var(--txt);
       font-family: 'JetBrains Mono', monospace; font-size: 12px; }"""],
)

_SCAN_STAT_LABELS = ["Scan #", "Files", "Energy (keV)", "Step X (nm)", "Step Y (nm)", "Points X×Y"]
stats_html = pn.pane.HTML(
    "<div class='stat-grid'>"
    + "".join(f"<div class='stat-chip'><div class='val'>—</div><div class='lbl'>{l}</div></div>"
              for l in _SCAN_STAT_LABELS)
    + "</div>", sizing_mode="stretch_width",
)


def _update_scan_stats(scan):
    n_files = len(getattr(scan, "file_list", []) or [])
    energy  = getattr(scan, "energy", None)
    sx = getattr(scan, "step_size_x", None)
    sy = getattr(scan, "step_size_y", None)
    px = getattr(scan, "scan_x_steps_count", None)
    py = getattr(scan, "scan_y_steps_count", None)
    stats_html.object = (
        "<div class='stat-grid'>"
        f"<div class='stat-chip'><div class='val'>{scan.scan_num}</div><div class='lbl'>Scan #</div></div>"
        f"<div class='stat-chip'><div class='val'>{n_files}</div><div class='lbl'>Files</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{energy:.4f}' if energy else '—'}</div><div class='lbl'>Energy (keV)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{sx:.2f}' if sx else '—'}</div><div class='lbl'>Step X (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{sy:.2f}' if sy else '—'}</div><div class='lbl'>Step Y (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{px}×{py}' if (px and py) else '—'}</div><div class='lbl'>Points X×Y</div></div>"
        "</div>"
    )


def _reset_scan_stats():
    stats_html.object = (
        "<div class='stat-grid'>"
        + "".join(f"<div class='stat-chip'><div class='val'>—</div><div class='lbl'>{l}</div></div>"
                  for l in _SCAN_STAT_LABELS)
        + "</div>"
    )


def on_create_exp(event):
    year = w_year.value
    btid = w_btid.value.strip()
    bl   = w_beamline.value
    if not btid:
        log("Beamtime ID is required.", "err"); _set_status(False); return
    log(f"Creating Exp(year={year}, beamtime_id='{btid}', beamline='{bl}')…", "info")
    try:
        from ptyxplore.cbd_loader import Exp
        state["exp"] = Exp(year=year, beamtime_id=btid, beamline=bl)
        log(f"Exp created ✓  scan_frames → {state['exp'].scan_frames_dir}", "ok")
        _set_status(True)
        btn_create_scan.disabled = False
        pn.state.notifications.success("Experiment created!", duration=3000)
    except Exception as exc:
        log(f"Failed to create Exp: {exc}", "err"); _set_status(False)
        pn.state.notifications.error(str(exc), duration=5000)


def on_create_scan(event):
    if state["exp"] is None:
        log("Create an Experiment first.", "warn"); return
    snum   = w_scan_num.value
    det    = w_detector.value
    sample = w_sample.value.strip() or None
    ddist  = w_det_dist.value
    log(f"Loading Scan {snum} (det={det}, dist={ddist} m)…", "info")
    try:
        from ptyxplore.cbd_loader import Scan
        state["scan"] = Scan(exp=state["exp"], scan_num=snum,
                             detector=det, sample_name=sample, det_distance=ddist)
        scan = state["scan"]
        log(f"Scan {snum} loaded ✓  ({len(scan.file_list)} files found)", "ok")
        _set_status(True)
        scan_info_html.object = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;line-height:1.9'>"
            f"<span style='color:var(--txt-muted)'>data_dir  :</span> <span style='color:var(--txt)'>{scan.data_dir}</span><br>"
            f"<span style='color:var(--txt-muted)'>slow_axis :</span> <span style='color:{ACCENT2}'>{scan.slow_axis}</span><br>"
            f"<span style='color:var(--txt-muted)'>wavelength:</span> <span style='color:{ACCENT}'>{scan.wavelength*10:.4f} Å</span>"
            f"&nbsp;&nbsp;<span style='color:var(--txt-muted)'>energy:</span> <span style='color:{ACCENT}'>{scan.energy:.4f} keV</span><br>"
            f"<span style='color:var(--txt-muted)'>det psize :</span> <span style='color:var(--txt)'>{scan.det_psize*1e-3:.0f} µm</span>"
            f"</div>"
        )
        import pandas as pd
        file_table.value = pd.DataFrame([{"#": i+1, "File": f} for i, f in enumerate(scan.file_list)])
        _update_scan_stats(scan)
        n = len(scan.file_list)
        w_file_idx.end   = max(0, n - 1)
        w_file_idx.value = 0
        pn.state.notifications.success(f"Scan {snum} loaded!", duration=3000)
    except Exception as exc:
        log(f"Failed to load Scan: {exc}", "err"); _set_status(False)
        pn.state.notifications.error(str(exc), duration=5000)


def on_clear_scan(event):
    state["scan"] = None
    state["current_frame"] = state["processed_frame"] = None
    scan_info_html.object = (
        "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;"
        "font-size:12px'>No scan loaded yet.</span>")
    file_table.value = None
    _reset_scan_stats(); _set_status(None)
    log("Scan cleared.", "warn")
    w_file_idx.end = w_file_idx.value = 0
    w_frame_idx.end = w_frame_idx.value = 0


btn_create_exp.on_click(on_create_exp)
btn_create_scan.on_click(on_create_scan)
btn_clear_scan.on_click(on_clear_scan)

tab_loading = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>01 — Experiment</p>", sizing_mode="stretch_width"),
        pn.Row(w_year, w_btid, w_beamline, align="end"),
        exp_path_md,
        pn.Row(btn_create_exp, status_badge, align="center"),
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>02 — Scan</p>", sizing_mode="stretch_width"),
        pn.Row(w_scan_num, w_detector, w_sample, w_det_dist, align="end"),
        pn.Row(btn_create_scan, btn_clear_scan, align="center"),
        pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
        scan_info_html, stats_html,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>03 — Detector Files</p>", sizing_mode="stretch_width"),
        file_table,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 0, 0),
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ROI MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

w_roi_name = pn.widgets.TextInput(name="ROI name", placeholder="e.g. bragg_001", width=180)
w_roi_kind = pn.widgets.Select(name="Kind", options=["Bragg", "Pupil"], value="Bragg", width=120)
w_roi_rs   = pn.widgets.IntInput(name="row start", value=0,   width=110)
w_roi_re   = pn.widgets.IntInput(name="row end",   value=100, width=110)
w_roi_cs   = pn.widgets.IntInput(name="col start", value=0,   width=110)
w_roi_ce   = pn.widgets.IntInput(name="col end",   value=100, width=110)
w_roi_cx   = pn.widgets.FloatInput(name="centre px row", value=0.0, width=140)
w_roi_cy   = pn.widgets.FloatInput(name="centre px col", value=0.0, width=140)
btn_add_roi = pn.widgets.Button(name="Add ROI", button_type="primary", width=140)

roi_cards_col = pn.Column(sizing_mode="stretch_width")
_roi_placeholder = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
    "No ROIs loaded yet.</span>", sizing_mode="stretch_width")


def _roi_card_html(roi, name: str) -> str:
    tags = " ".join(
        f"<span class='badge badge-ok'>{t}</span>" if v
        else f"<span class='badge badge-idle'>{t}</span>"
        for t, v in [
            ("4D",   roi.data_4d is not None),
            ("coh",  roi.coherent_imgs is not None),
            ("kouts",roi.kout_coords is not None),
            ("kins", roi.kin_coords is not None),
        ]
    )
    pupil = " <span class='badge badge-ok'>PUPIL</span>" if name == state["pupil_name"] else ""
    history = state["roi_history"].get(name, [])
    hist_html = ""
    if history:
        items = "".join(f"<div>{h}</div>" for h in history[-6:])
        hist_html = (
            f"<div style='margin-top:8px;font-family:JetBrains Mono,monospace;"
            f"font-size:11px;color:var(--txt-muted);max-height:90px;overflow-y:auto'>{items}</div>"
        )
    return (
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;line-height:2'>"
        f"<b style='color:var(--accent)'>{name}</b>&nbsp; kind: {roi.kind}&nbsp; "
        f"coords: {roi.coords}<br>{tags}{pupil}{hist_html}</div>"
    )


def _make_roi_card(name: str) -> pn.Column:
    roi = state["rois"][name]
    info = pn.pane.HTML(_roi_card_html(roi, name), sizing_mode="stretch_width")
    btn_p = pn.widgets.Button(name="Set as Pupil", button_type="default", width=140)
    btn_r = pn.widgets.Button(name="Remove",       button_type="default", width=100)

    def _on_pupil(event):
        state["pupil_name"] = name
        log(f"'{name}' set as Pupil ROI.", "ok")
        pn.state.notifications.success(f"'{name}' is now the Pupil ROI.", duration=2000)
        _rebuild_roi_cards()
        _refresh_pipe_roi_select()

    def _on_remove(event):
        if name in state["rois"]:
            del state["rois"][name]
            state["roi_history"].pop(name, None)
            if state["pupil_name"] == name:
                state["pupil_name"] = None
            log(f"ROI '{name}' removed.", "warn")
            _rebuild_roi_cards()
            _refresh_pipe_roi_select()

    btn_p.on_click(_on_pupil)
    btn_r.on_click(_on_remove)
    return pn.Column(info, pn.Row(btn_p, btn_r),
                     sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 8, 0))


def _rebuild_roi_cards():
    roi_cards_col.clear()
    if not state["rois"]:
        roi_cards_col.append(_roi_placeholder)
        return
    for name in state["rois"]:
        roi_cards_col.append(_make_roi_card(name))


def on_add_roi(event):
    if state["scan"] is None:
        log("Load a scan first.", "warn")
        pn.state.notifications.warning("Load a scan first.", duration=3000); return
    name = w_roi_name.value.strip()
    if not name:
        log("ROI name is required.", "err")
        pn.state.notifications.error("Enter a name for the ROI.", duration=3000); return
    if name in state["rois"]:
        log(f"ROI '{name}' already exists.", "err")
        pn.state.notifications.error(f"Name '{name}' already in use.", duration=3000); return
    kind   = w_roi_kind.value.lower()
    coords = [w_roi_rs.value, w_roi_re.value, w_roi_cs.value, w_roi_ce.value]
    centre = [w_roi_cx.value, w_roi_cy.value]
    try:
        from ptyxplore.cbd_loader import ROI
        roi = ROI(scan=state["scan"], kind=kind, coords=coords, centre_pixel=centre)
        state["rois"][name] = roi
        state["roi_history"][name] = [f"[{ts()}] Created ({kind}, coords={coords})"]
        if kind == "pupil" and state["pupil_name"] is None:
            state["pupil_name"] = name
        _rebuild_roi_cards()
        _refresh_pipe_roi_select()
        log(f"ROI '{name}' created ✓", "ok")
        pn.state.notifications.success(f"ROI '{name}' added!", duration=2000)
    except Exception as exc:
        log(f"Failed to create ROI: {exc}", "err")
        pn.state.notifications.error(str(exc), duration=5000)


btn_add_roi.on_click(on_add_roi)

tab_roi_manager = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Add ROI</p>", sizing_mode="stretch_width"),
        pn.Row(w_roi_name, w_roi_kind, align="end"),
        pn.pane.HTML("<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;"
                     "font-size:11px'>Detector coords (pixels)</span>"),
        pn.Row(w_roi_rs, w_roi_re, w_roi_cs, w_roi_ce),
        pn.Row(w_roi_cx, w_roi_cy),
        pn.Row(btn_add_roi),
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Active ROIs</p>", sizing_mode="stretch_width"),
        roi_cards_col,
        sizing_mode="stretch_width",
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

_ALL_OPS = {
    # ROI data operations
    "Make 4D dataset":          {"target": "roi"},
    "Pool detector":            {"target": "roi"},
    "Mask region detector":     {"target": "roi"},
    "Make k-outs":              {"target": "roi"},
    "Estimate 2θ":              {"target": "roi", "needs_pupil": True},
    "Compute kins":             {"target": "roi", "needs_pupil": True},
    "Compute kins w geometry":  {"target": "roi", "needs_pupil": True},
    "Refine kins":              {"target": "roi"},
    "Make coherent images":     {"target": "roi"},
    "Filter coherent images":   {"target": "roi"},
    "Remove background":        {"target": "roi"},
    "Median filter (coh)":      {"target": "roi"},
    "Detect objects":           {"target": "roi"},
    "Align coherent images":    {"target": "roi"},
    "Mask cohimgs threshold":   {"target": "roi"},
    "Mask region cohimgs":      {"target": "roi"},
    # Scan-level
    "Sum detector frames":      {"target": "scan"},
    # Plot → Visualisation tab
    "Plot: Average detector":   {"target": "roi"},
    "Plot: Averaged coherent":  {"target": "roi"},
    "Plot: Coherent sequence":  {"target": "roi"},
    "Plot: 4D viewer (single)": {"target": "roi"},
    "Plot: K-outs":             {"target": "roi"},
    "Plot: K-ins":              {"target": "roi", "needs_pupil": True},
    # Save
    "Save: ROI data (HDF5)":    {"target": "roi"},
    "Save: Frame (NumPy)":      {"target": "scan"},
    "Save: Frame (TIFF)":       {"target": "scan"},
}

w_pipe_op  = pn.widgets.Select(
    name="Operation", options=list(_ALL_OPS.keys()), value="Make 4D dataset", width=280)
w_pipe_roi = pn.widgets.Select(
    name="ROI", options=["— none —"], value="— none —", width=200)
btn_pipe_add   = pn.widgets.Button(name="Add to Queue",  button_type="primary", width=160)
btn_pipe_run   = pn.widgets.Button(name="▶  Run All",    button_type="primary", width=150)
btn_pipe_clear = pn.widgets.Button(name="Clear Queue",   button_type="default", width=130)
pipe_params_col = pn.Column(sizing_mode="stretch_width")
queue_col = pn.Column(sizing_mode="stretch_width")

# ── Param widgets ──────────────────────────────────────────────────────────────
pw_num_jobs    = pn.widgets.IntInput(  name="num_jobs",           value=32,  step=8,   width=130)
pw_kernel_size = pn.widgets.IntInput(  name="kernel_size",        value=2,   step=1,   width=130)
pw_stride      = pn.widgets.IntInput(  name="stride",             value=2,   step=1,   width=110)
pw_padding     = pn.widgets.IntInput(  name="padding",            value=0,             width=110)
pw_mask_val    = pn.widgets.FloatInput(name="mask_val",           value=1.0, step=0.1, width=130)
pw_region_rs   = pn.widgets.IntInput(  name="region row start",   value=0,   width=130)
pw_region_re   = pn.widgets.IntInput(  name="region row end",     value=10,  width=130)
pw_region_cs   = pn.widgets.IntInput(  name="region col start",   value=0,   width=130)
pw_region_ce   = pn.widgets.IntInput(  name="region col end",     value=10,  width=130)
pw_region_mode = pn.widgets.Select(    name="mode", options=["zeros","median","ones"], value="median", width=120)
pw_est_ttheta  = pn.widgets.FloatInput(name="est_ttheta (deg)",   value=1.0, step=0.1, width=160)
pw_method      = pn.widgets.Select(    name="method", options=["BFGS","L-BFGS-B","Nelder-Mead"], value="BFGS", width=150)
pw_gtol        = pn.widgets.FloatInput(name="gtol",               value=1e-6, width=130)
pw_alpha       = pn.widgets.FloatInput(name="alpha shift",        value=0.0, step=1e-4, width=130)
pw_beta        = pn.widgets.FloatInput(name="beta shift",         value=0.0, step=1e-4, width=130)
pw_gamma       = pn.widgets.FloatInput(name="gamma shift",        value=0.0, step=1e-4, width=130)
pw_var_thresh  = pn.widgets.FloatInput(name="variance threshold", value=0.1, step=0.01, width=180)
pw_sigma       = pn.widgets.FloatInput(name="sigma",              value=2.0, step=0.5,  width=120)
pw_med_thresh  = pn.widgets.FloatInput(name="threshold",          value=3.0, step=0.5,  width=120)
pw_det_thresh  = pn.widgets.FloatInput(name="detection threshold",value=0.1, step=0.01, width=180)
pw_det_min     = pn.widgets.FloatInput(name="min object sum",     value=10.0, step=1.0, width=160)
pw_det_max     = pn.widgets.FloatInput(name="max object sum",     value=1e6, step=100., width=160)
pw_coh_thresh  = pn.widgets.FloatInput(name="threshold value",    value=0.0, step=0.1,  width=160)
pw_cmap        = pn.widgets.Select(    name="colormap", options=["viridis","jet","plasma","inferno","gray","hot"], value="viridis", width=140)
pw_log_scale   = pn.widgets.Checkbox(  name="log scale", value=False)
pw_vmin        = pn.widgets.FloatInput(name="vmin (blank=auto)", width=130)
pw_vmax        = pn.widgets.FloatInput(name="vmax (blank=auto)", width=130)
pw_4d_px       = pn.widgets.IntInput(  name="detector row (px)", value=0, width=150)
pw_4d_py       = pn.widgets.IntInput(  name="detector col (py)", value=0, width=150)
pw_save_dir    = pn.widgets.TextInput( name="output directory", placeholder="/path/to/out/", sizing_mode="stretch_width")
pw_save_name   = pn.widgets.TextInput( name="file name (no ext)", placeholder="export", width=200)


def _refresh_pipe_params(event=None):
    op = w_pipe_op.value
    pipe_params_col.clear()
    if op in ("Make 4D dataset", "Sum detector frames"):
        pipe_params_col.append(pw_num_jobs)
    elif op == "Pool detector":
        pipe_params_col.append(pn.Row(pw_kernel_size, pw_stride, pw_padding))
    elif op in ("Mask region detector", "Mask region cohimgs"):
        pipe_params_col.append(pn.Row(pw_region_rs, pw_region_re, pw_region_cs, pw_region_ce))
        pipe_params_col.append(pw_region_mode)
    elif op == "Make k-outs":
        pipe_params_col.append(pw_mask_val)
    elif op == "Compute kins":
        pipe_params_col.append(pn.Row(pw_est_ttheta, pw_method, pw_gtol))
    elif op == "Compute kins w geometry":
        pipe_params_col.append(pn.Row(pw_method, pw_gtol))
    elif op == "Refine kins":
        pipe_params_col.append(pn.Row(pw_alpha, pw_beta, pw_gamma))
    elif op == "Filter coherent images":
        pipe_params_col.append(pw_var_thresh)
    elif op == "Remove background":
        pipe_params_col.append(pw_sigma)
    elif op == "Median filter (coh)":
        pipe_params_col.append(pn.Row(pw_kernel_size, pw_stride, pw_med_thresh))
    elif op == "Detect objects":
        pipe_params_col.append(pn.Row(pw_det_thresh, pw_det_min, pw_det_max))
    elif op == "Mask cohimgs threshold":
        pipe_params_col.append(pw_coh_thresh)
    elif op.startswith("Plot:"):
        pipe_params_col.append(pn.Row(pw_cmap, pw_log_scale, pw_vmin, pw_vmax))
        if op == "Plot: 4D viewer (single)":
            pipe_params_col.append(pn.Row(pw_4d_px, pw_4d_py))
    elif op.startswith("Save:"):
        pipe_params_col.append(pw_save_dir)
        pipe_params_col.append(pw_save_name)
    elif op == "Estimate 2θ":
        pipe_params_col.append(pn.pane.HTML(
            "<span style='color:var(--txt-muted);font-size:12px;font-family:JetBrains Mono,monospace'>"
            "Requires a designated Pupil ROI with k-outs computed.</span>"))


w_pipe_op.param.watch(_refresh_pipe_params, "value")
_refresh_pipe_params()


def _snapshot_params(op: str) -> dict:
    p = {}
    if op in ("Make 4D dataset", "Sum detector frames"):
        p["num_jobs"] = pw_num_jobs.value
    elif op == "Pool detector":
        p.update(kernel_size=pw_kernel_size.value, stride=pw_stride.value, padding=pw_padding.value)
    elif op in ("Mask region detector", "Mask region cohimgs"):
        p.update(region=[pw_region_rs.value, pw_region_re.value,
                         pw_region_cs.value, pw_region_ce.value],
                 mode=pw_region_mode.value)
    elif op == "Make k-outs":
        p["mask_val"] = pw_mask_val.value
    elif op == "Compute kins":
        p.update(est_ttheta=pw_est_ttheta.value, method=pw_method.value, gtol=pw_gtol.value)
    elif op == "Compute kins w geometry":
        p.update(method=pw_method.value, gtol=pw_gtol.value)
    elif op == "Refine kins":
        p.update(alpha=pw_alpha.value, beta=pw_beta.value, gamma=pw_gamma.value)
    elif op == "Filter coherent images":
        p["var_thresh"] = pw_var_thresh.value
    elif op == "Remove background":
        p["sigma"] = pw_sigma.value
    elif op == "Median filter (coh)":
        p.update(kernel_size=pw_kernel_size.value, stride=pw_stride.value, threshold=pw_med_thresh.value)
    elif op == "Detect objects":
        p.update(det_thresh=pw_det_thresh.value, det_min=pw_det_min.value, det_max=pw_det_max.value)
    elif op == "Mask cohimgs threshold":
        p["coh_thresh"] = pw_coh_thresh.value
    elif op.startswith("Plot:"):
        p.update(cmap=pw_cmap.value, log_scale=pw_log_scale.value,
                 vmin=pw_vmin.value or None, vmax=pw_vmax.value or None)
        if op == "Plot: 4D viewer (single)":
            p.update(px=pw_4d_px.value, py=pw_4d_py.value)
    elif op.startswith("Save:"):
        p.update(save_dir=pw_save_dir.value.strip(),
                 save_name=pw_save_name.value.strip() or "ptyxplore_export")
    return p


def _refresh_pipe_roi_select():
    names = list(state["rois"].keys())
    opts  = names if names else ["— none —"]
    w_pipe_roi.options = opts
    if w_pipe_roi.value not in names:
        w_pipe_roi.value = names[0] if names else "— none —"
    w_4dv_roi.options = opts
    if w_4dv_roi.value not in names:
        w_4dv_roi.value = names[0] if names else "— none —"


def _rebuild_queue():
    queue_col.clear()
    if not pipeline_queue:
        queue_col.append(pn.pane.HTML(
            "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
            "Queue is empty. Add steps above.</span>"))
        return
    for i, item in enumerate(pipeline_queue):
        cls = {"pending":"badge-idle","running":"badge-run","done":"badge-ok","error":"badge-err"}.get(item["status"],"badge-idle")
        pstr = ", ".join(f"{k}={v}" for k, v in item["params"].items())
        if len(pstr) > 55:
            pstr = pstr[:55] + "…"
        row_html = pn.pane.HTML(
            f"<div class='queue-row'>"
            f"<span class='q-idx'>#{i+1}</span>"
            f"<span class='q-op'>{item['op']}</span>"
            f"<span class='q-roi'>{item['roi_name'] or '—'}</span>"
            f"<span class='q-par'>{pstr}</span>"
            f"<span><span class='badge {cls}'>{item['status'].upper()}</span></span>"
            f"</div>", sizing_mode="stretch_width")

        def _make_rm(idx=i):
            btn = pn.widgets.Button(name="✕", button_type="default", width=36)
            def _rm(event):
                if idx < len(pipeline_queue):
                    pipeline_queue.pop(idx)
                    _rebuild_queue()
            btn.on_click(_rm)
            return btn

        queue_col.append(pn.Row(row_html, _make_rm(), align="center", sizing_mode="stretch_width"))


def on_pipe_add(event):
    op       = w_pipe_op.value
    roi_name = w_pipe_roi.value if w_pipe_roi.value != "— none —" else None
    if _ALL_OPS.get(op, {}).get("target") == "roi" and roi_name is None:
        log("Select a target ROI for this operation.", "warn")
        pn.state.notifications.warning("Select a target ROI.", duration=3000); return
    pipeline_queue.append({"op": op, "roi_name": roi_name,
                           "params": _snapshot_params(op), "status": "pending"})
    _rebuild_queue()
    log(f"Queued: {op} → {roi_name or 'scan'}", "info")


def _execute_step(item: dict) -> None:
    op       = item["op"]
    roi_name = item["roi_name"]
    params   = item["params"]
    roi      = state["rois"].get(roi_name) if roi_name else None
    scan     = state["scan"]
    pupil    = state["rois"].get(state["pupil_name"]) if state["pupil_name"] else None

    from ptyxplore import roi_ops

    if op == "Sum detector frames":
        result = roi_ops.sum_detector_frames(scan, num_jobs=params.get("num_jobs", 32))
        state["current_frame"] = state["processed_frame"] = result
        log(f"Sum done ✓  shape={result.shape}", "ok")
    elif op == "Make 4D dataset":
        roi_ops.make_4d_dataset(roi, num_jobs=params.get("num_jobs", 32))
        _add_roi_history(roi_name, f"Make 4D dataset → shape={roi.data_4d.shape}")
        if w_4dv_roi.value == roi_name:
            _on_4dv_roi_change(type('_', (), {'new': roi_name})())
    elif op == "Pool detector":
        roi_ops.pool_detector(roi, params["kernel_size"], params["stride"], params["padding"])
        _add_roi_history(roi_name, f"Pool detector → {roi.data_4d.shape}")
    elif op in ("Mask region detector", "Mask region cohimgs"):
        fn = roi_ops.mask_region_detector if op == "Mask region detector" else roi_ops.mask_region_cohimgs
        fn(roi, params["region"], params["mode"])
        _add_roi_history(roi_name, f"{op} ✓")
    elif op == "Make k-outs":
        roi_ops.make_kouts(roi, params["mask_val"])
        _add_roi_history(roi_name, f"Make k-outs → n={len(roi.kout_coords)}")
    elif op == "Estimate 2θ":
        roi_ops.estimate_ttheta(roi, pupil)
        _add_roi_history(roi_name, f"Estimate 2θ → {roi.est_ttheta:.3f}°")
    elif op == "Compute kins":
        roi_ops.compute_kins(roi, pupil, params["est_ttheta"], params["method"], params["gtol"])
        _add_roi_history(roi_name, f"Compute kins → n={len(roi.kins)}")
    elif op == "Compute kins w geometry":
        roi_ops.compute_kins_w_geometry(roi, pupil, params["method"], params["gtol"])
        _add_roi_history(roi_name, "Compute kins w geometry ✓")
    elif op == "Refine kins":
        roi_ops.refine_kins(roi, (params["alpha"], params["beta"], params["gamma"]))
        _add_roi_history(roi_name, "Refine kins ✓")
    elif op == "Make coherent images":
        roi_ops.make_coherent_images(roi)
        _add_roi_history(roi_name, f"Make coherent images → n={len(roi.coherent_imgs)}")
    elif op == "Filter coherent images":
        roi_ops.filter_coherent_images(roi, params["var_thresh"])
        _add_roi_history(roi_name, f"Filter coherent images → n={len(roi.coherent_imgs)}")
    elif op == "Remove background":
        roi_ops.remove_coh_background(roi, params["sigma"])
        _add_roi_history(roi_name, "Remove background ✓")
    elif op == "Median filter (coh)":
        roi_ops.apply_median_filter(roi, params["kernel_size"], params["stride"], params["threshold"])
        _add_roi_history(roi_name, "Median filter (coh) ✓")
    elif op == "Detect objects":
        roi_ops.detect_object(roi, params["det_thresh"], params["det_min"], params["det_max"])
        _add_roi_history(roi_name, f"Detect objects → n={len(roi.detected_objects)}")
    elif op == "Align coherent images":
        roi_ops.align_coherent_images(roi)
        _add_roi_history(roi_name, "Align coherent images ✓")
    elif op == "Mask cohimgs threshold":
        roi_ops.mask_cohimgs_threshold(roi, params["coh_thresh"])
        _add_roi_history(roi_name, "Mask cohimgs threshold ✓")
    elif op == "Plot: Average detector":
        if roi.averaged_det_images is None:
            raise ValueError("Run Make 4D dataset first.")
        fig = _make_image_fig(roi.averaged_det_images, f"Avg Detector — {roi_name}",
                              params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Average Detector — {roi_name}", fig)
        _add_roi_history(roi_name, "Plot: Average detector → gallery")
    elif op == "Plot: Averaged coherent":
        if roi.averaged_coherent_images is None:
            raise ValueError("averaged_coherent_images not available.")
        fig = _make_image_fig(roi.averaged_coherent_images, f"Avg Coherent — {roi_name}",
                              params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Averaged Coherent — {roi_name}", fig)
        _add_roi_history(roi_name, "Plot: Averaged coherent → gallery")
    elif op == "Plot: Coherent sequence":
        if roi.coherent_imgs is None:
            raise ValueError("coherent_imgs not available.")
        for idx_img, img in enumerate(roi.coherent_imgs):
            fig = _make_image_fig(img, f"Coherent [{idx_img}] — {roi_name}",
                                  params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
            _add_to_gallery(f"Coherent [{idx_img}] — {roi_name}", fig)
        _add_roi_history(roi_name, f"Plot: Coherent sequence → {len(roi.coherent_imgs)} frames to gallery")
    elif op == "Plot: 4D viewer (single)":
        if roi.data_4d is None:
            raise ValueError("Run Make 4D dataset first.")
        from ptyxplore.data_proc import make_coherent_image, make_detector_image
        px, py = params["px"], params["py"]
        coh = make_coherent_image(roi.data_4d, np.array([px, py]))
        det = make_detector_image(roi.data_4d, np.array([0, 0]))
        _add_to_gallery(f"4D Coherent px=({px},{py}) — {roi_name}",
                        _make_image_fig(coh, f"Coherent px=({px},{py})", params["log_scale"], "plasma", None, None, (5,4)))
        _add_to_gallery(f"4D Detector — {roi_name}",
                        _make_image_fig(det, f"Detector", params["log_scale"], params["cmap"], None, None, (5,4)))
        _add_roi_history(roi_name, f"Plot: 4D viewer (px={px},{py}) → gallery")
    elif op == "Plot: K-outs":
        if roi.averaged_det_images is None or roi.kout_coords is None:
            raise ValueError("Compute k-outs first.")
        fig = _make_kcoord_fig(roi.averaged_det_images, roi.kout_coords, roi.coords, params["cmap"], f"K-outs — {roi_name}")
        _add_to_gallery(f"K-outs — {roi_name}", fig)
        _add_roi_history(roi_name, "Plot: K-outs → gallery")
    elif op == "Plot: K-ins":
        if pupil is None or pupil.averaged_det_images is None:
            raise ValueError("Pupil ROI with averaged_det_images required.")
        if roi.kin_coords is None:
            raise ValueError("Compute kins first.")
        fig = _make_kcoord_fig(pupil.averaged_det_images, roi.kin_coords, pupil.coords, params["cmap"], f"K-ins — {roi_name}")
        _add_to_gallery(f"K-ins — {roi_name}", fig)
        _add_roi_history(roi_name, "Plot: K-ins → gallery")
    elif op == "Save: ROI data (HDF5)":
        outdir = params["save_dir"]
        if not outdir:
            raise ValueError("Output directory required.")
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, params["save_name"] + ".h5")
        roi.save_data(path)
        _add_roi_history(roi_name, f"Save: ROI data → {path}")
        log(f"ROI data saved → {path}", "ok")
    elif op in ("Save: Frame (NumPy)", "Save: Frame (TIFF)"):
        outdir = params["save_dir"]
        data   = state["current_frame"]
        if data is None:
            raise ValueError("No frame loaded.")
        if not outdir:
            raise ValueError("Output directory required.")
        os.makedirs(outdir, exist_ok=True)
        if op == "Save: Frame (NumPy)":
            path = os.path.join(outdir, params["save_name"] + ".npy")
            np.save(path, data)
        else:
            from PIL import Image as PILImage
            path = os.path.join(outdir, params["save_name"] + ".tif")
            norm = ((data - data.min()) / (data.max() - data.min() + 1e-12) * 65535).astype(np.uint16)
            PILImage.fromarray(norm).save(path)
        log(f"Frame saved → {path}", "ok")


def on_pipe_run(event):
    pending = [i for i, it in enumerate(pipeline_queue) if it["status"] == "pending"]
    if not pending:
        log("No pending items in queue.", "warn")
        pn.state.notifications.warning("No pending items.", duration=2000); return
    for i in pending:
        item = pipeline_queue[i]
        item["status"] = "running"
        _rebuild_queue()
        try:
            _execute_step(item)
            item["status"] = "done"
            log(f"✓ [{i+1}] {item['op']}", "ok")
        except Exception as exc:
            item["status"] = "error"
            log(f"✗ [{i+1}] {item['op']}: {exc}", "err")
            pn.state.notifications.error(f"Step {i+1} failed: {exc}", duration=6000)
        _rebuild_queue()
    _rebuild_roi_cards()
    pn.state.notifications.success("Pipeline run complete.", duration=3000)


def on_pipe_clear(event):
    pipeline_queue.clear()
    _rebuild_queue()
    log("Queue cleared.", "info")


btn_pipe_add.on_click(on_pipe_add)
btn_pipe_run.on_click(on_pipe_run)
btn_pipe_clear.on_click(on_pipe_clear)
_rebuild_queue()

tab_pipeline = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Add Step</p>", sizing_mode="stretch_width"),
        pn.Row(w_pipe_op, w_pipe_roi, align="end"),
        pipe_params_col,
        pn.Row(btn_pipe_add),
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Queue</p>", sizing_mode="stretch_width"),
        queue_col,
        pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
        pn.Row(btn_pipe_run, btn_pipe_clear),
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

w_file_idx  = pn.widgets.IntSlider(name="File index",  start=0, end=0, value=0, width=320)
w_frame_idx = pn.widgets.IntSlider(name="Frame index", start=0, end=0, value=0, width=320)
w_log_scale = pn.widgets.Checkbox(name="Log scale", value=True)
w_cmap_vis  = pn.widgets.Select(
    name="Colormap", options=["viridis","jet","plasma","inferno","gray","hot","turbo"], value="viridis", width=140)
w_vmin_vis  = pn.widgets.FloatInput(name="vmin (blank=auto)", width=160)
w_vmax_vis  = pn.widgets.FloatInput(name="vmax (blank=auto)", width=160)
btn_load_frame = pn.widgets.Button(name="Load Frame",   button_type="primary", width=160)
btn_avg_frame  = pn.widgets.Button(name="Show Average", button_type="default",  width=160)
vis_info = pn.pane.HTML("", sizing_mode="stretch_width")
vis_pane = pn.pane.Matplotlib(
    _placeholder_fig("Load a scan, then click 'Load Frame'"),
    tight=True, sizing_mode="stretch_width", height=500)

btn_gallery_clear = pn.widgets.Button(name="Clear All Plots", button_type="default", width=160)


def _load_h5_frame(file_path: str, frame_idx: int):
    with h5py.File(file_path, "r") as f:
        dset = f["/entry/data/data"]
        n = dset.shape[0]
        fi = min(frame_idx, n - 1)
        return dset[fi, :, :], n


def on_load_frame(event):
    scan = state["scan"]
    if scan is None:
        log("Load a scan first.", "warn")
        pn.state.notifications.warning("Load a scan first.", duration=3000); return
    fidx  = w_file_idx.value
    fridx = w_frame_idx.value
    fpath = scan.file_list[fidx]
    log(f"Loading file {fidx+1}/{len(scan.file_list)}, frame {fridx}…", "info")
    try:
        frame, n_frames = _load_h5_frame(fpath, fridx)
        w_frame_idx.end = max(0, n_frames - 1)
        if fridx >= n_frames:
            w_frame_idx.value = 0; fridx = 0
        state["current_frame"]   = frame.copy()
        state["processed_frame"] = frame.copy()
        fig = _make_image_fig(frame,
            f"File {fidx+1}/{len(scan.file_list)} · Frame {fridx}/{n_frames-1}",
            w_log_scale.value, w_cmap_vis.value, w_vmin_vis.value or None, w_vmax_vis.value or None)
        _set_pane_fig(vis_pane, fig, "vis")
        vis_info.object = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted);margin-top:6px'>"
            f"shape: {frame.shape} &nbsp;|&nbsp; dtype: {frame.dtype} &nbsp;|&nbsp; "
            f"min: {frame.min():.0f} &nbsp;|&nbsp; max: {frame.max():.0f} &nbsp;|&nbsp; "
            f"mean: {frame.mean():.1f}</div>")
        log(f"Frame loaded ✓  shape={frame.shape}, max={frame.max():.0f}", "ok")
    except Exception as exc:
        log(f"Failed to load frame: {exc}", "err")
        pn.state.notifications.error(str(exc), duration=5000)


def on_avg_frame(event):
    scan = state["scan"]
    if scan is None:
        log("Load a scan first.", "warn"); return
    fidx  = w_file_idx.value
    fpath = scan.file_list[fidx]
    log(f"Computing average of all frames in file {fidx+1}…", "info")
    try:
        with h5py.File(fpath, "r") as f:
            avg = f["/entry/data/data"][...].mean(axis=0)
        state["current_frame"]   = avg.copy()
        state["processed_frame"] = avg.copy()
        fig = _make_image_fig(avg, f"Average — File {fidx+1}/{len(scan.file_list)}",
                              w_log_scale.value, w_cmap_vis.value,
                              w_vmin_vis.value or None, w_vmax_vis.value or None)
        _set_pane_fig(vis_pane, fig, "vis")
        vis_info.object = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted);margin-top:6px'>"
            f"shape: {avg.shape} &nbsp;|&nbsp; min: {avg.min():.1f} &nbsp;|&nbsp; "
            f"max: {avg.max():.1f} &nbsp;|&nbsp; mean: {avg.mean():.2f}</div>")
        log(f"Average computed ✓  shape={avg.shape}", "ok")
    except Exception as exc:
        log(f"Failed to compute average: {exc}", "err")
        pn.state.notifications.error(str(exc), duration=5000)


def on_gallery_clear(event):
    for item in plot_gallery:
        try:
            plt.close(item["fig"])
        except Exception:
            pass
    plot_gallery.clear()
    _rebuild_gallery()
    log("Plot gallery cleared.", "info")


btn_load_frame.on_click(on_load_frame)
btn_avg_frame.on_click(on_avg_frame)
btn_gallery_clear.on_click(on_gallery_clear)
_rebuild_gallery()

# ─── 4D Dataset Viewer ───────────────────────────────────────────────────────
w_4dv_roi  = pn.widgets.Select(name="ROI", options=["— none —"], value="— none —", width=200)
w_4dv_prow = pn.widgets.IntSlider(name="py (det row)", start=0, end=1, value=0, width=300)
w_4dv_pcol = pn.widgets.IntSlider(name="px (det col)", start=0, end=1, value=0, width=300)
w_4dv_lrow = pn.widgets.IntSlider(name="ly (scan row)", start=0, end=1, value=0, width=300)
w_4dv_lcol = pn.widgets.IntSlider(name="lx (scan col)", start=0, end=1, value=0, width=300)
w_4dv_cmap_coh = pn.widgets.Select(name="coh cmap", options=["plasma","viridis","jet","gray","hot"], value="plasma", width=130)
w_4dv_cmap_det = pn.widgets.Select(name="det cmap", options=["viridis","plasma","jet","gray","hot"], value="viridis", width=130)

fourD_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", tight=True, height=420)
fourD_msg  = pn.pane.HTML("", sizing_mode="stretch_width")
_fourD_fig = [None]


def _update_4d_viewer(*_):
    roi_name = w_4dv_roi.value
    if not roi_name or roi_name == "— none —" or roi_name not in state["rois"]:
        fourD_msg.object = "<span style='color:var(--txt-muted)'>Select a ROI with 4D data loaded.</span>"
        fourD_pane.object = None
        return
    roi = state["rois"][roi_name]
    if roi.data_4d is None:
        fourD_msg.object = "<span style='color:var(--warn)'>Run Make 4D dataset on this ROI first.</span>"
        fourD_pane.object = None
        return
    fourD_msg.object = ""
    from ptyxplore.data_proc import make_coherent_image, make_detector_image
    prow, pcol = w_4dv_prow.value, w_4dv_pcol.value
    lrow, lcol = w_4dv_lrow.value, w_4dv_lcol.value
    coh = make_coherent_image(roi.data_4d, np.array([prow, pcol]))
    det = make_detector_image(roi.data_4d, np.array([lrow, lcol]))
    if _fourD_fig[0] is not None:
        plt.close(_fourD_fig[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m = np.mean(coh)
    im0 = axes[0].imshow(coh, cmap=w_4dv_cmap_coh.value, vmin=m - 0.05 * abs(m), vmax=m + 0.05 * abs(m))
    axes[0].set_title(f"Coherent  px=({pcol}, {prow})")
    axes[0].add_patch(Rectangle((lcol - 1, lrow - 1), 2, 2, edgecolor='white', facecolor='none', lw=1.5))
    plt.colorbar(im0, ax=axes[0])
    m2 = np.mean(det)
    im1 = axes[1].imshow(det, cmap=w_4dv_cmap_det.value, vmin=m2 - 0.3 * abs(m2), vmax=m2 + 0.2 * abs(m2))
    axes[1].set_title(f"Detector  lx=({lcol}, {lrow})")
    axes[1].add_patch(Rectangle((pcol - 2, prow - 2), 4, 4, edgecolor='white', facecolor='none', lw=1.5))
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    _fourD_fig[0] = fig
    fourD_pane.object = fig
    plt.close(fig)


def _on_4dv_roi_change(event):
    roi_name = event.new
    if not roi_name or roi_name == "— none —" or roi_name not in state["rois"]:
        return
    roi = state["rois"][roi_name]
    if roi.data_4d is None:
        _update_4d_viewer()
        return
    dsh, csh = roi.data_4d.shape[2:], roi.data_4d.shape[:2]
    w_4dv_prow.end = dsh[0] - 1;  w_4dv_prow.value = dsh[0] // 2
    w_4dv_pcol.end = dsh[1] - 1;  w_4dv_pcol.value = dsh[1] // 2
    w_4dv_lrow.end = csh[0] - 1;  w_4dv_lrow.value = csh[0] // 2
    w_4dv_lcol.end = csh[1] - 1;  w_4dv_lcol.value = csh[1] // 2
    _update_4d_viewer()


w_4dv_roi.param.watch(_on_4dv_roi_change, 'value')
for _w4 in (w_4dv_prow, w_4dv_pcol, w_4dv_lrow, w_4dv_lcol, w_4dv_cmap_coh, w_4dv_cmap_det):
    _w4.param.watch(_update_4d_viewer, 'value')

tab_vis = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Frame Browser</p>", sizing_mode="stretch_width"),
        pn.Row(w_file_idx, w_frame_idx),
        pn.Row(w_log_scale, w_cmap_vis, w_vmin_vis, w_vmax_vis, align="end"),
        pn.Row(btn_load_frame, btn_avg_frame),
        vis_pane, vis_info,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>4D Dataset Viewer</p>", sizing_mode="stretch_width"),
        pn.Row(w_4dv_roi, w_4dv_cmap_coh, w_4dv_cmap_det, align="end"),
        pn.Row(w_4dv_prow, w_4dv_pcol),
        pn.Row(w_4dv_lrow, w_4dv_lcol),
        fourD_msg, fourD_pane,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Pipeline Outputs</p>", sizing_mode="stretch_width"),
        pn.Row(btn_gallery_clear),
        gallery_col,
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PYTHON
# ═══════════════════════════════════════════════════════════════════════════════

_CONSOLE_PRELUDE = """\
# Available in this namespace:
#   state          — dict with keys: rois, scan, exp, current_frame, roi_history
#   state['rois']  — dict of name → ROI objects
#   np, plt, pn    — numpy, matplotlib.pyplot, panel
#
# Example:
#   roi = state['rois']['bragg_001']
#   print(roi.data_4d.shape)
"""

console_editor = pn.widgets.CodeEditor(
    value=_CONSOLE_PRELUDE,
    language="python",
    theme="monokai",
    sizing_mode="stretch_width",
    height=320,
)
btn_console_run   = pn.widgets.Button(name="▶  Run", button_type="primary", width=120)
btn_console_clear = pn.widgets.Button(name="Clear output", button_type="default", width=130)
console_output = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
    "Output will appear here.</span>", sizing_mode="stretch_width")
_console_ns: dict = {}


def on_console_run(event):
    import io, contextlib
    code = console_editor.value
    if not code.strip():
        return
    _console_ns.update({"state": state, "np": np, "plt": plt, "pn": pn})
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, _console_ns)  # noqa: S102
        out = buf.getvalue()
        console_output.object = (
            f"<div style='background:{BG_DARK};border:1px solid {BORDER};border-radius:8px;"
            f"padding:14px;font-family:JetBrains Mono,monospace;font-size:12px;"
            f"color:{ACCENT};white-space:pre-wrap;max-height:400px;overflow-y:auto'>"
            f"{out or '(no output)'}</div>")
    except Exception:
        tb = traceback.format_exc()
        console_output.object = (
            f"<div style='background:{BG_DARK};border:1px solid {BORDER};border-radius:8px;"
            f"padding:14px;font-family:JetBrains Mono,monospace;font-size:12px;"
            f"color:{WARN};white-space:pre-wrap;max-height:400px;overflow-y:auto'>"
            f"{tb}</div>")


def on_console_clear(event):
    console_output.object = (
        "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
        "Output cleared.</span>")


btn_console_run.on_click(on_console_run)
btn_console_clear.on_click(on_console_clear)

tab_python = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Python Console</p>", sizing_mode="stretch_width"),
        pn.pane.HTML(
            "<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--txt-muted);margin-bottom:10px'>"
            "Namespace: <span style='color:var(--accent)'>state</span> &nbsp;·&nbsp; "
            "<span style='color:var(--accent)'>np</span> &nbsp;·&nbsp; "
            "<span style='color:var(--accent)'>plt</span> &nbsp;·&nbsp; "
            "<span style='color:var(--accent)'>pn</span></div>",
            sizing_mode="stretch_width"),
        console_editor,
        pn.Row(btn_console_run, btn_console_clear),
        pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
        console_output,
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs + main dashboard
# ═══════════════════════════════════════════════════════════════════════════════

tabs = pn.Tabs(
    ("Loading",      tab_loading),
    ("ROI Manager",  tab_roi_manager),
    ("Pipeline",     tab_pipeline),
    ("Visualisation",tab_vis),
    ("Python",       tab_python),
    sizing_mode="stretch_width",
    dynamic=True,
)

dashboard = pn.template.FastListTemplate(
    title="Ptycho Dashboard",
    theme="dark",
    accent_base_color=ACCENT,
    header_background=BG_MID,
    header_color=TXT,
    sidebar_width=0,
    raw_css=[CSS],
    main=[
        pn.pane.HTML(
            f"""<div class="dash-header">
              <div>
                <p class="dash-title">⬡ PTYCHO DASHBOARD</p>
                <p class="dash-sub">CID · Beamline P11 · PETRA III · DESY</p>
              </div>
            </div>""",
            sizing_mode="stretch_width",
        ),
        tabs,
        card_log,
    ],
)

dashboard.servable()

if __name__ == "__main__":
    dashboard.show()
