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
from matplotlib.patches import Rectangle

import panel as pn
import numpy as np
from datetime import datetime
import h5py
import os
import traceback
import threading
import copy

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
    "scans":           {},    # label → Scan object
    "current_frame":   None,
    "processed_frame": None,
    "rois":            {},    # name → ROI object
    "pupil_name":      None,  # name of the ROI designated as pupil
    "roi_history":     {},    # name → list[str]
}
_prev_figs: dict = {}
pipeline_queue: list[dict] = []
plot_gallery:   list[dict] = []
roi_gallery_hooks: dict = {}   # roi_name → callable(title, fig) for per-ROI gallery
roi_4dv_hooks:    dict = {}    # roi_name → callable() to refresh per-ROI 4D viewer
_stop_event = threading.Event()  # set this to request a halt between pipeline steps


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


def _bk_palette(cmap_name: str, n: int = 256) -> list:
    """Convert a matplotlib colormap to a Bokeh-compatible hex palette list."""
    import matplotlib.colors as mcolors
    cmap = plt.get_cmap(cmap_name)
    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]


def _make_bk_image_panel(data: np.ndarray, title: str,
                          log_scale: bool = False, cmap: str = "viridis",
                          vmin=None, vmax=None, height: int = 440) -> pn.Column:
    """
    Return an interactive Panel component wrapping a Bokeh image figure.
    Features: scroll-to-zoom, pan, box-select for ROI, tap for point coords,
              vmin/vmax inputs, autoscale button, Log Point and Log ROI buttons.
    """
    from bokeh.models import (
        ColumnDataSource, LinearColorMapper, ColorBar,
        HoverTool, CustomJS, BasicTicker,
    )
    from bokeh.plotting import figure as bk_figure

    disp = np.log1p(np.abs(data.astype(float))) if log_scale else data.astype(float)
    h, w = disp.shape
    lo   = float(vmin) if vmin is not None else float(np.nanmin(disp))
    hi   = float(vmax) if vmax is not None else float(np.nanmax(disp))
    if abs(hi - lo) < 1e-10:
        hi = lo + 1.0

    bk_src = ColumnDataSource({'image': [disp], 'x': [0], 'y': [0], 'dw': [w], 'dh': [h]})
    bk_map = LinearColorMapper(palette=_bk_palette(cmap), low=lo, high=hi, nan_color='black')

    fig = bk_figure(
        title=title, sizing_mode="stretch_width", height=height,
        tools="pan,wheel_zoom,box_select,reset,tap", active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    fig.y_range.flipped = True
    fig.image('image', 'x', 'y', 'dw', 'dh', source=bk_src, color_mapper=bk_map)
    fig.add_tools(HoverTool(tooltips=[("col", "$x{0}"), ("row", "$y{0}"), ("val", "@image")]))
    fig.add_layout(ColorBar(color_mapper=bk_map, ticker=BasicTicker(), location=(0, 0)), 'right')

    cur_src = ColumnDataSource({'x': [0.0], 'y': [0.0]})
    sel_src = ColumnDataSource({'x0': [0.0], 'y0': [0.0], 'x1': [0.0], 'y1': [0.0]})
    fig.js_on_event('tap', CustomJS(args={'s': cur_src}, code="""
        s.data = {x: [cb_obj.x], y: [cb_obj.y]};  s.change.emit();
    """))
    fig.js_on_event('selectiongeometry', CustomJS(args={'s': sel_src}, code="""
        if (cb_obj.geometry && cb_obj.geometry.type === 'rect') {
            const g = cb_obj.geometry;
            s.data = {x0: [Math.min(g.x0,g.x1)], y0: [Math.min(g.y0,g.y1)],
                      x1: [Math.max(g.x0,g.x1)], y1: [Math.max(g.y0,g.y1)]};
            s.change.emit();
        }
    """))

    _cursor   = {'row': 0, 'col': 0}
    _sel      = {'rs': 0, 're': 0, 'cs': 0, 'ce': 0}
    coord_md  = pn.pane.HTML("", sizing_mode="stretch_width")
    fi_vmin   = pn.widgets.FloatInput(name="vmin", width=100)
    fi_vmax   = pn.widgets.FloatInput(name="vmax", width=100)
    btn_auto  = pn.widgets.Button(name="Autoscale", button_type="default", width=95)
    btn_lp    = pn.widgets.Button(name="Log Point [p]",  button_type="default", width=115)
    btn_lr    = pn.widgets.Button(name="Log ROI [r]",    button_type="default", width=105)

    def _fmt():
        coord_md.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted)'>"
            f"cursor → row={_cursor['row']}, col={_cursor['col']}"
            f" &nbsp;|&nbsp; box → row=[{_sel['rs']},{_sel['re']}]"
            f" col=[{_sel['cs']},{_sel['ce']}]</span>")

    def _on_cur(attr, old, new):
        _cursor['col'] = max(0, int(new['x'][0]))
        _cursor['row'] = max(0, int(new['y'][0]))
        pn.state.execute(_fmt)

    def _on_sel(attr, old, new):
        _sel['cs'] = max(0, int(new['x0'][0]))
        _sel['ce'] = max(0, int(new['x1'][0]))
        _sel['rs'] = max(0, int(new['y0'][0]))
        _sel['re'] = max(0, int(new['y1'][0]))
        pn.state.execute(_fmt)

    cur_src.on_change('data', _on_cur)
    sel_src.on_change('data', _on_sel)

    def _on_scale(event=None):
        _bk_map = bk_map
        _bk_map.low  = fi_vmin.value if fi_vmin.value is not None else float(np.nanmin(bk_src.data['image'][0]))
        _bk_map.high = fi_vmax.value if fi_vmax.value is not None else float(np.nanmax(bk_src.data['image'][0]))

    def _on_auto(event):
        fi_vmin.value = fi_vmax.value = None
        d = bk_src.data['image'][0]
        bk_map.low  = float(np.nanmin(d))
        bk_map.high = float(np.nanmax(d))

    fi_vmin.param.watch(_on_scale, 'value')
    fi_vmax.param.watch(_on_scale, 'value')
    btn_auto.on_click(_on_auto)

    def _log_pt(event):
        log(f"[{title}] Point → row={_cursor['row']}, col={_cursor['col']}", "info")

    def _log_roi(event):
        log(f"[{title}] ROI → row=[{_sel['rs']},{_sel['re']}], col=[{_sel['cs']},{_sel['ce']}]", "info")

    btn_lp.on_click(_log_pt)
    btn_lr.on_click(_log_roi)

    return pn.Column(
        fig,
        pn.Row(fi_vmin, fi_vmax, btn_auto, btn_lp, btn_lr, coord_md, align="center"),
        sizing_mode="stretch_width",
    )


def _add_roi_history(name: str, entry: str):
    if name and name in state["roi_history"]:
        state["roi_history"][name].append(f"[{ts()}] {entry}")


def _add_to_gallery(title: str, fig_or_panel, roi_name: str = None) -> None:
    """Add a matplotlib figure or a Panel component to the global gallery."""
    if isinstance(fig_or_panel, pn.viewable.Viewable):
        pane = fig_or_panel
    else:
        pane = pn.pane.Matplotlib(fig_or_panel, tight=True, sizing_mode="stretch_width", height=460)
        plt.close(fig_or_panel)
    plot_gallery.append({"title": title, "pane": pane, "fig": None, "roi_name": roi_name})
    _rebuild_gallery()
    if roi_name and roi_name in roi_gallery_hooks:
        roi_gallery_hooks[roi_name](title, fig_or_panel)


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

w_scan_num   = pn.widgets.IntInput( name="Scan Number",            value=1,   step=1,   width=140)
w_detector   = pn.widgets.Select(   name="Detector", options=["Eiger","Lambda"], value="Eiger", width=140)
w_sample     = pn.widgets.TextInput(name="Sample Name",            placeholder="e.g. SampleA_001", width=220)
w_det_dist   = pn.widgets.FloatInput(name="Detector Distance (m)", value=2.0, step=0.05, width=200)
w_scan_label = pn.widgets.TextInput(name="Scan label",             placeholder="auto", width=160)
btn_create_scan = pn.widgets.Button(name="Load Scan", button_type="primary", width=180, disabled=True)

_SCAN_STAT_LABELS = ["Scan #", "Files", "Energy (keV)", "Step X (nm)", "Step Y (nm)", "Points X×Y"]
loaded_scans_col = pn.Column(sizing_mode="stretch_width")


def _make_stats_html(scan) -> str:
    n_files = len(getattr(scan, "file_list", []) or [])
    energy  = getattr(scan, "energy", None)
    sx = getattr(scan, "step_size_x", None)
    sy = getattr(scan, "step_size_y", None)
    px = getattr(scan, "scan_x_steps_count", None)
    py = getattr(scan, "scan_y_steps_count", None)
    return (
        "<div class='stat-grid'>"
        f"<div class='stat-chip'><div class='val'>{scan.scan_num}</div><div class='lbl'>Scan #</div></div>"
        f"<div class='stat-chip'><div class='val'>{n_files}</div><div class='lbl'>Files</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{energy:.4f}' if energy else '—'}</div><div class='lbl'>Energy (keV)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{sx:.2f}' if sx else '—'}</div><div class='lbl'>Step X (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{sy:.2f}' if sy else '—'}</div><div class='lbl'>Step Y (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{f'{px}×{py}' if (px and py) else '—'}</div><div class='lbl'>Points X×Y</div></div>"
        "</div>"
    )


def _rebuild_loaded_scans():
    loaded_scans_col.clear()
    if not state["scans"]:
        loaded_scans_col.append(pn.pane.HTML(
            "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
            "No scans loaded yet.</span>"))
        return
    for lbl in state["scans"]:
        loaded_scans_col.append(pn.pane.HTML(
            f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
            f"color:var(--accent)'>● {lbl}</span> "
            f"<span style='color:var(--txt-muted);font-size:11px'>"
            f"({len(state['scans'][lbl].file_list)} files)</span>",
            sizing_mode="stretch_width"))


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
    label  = w_scan_label.value.strip() or f"scan_{snum}"
    if label in state["scans"]:
        log(f"Scan label '{label}' already in use.", "err")
        pn.state.notifications.error(f"Label '{label}' already loaded.", duration=3000); return
    log(f"Loading Scan {snum} (det={det}, dist={ddist} m, label='{label}')…", "info")
    try:
        from ptyxplore.cbd_loader import Scan
        scan = Scan(exp=state["exp"], scan_num=snum,
                    detector=det, sample_name=sample, det_distance=ddist)
        state["scans"][label] = scan
        _refresh_scan_select()
        _rebuild_loaded_scans()
        tabs.append((f"Scan {scan.scan_num}", _make_scan_tab(label)))
        log(f"Scan '{label}' loaded ✓  ({len(scan.file_list)} files)", "ok")
        _set_status(True)
        pn.state.notifications.success(f"Scan '{label}' loaded!", duration=3000)
    except Exception as exc:
        log(f"Failed to load Scan: {exc}", "err"); _set_status(False)
        pn.state.notifications.error(str(exc), duration=5000)


btn_create_exp.on_click(on_create_exp)
btn_create_scan.on_click(on_create_scan)

tab_loading = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>01 — Experiment</p>", sizing_mode="stretch_width"),
        pn.Row(w_year, w_btid, w_beamline, align="end"),
        exp_path_md,
        pn.Row(btn_create_exp, status_badge, align="center"),
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>02 — Load Scan</p>", sizing_mode="stretch_width"),
        pn.Row(w_scan_num, w_detector, w_sample, w_det_dist, align="end"),
        pn.Row(w_scan_label, align="end"),
        pn.Row(btn_create_scan),
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>03 — Loaded Scans</p>", sizing_mode="stretch_width"),
        loaded_scans_col,
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ROI MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

w_roi_scan = pn.widgets.Select(name="Scan", options=["— none —"], value="— none —", width=200)
w_roi_name = pn.widgets.TextInput(name="ROI name", placeholder="e.g. bragg_001", width=180)
w_roi_kind = pn.widgets.Select(name="Kind", options=["Bragg", "Pupil"], value="Bragg", width=120)
w_roi_rs   = pn.widgets.IntInput(name="row start", value=0,   width=110)
w_roi_re   = pn.widgets.IntInput(name="row end",   value=100, width=110)
w_roi_cs   = pn.widgets.IntInput(name="col start", value=0,   width=110)
w_roi_ce   = pn.widgets.IntInput(name="col end",   value=100, width=110)
w_roi_cx   = pn.widgets.FloatInput(name="centre px row", value=0.0, width=140)
w_roi_cy   = pn.widgets.FloatInput(name="centre px col", value=0.0, width=140)
btn_add_roi = pn.widgets.Button(name="Add ROI", button_type="primary", width=140)


def _refresh_scan_select():
    names = list(state["scans"].keys())
    opts  = names if names else ["— none —"]
    w_roi_scan.options = opts
    if w_roi_scan.value not in names:
        w_roi_scan.value = names[0] if names else "— none —"

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
            roi_4dv_hooks.pop(name, None)
            roi_gallery_hooks.pop(name, None)
            # Tab title format is "{scan_label}: {roi_name}" — remove any tab ending with ": {name}"
            tabs.objects = [(n, obj) for n, obj in tabs.objects if not n.endswith(f": {name}")]
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
    scan_label = w_roi_scan.value
    if scan_label == "— none —" or scan_label not in state["scans"]:
        log("Select a loaded scan first.", "warn")
        pn.state.notifications.warning("Select a scan first.", duration=3000); return
    scan = state["scans"][scan_label]
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
        roi = ROI(scan=scan, kind=kind, coords=coords, centre_pixel=centre)
        state["rois"][name] = roi
        state["roi_history"][name] = [f"[{ts()}] Created ({kind}, coords={coords})"]
        if kind == "pupil" and state["pupil_name"] is None:
            state["pupil_name"] = name
        _rebuild_roi_cards()
        _refresh_pipe_roi_select()
        tabs.append((f"{scan_label}: {name}", _make_roi_tab(name)))
        log(f"ROI '{name}' created ✓", "ok")
        pn.state.notifications.success(f"ROI '{name}' added!", duration=2000)
    except Exception as exc:
        log(f"Failed to create ROI: {exc}", "err")
        pn.state.notifications.error(str(exc), duration=5000)


btn_add_roi.on_click(on_add_roi)

tab_roi_manager = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Add ROI</p>", sizing_mode="stretch_width"),
        pn.Row(w_roi_scan, align="end"),
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
    _refresh_scan_select()


def _rebuild_queue(queue=None, col=None):
    if queue is None:
        queue = pipeline_queue
    if col is None:
        col = queue_col
    col.clear()
    if not queue:
        col.append(pn.pane.HTML(
            "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
            "Queue is empty. Add steps above.</span>"))
        return
    for i, item in enumerate(queue):
        cls     = {"pending":"badge-idle","running":"badge-run","done":"badge-ok","error":"badge-err"}.get(item["status"],"badge-idle")
        enabled = item.get("enabled", True)
        opacity = "1.0" if enabled else "0.4"
        pstr    = ", ".join(f"{k}={v}" for k, v in item["params"].items())
        if len(pstr) > 55:
            pstr = pstr[:55] + "…"
        row_html = pn.pane.HTML(
            f"<div class='queue-row' style='opacity:{opacity}'>"
            f"<span class='q-idx'>#{i+1}</span>"
            f"<span class='q-op'>{item['op']}</span>"
            f"<span class='q-roi'>{item['roi_name'] or '—'}</span>"
            f"<span class='q-par'>{pstr}</span>"
            f"<span><span class='badge {cls}'>{item['status'].upper()}</span></span>"
            f"</div>", sizing_mode="stretch_width")

        def _make_toggle(idx=i, q=queue, c=col):
            cb = pn.widgets.Checkbox(name="", value=q[idx].get("enabled", True), width=20, margin=(8, 4))
            def _toggle(event):
                q[idx]["enabled"] = event.new
                _rebuild_queue(q, c)
            cb.param.watch(_toggle, "value")
            return cb

        def _make_up(idx=i, q=queue, c=col):
            btn = pn.widgets.Button(name="▲", width=32, height=30, margin=(4, 1),
                                    disabled=(idx == 0))
            def _up(event):
                if idx > 0:
                    q[idx - 1], q[idx] = q[idx], q[idx - 1]
                    _rebuild_queue(q, c)
            btn.on_click(_up)
            return btn

        def _make_down(idx=i, q=queue, c=col):
            btn = pn.widgets.Button(name="▼", width=32, height=30, margin=(4, 1),
                                    disabled=(idx == len(q) - 1))
            def _down(event):
                if idx < len(q) - 1:
                    q[idx + 1], q[idx] = q[idx], q[idx + 1]
                    _rebuild_queue(q, c)
            btn.on_click(_down)
            return btn

        def _make_rm(idx=i, q=queue, c=col):
            btn = pn.widgets.Button(name="✕", button_type="default", width=36, height=30, margin=(4, 2))
            def _rm(event):
                if idx < len(q):
                    q.pop(idx)
                    _rebuild_queue(q, c)
            btn.on_click(_rm)
            return btn

        col.append(pn.Row(
            _make_toggle(), _make_up(), _make_down(),
            row_html, _make_rm(),
            align="center", sizing_mode="stretch_width"))


def on_pipe_add(event):
    op       = w_pipe_op.value
    roi_name = w_pipe_roi.value if w_pipe_roi.value != "— none —" else None
    if _ALL_OPS.get(op, {}).get("target") == "roi" and roi_name is None:
        log("Select a target ROI for this operation.", "warn")
        pn.state.notifications.warning("Select a target ROI.", duration=3000); return
    pipeline_queue.append({"op": op, "roi_name": roi_name,
                           "params": _snapshot_params(op), "status": "pending", "enabled": True})
    _rebuild_queue()
    log(f"Queued: {op} → {roi_name or 'scan'}", "info")


def _execute_step(item: dict) -> None:
    op       = item["op"]
    roi_name = item["roi_name"]
    params   = item["params"]
    roi      = state["rois"].get(roi_name) if roi_name else None
    scan     = roi.scan if roi else None
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
        if roi_name in roi_4dv_hooks:
            roi_4dv_hooks[roi_name]()
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
        panel = _make_bk_image_panel(
            roi.averaged_det_images, f"Avg Detector — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Average Detector — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, "Plot: Average detector → gallery")
    elif op == "Plot: Averaged coherent":
        if roi.averaged_coherent_images is None:
            raise ValueError("averaged_coherent_images not available.")
        panel = _make_bk_image_panel(
            roi.averaged_coherent_images, f"Avg Coherent — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Averaged Coherent — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, "Plot: Averaged coherent → gallery")
    elif op == "Plot: Coherent sequence":
        if roi.coherent_imgs is None:
            raise ValueError("coherent_imgs not available.")
        imgs = roi.coherent_imgs
        n = len(imgs)
        # Build a slider-based viewer as a single gallery item (avoids opening n figures)
        w_seq  = pn.widgets.IntSlider(name="Frame index", start=0, end=n - 1, value=0, width=320)
        info_h = pn.pane.HTML(
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--accent2)'>{n} frames — use slider to browse</span>")
        _init_fig = _make_image_fig(imgs[0], f"Coherent [0/{n-1}] — {roi_name}",
                                    params["log_scale"], params["cmap"],
                                    params["vmin"], params["vmax"])
        seq_pane   = pn.pane.Matplotlib(_init_fig, tight=True,
                                        sizing_mode="stretch_width", height=440)
        plt.close(_init_fig)
        _seq_fc = [None]

        def _upd_seq(event, _imgs=imgs, _rn=roi_name, _p=params):
            idx = w_seq.value
            if _seq_fc[0] is not None:
                plt.close(_seq_fc[0])
            fig = _make_image_fig(_imgs[idx], f"Coherent [{idx}/{len(_imgs)-1}] — {_rn}",
                                  _p["log_scale"], _p["cmap"], _p["vmin"], _p["vmax"])
            _seq_fc[0] = fig
            seq_pane.object = fig
            plt.close(fig)

        w_seq.param.watch(_upd_seq, "value")
        seq_component = pn.Column(info_h, w_seq, seq_pane, sizing_mode="stretch_width")
        _add_to_gallery(f"Coherent sequence ({n} frames) — {roi_name}",
                        seq_component, roi_name=roi_name)
        _add_roi_history(roi_name, f"Plot: Coherent sequence → {n} frames (slider viewer)")
    elif op == "Plot: 4D viewer (single)":
        if roi.data_4d is None:
            raise ValueError("Run Make 4D dataset first.")
        from ptyxplore.data_proc import make_coherent_image, make_detector_image
        px, py = params["px"], params["py"]
        coh = make_coherent_image(roi.data_4d, np.array([px, py]))
        det = make_detector_image(roi.data_4d, np.array([0, 0]))
        _add_to_gallery(f"4D Coherent px=({px},{py}) — {roi_name}",
                        _make_bk_image_panel(coh, f"Coherent px=({px},{py})", params["log_scale"], "plasma"),
                        roi_name=roi_name)
        _add_to_gallery(f"4D Detector — {roi_name}",
                        _make_bk_image_panel(det, f"Detector", params["log_scale"], params["cmap"]),
                        roi_name=roi_name)
        _add_roi_history(roi_name, f"Plot: 4D viewer (px={px},{py}) → gallery")
    elif op == "Plot: K-outs":
        if roi.averaged_det_images is None or roi.kout_coords is None:
            raise ValueError("Compute k-outs first.")
        fig = _make_kcoord_fig(roi.averaged_det_images, roi.kout_coords, roi.coords, params["cmap"], f"K-outs — {roi_name}")
        _add_to_gallery(f"K-outs — {roi_name}", fig, roi_name=roi_name)
        _add_roi_history(roi_name, "Plot: K-outs → gallery")
    elif op == "Plot: K-ins":
        if pupil is None or pupil.averaged_det_images is None:
            raise ValueError("Pupil ROI with averaged_det_images required.")
        if roi.kin_coords is None:
            raise ValueError("Compute kins first.")
        fig = _make_kcoord_fig(pupil.averaged_det_images, roi.kin_coords, pupil.coords, params["cmap"], f"K-ins — {roi_name}")
        _add_to_gallery(f"K-ins — {roi_name}", fig, roi_name=roi_name)
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


def _run_queue(queue, queue_col_ref, label, btn_run, btn_stop, btn_clear=None):
    """
    Execute a pipeline queue in a background thread.
    Checks _stop_event between each step so the Stop button can interrupt.
    All UI updates are dispatched via pn.state.execute (thread-safe).
    """
    pending = [i for i, it in enumerate(queue)
               if it["status"] == "pending" and it.get("enabled", True)]
    if not pending:
        log(f"[{label}] No pending items.", "warn")
        pn.state.notifications.warning("No pending items.", duration=2000)
        return

    _stop_event.clear()
    btn_run.disabled  = True
    btn_stop.disabled = False
    if btn_clear:
        btn_clear.disabled = True

    def _work():
        try:
            for i in pending:
                if _stop_event.is_set():
                    pn.state.execute(lambda: log(f"[{label}] Stopped by user.", "warn"))
                    break
                item = queue[i]
                item["status"] = "running"
                pn.state.execute(lambda: _rebuild_queue(queue, queue_col_ref))
                try:
                    _execute_step(item)
                    item["status"] = "done"
                    def _ok(idx=i, op=item["op"]):
                        log(f"[{label}] ✓ [{idx+1}] {op}", "ok")
                    pn.state.execute(_ok)
                except Exception as exc:
                    item["status"] = "error"
                    def _err(idx=i, op=item["op"], e=str(exc)):
                        log(f"[{label}] ✗ [{idx+1}] {op}: {e}", "err")
                        pn.state.notifications.error(f"{op} failed: {e}", duration=6000)
                    pn.state.execute(_err)
                pn.state.execute(lambda: _rebuild_queue(queue, queue_col_ref))
        finally:
            def _cleanup():
                _rebuild_roi_cards()
                btn_run.disabled  = False
                btn_stop.disabled = True
                if btn_clear:
                    btn_clear.disabled = False
                if not _stop_event.is_set():
                    pn.state.notifications.success(f"[{label}] Run complete.", duration=3000)
            pn.state.execute(_cleanup)

    threading.Thread(target=_work, daemon=True).start()


btn_pipe_stop = pn.widgets.Button(name="⏹  Stop", button_type="danger", width=120, disabled=True)


def on_pipe_run(event):
    if not btn_pipe_run.disabled:  # guard against double-click while starting
        _run_queue(pipeline_queue, queue_col, "global",
                   btn_pipe_run, btn_pipe_stop, btn_pipe_clear)


def on_pipe_stop(event):
    _stop_event.set()
    log("Stop requested — will halt after the current step finishes.", "warn")
    btn_pipe_stop.disabled = True


def on_pipe_clear(event):
    pipeline_queue.clear()
    _rebuild_queue()
    log("Queue cleared.", "info")


btn_pipe_add.on_click(on_pipe_add)
btn_pipe_run.on_click(on_pipe_run)
btn_pipe_stop.on_click(on_pipe_stop)
btn_pipe_clear.on_click(on_pipe_clear)
_rebuild_queue()
_rebuild_loaded_scans()

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
        pn.Row(btn_pipe_run, btn_pipe_stop, btn_pipe_clear),
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

w_file_idx  = pn.widgets.IntSlider(name="File index",  start=0, end=0, value=0, width=320)
w_frame_idx = pn.widgets.IntSlider(name="Frame index", start=0, end=0, value=0, width=320)
w_log_scale     = pn.widgets.Checkbox(name="Log scale",        value=True)
w_mask_hot      = pn.widgets.Checkbox(name="Mask hot pixels",  value=True)
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
        state["current_frame"] = frame.copy()
        if w_mask_hot.value:
            from ptyxplore.data_proc import mask_hot_pixels
            frame = mask_hot_pixels(frame)
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
        state["current_frame"] = avg.copy()
        if w_mask_hot.value:
            from ptyxplore.data_proc import mask_hot_pixels
            avg = mask_hot_pixels(avg)
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
        pn.Row(w_log_scale, w_mask_hot, w_cmap_vis, w_vmin_vis, w_vmax_vis, align="end"),
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
#   state           — dict: rois, scans, exp, current_frame, roi_history
#   state['rois']   — dict of name → ROI objects
#   state['scans']  — dict of label → Scan objects
#   np, plt, pn     — numpy, matplotlib.pyplot, panel
#   roi_ops         — ptyxplore.roi_ops module
#   data_proc       — ptyxplore.data_proc module
#   data_io         — ptyxplore.data_io module
#   cbd_loader      — ptyxplore.cbd_loader module (Exp, Scan, ROI classes)
#
# Example:
#   roi = state['rois']['bragg_001']
#   print(roi.data_4d.shape)
#   cbd_loader.ROI?
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
    try:
        from ptyxplore import roi_ops, data_proc, data_io
        import ptyxplore.cbd_loader as cbd_loader
    except Exception:
        roi_ops = data_proc = data_io = cbd_loader = None
    _console_ns.update({
        "state": state, "np": np, "plt": plt, "pn": pn,
        "roi_ops": roi_ops, "data_proc": data_proc,
        "data_io": data_io, "cbd_loader": cbd_loader,
        "h5py": h5py, "os": os,
    })
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
#  Per-scan tab factory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scan_tab(scan_label: str) -> pn.Column:
    """Return a tab with scan info, Bokeh Frame Browser, and ROI creation from selection."""
    scan = state["scans"][scan_label]
    import pandas as pd
    from bokeh.models import (
        ColumnDataSource, LinearColorMapper, ColorBar,
        HoverTool, CustomJS, BasicTicker,
    )
    from bokeh.plotting import figure as bk_figure

    # ── Scan info ─────────────────────────────────────────────────────────────
    info_html = pn.pane.HTML(
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;line-height:1.9'>"
        f"<span style='color:var(--txt-muted)'>data_dir  :</span> <span style='color:var(--txt)'>{scan.data_dir}</span><br>"
        f"<span style='color:var(--txt-muted)'>slow_axis :</span> <span style='color:{ACCENT2}'>{scan.slow_axis}</span><br>"
        f"<span style='color:var(--txt-muted)'>wavelength:</span> <span style='color:{ACCENT}'>{scan.wavelength*10:.4f} Å</span>"
        f"&nbsp;&nbsp;<span style='color:var(--txt-muted)'>energy:</span> <span style='color:{ACCENT}'>{scan.energy:.4f} keV</span><br>"
        f"<span style='color:var(--txt-muted)'>det psize :</span> <span style='color:var(--txt)'>{scan.det_psize*1e-3:.0f} µm</span>"
        f"</div>",
        sizing_mode="stretch_width",
    )
    stats_pane = pn.pane.HTML(_make_stats_html(scan), sizing_mode="stretch_width")
    ftable = pn.widgets.Tabulator(
        value=pd.DataFrame([{"#": i+1, "File": f} for i, f in enumerate(scan.file_list)]),
        pagination="local", page_size=10,
        sizing_mode="stretch_width", height=240, theme="midnight",
        stylesheets=[""":host .tabulator { background: var(--bg-dark); color: var(--txt);
           font-family: 'JetBrains Mono', monospace; font-size: 12px; }"""],
    )

    # ── Bokeh Frame Browser ───────────────────────────────────────────────────
    n_files  = len(scan.file_list)
    s_file   = pn.widgets.IntSlider(name="File index",  start=0, end=max(0, n_files-1), value=0, width=280)
    s_frame  = pn.widgets.IntSlider(name="Frame index", start=0, end=0, value=0, width=280)
    cb_hot   = pn.widgets.Checkbox(name="Mask hot pixels", value=True)
    cb_log   = pn.widgets.Checkbox(name="Log scale", value=False)
    sel_cmap = pn.widgets.Select(
        name="Colormap", options=["viridis","jet","plasma","inferno","gray","hot","turbo"],
        value="viridis", width=130)
    fi_vmin    = pn.widgets.FloatInput(name="vmin (blank=auto)", width=140)
    fi_vmax    = pn.widgets.FloatInput(name="vmax (blank=auto)", width=140)
    btn_auto   = pn.widgets.Button(name="Autoscale", button_type="default", width=100)
    btn_load   = pn.widgets.Button(name="Load Frame",   button_type="primary", width=140)
    btn_avg    = pn.widgets.Button(name="Show Average", button_type="default",  width=140)
    fb_info    = pn.pane.HTML("", sizing_mode="stretch_width")
    coord_info = pn.pane.HTML("", sizing_mode="stretch_width")

    # Persistent Bokeh figure + data source
    _blank   = np.zeros((64, 64), dtype=np.float32)
    _bk_src  = ColumnDataSource({'image': [_blank], 'x': [0], 'y': [0], 'dw': [64], 'dh': [64]})
    _bk_map  = LinearColorMapper(palette=_bk_palette("viridis"), low=0, high=1, nan_color='black')
    _bk_fig  = bk_figure(
        sizing_mode="stretch_width", height=480,
        tools="pan,wheel_zoom,box_select,reset,tap",
        active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    _bk_fig.y_range.flipped = True
    _bk_fig.image('image', 'x', 'y', 'dw', 'dh', source=_bk_src, color_mapper=_bk_map)
    _bk_fig.add_tools(HoverTool(tooltips=[("col", "$x{0}"), ("row", "$y{0}"), ("val", "@image")]))
    _bk_fig.add_layout(ColorBar(color_mapper=_bk_map, ticker=BasicTicker(), location=(0, 0)), 'right')

    # JS → Python tracking for tap (cursor) and box-select (ROI)
    _cur_src = ColumnDataSource({'x': [0.0], 'y': [0.0]})
    _sel_src = ColumnDataSource({'x0': [0.0], 'y0': [0.0], 'x1': [0.0], 'y1': [0.0]})
    _bk_fig.js_on_event('tap', CustomJS(args={'s': _cur_src}, code="""
        s.data = {x: [cb_obj.x], y: [cb_obj.y]};  s.change.emit();
    """))
    _bk_fig.js_on_event('selectiongeometry', CustomJS(args={'s': _sel_src}, code="""
        if (cb_obj.geometry && cb_obj.geometry.type === 'rect') {
            const g = cb_obj.geometry;
            s.data = {x0: [Math.min(g.x0,g.x1)], y0: [Math.min(g.y0,g.y1)],
                      x1: [Math.max(g.x0,g.x1)], y1: [Math.max(g.y0,g.y1)]};
            s.change.emit();
        }
    """))

    _cursor = {'row': 0, 'col': 0}
    _sel    = {'rs': 0, 're': 0, 'cs': 0, 'ce': 0}

    def _fmt_coord():
        coord_info.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;color:var(--txt-muted)'>"
            f"cursor → row={_cursor['row']}, col={_cursor['col']}"
            f" &nbsp;|&nbsp; box → row=[{_sel['rs']}, {_sel['re']}]"
            f" col=[{_sel['cs']}, {_sel['ce']}]</span>")

    def _on_cur(attr, old, new):
        _cursor['col'] = max(0, int(new['x'][0]))
        _cursor['row'] = max(0, int(new['y'][0]))
        pn.state.execute(_fmt_coord)

    def _on_sel(attr, old, new):
        _sel['cs'] = max(0, int(new['x0'][0]))
        _sel['ce'] = max(0, int(new['x1'][0]))
        _sel['rs'] = max(0, int(new['y0'][0]))
        _sel['re'] = max(0, int(new['y1'][0]))
        pn.state.execute(_fmt_coord)

    _cur_src.on_change('data', _on_cur)
    _sel_src.on_change('data', _on_sel)

    def _update_bk(frame_data):
        h, w   = frame_data.shape
        disp   = np.log1p(np.abs(frame_data.astype(float))) if cb_log.value else frame_data.astype(float)
        lo     = fi_vmin.value if fi_vmin.value is not None else float(np.nanmin(disp))
        hi     = fi_vmax.value if fi_vmax.value is not None else float(np.nanmax(disp))
        if abs(hi - lo) < 1e-10:
            hi = lo + 1.0
        _bk_map.palette = _bk_palette(sel_cmap.value)
        _bk_map.low  = lo
        _bk_map.high = hi
        _bk_src.data = {'image': [disp], 'x': [0], 'y': [0], 'dw': [w], 'dh': [h]}

    def _on_autoscale(event):
        if len(_bk_src.data['image']) and _bk_src.data['image'][0] is not None:
            d = _bk_src.data['image'][0]
            fi_vmin.value = fi_vmax.value = None
            _bk_map.low  = float(d.min())
            _bk_map.high = float(d.max())

    def _on_vscale(event=None):
        if len(_bk_src.data['image']) and _bk_src.data['image'][0] is not None:
            d = _bk_src.data['image'][0]
            _bk_map.low  = fi_vmin.value if fi_vmin.value is not None else float(d.min())
            _bk_map.high = fi_vmax.value if fi_vmax.value is not None else float(d.max())

    btn_auto.on_click(_on_autoscale)
    fi_vmin.param.watch(_on_vscale, 'value')
    fi_vmax.param.watch(_on_vscale, 'value')

    def _load(event):
        fidx  = s_file.value
        fridx = s_frame.value
        fpath = scan.file_list[fidx]
        log(f"[{scan_label}] Loading file {fidx+1}/{n_files}, frame {fridx}…", "info")
        try:
            with h5py.File(fpath, "r") as f:
                dset  = f["/entry/data/data"]
                nfrm  = dset.shape[0]
                frame = dset[min(fridx, nfrm - 1), :, :]
            s_frame.end = max(0, nfrm - 1)
            if cb_hot.value:
                from ptyxplore.data_proc import mask_hot_pixels
                frame = mask_hot_pixels(frame)
            _update_bk(frame)
            fb_info.object = (
                f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--txt-muted);margin-top:6px'>"
                f"shape: {frame.shape} &nbsp;|&nbsp; dtype: {frame.dtype} &nbsp;|&nbsp; "
                f"min: {frame.min():.0f} &nbsp;|&nbsp; max: {frame.max():.0f} &nbsp;|&nbsp; "
                f"mean: {frame.mean():.1f}</div>")
            log(f"[{scan_label}] Frame loaded ✓", "ok")
        except Exception as exc:
            log(f"[{scan_label}] Failed to load frame: {exc}", "err")
            pn.state.notifications.error(str(exc), duration=5000)

    def _avg(event):
        fidx  = s_file.value
        fpath = scan.file_list[fidx]
        log(f"[{scan_label}] Computing average of file {fidx+1}…", "info")
        try:
            with h5py.File(fpath, "r") as f:
                avg = f["/entry/data/data"][...].mean(axis=0)
            if cb_hot.value:
                from ptyxplore.data_proc import mask_hot_pixels
                avg = mask_hot_pixels(avg)
            _update_bk(avg)
            fb_info.object = (
                f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--txt-muted);margin-top:6px'>"
                f"shape: {avg.shape} &nbsp;|&nbsp; min: {avg.min():.1f} &nbsp;|&nbsp; "
                f"max: {avg.max():.1f} &nbsp;|&nbsp; mean: {avg.mean():.2f}</div>")
            log(f"[{scan_label}] Average computed ✓", "ok")
        except Exception as exc:
            log(f"[{scan_label}] Failed to compute average: {exc}", "err")
            pn.state.notifications.error(str(exc), duration=5000)

    btn_load.on_click(_load)
    btn_avg.on_click(_avg)

    # Log-point and log-ROI buttons ('p' / 'r' equivalents)
    btn_log_pt  = pn.widgets.Button(name="Log Point [p]", button_type="default", width=120)
    btn_log_roi = pn.widgets.Button(name="Log ROI [r]",   button_type="default", width=110)

    def _log_pt(event):
        log(f"[{scan_label}] Point → row={_cursor['row']}, col={_cursor['col']}", "info")

    def _log_roi(event):
        log(f"[{scan_label}] Selection → "
            f"row=[{_sel['rs']}, {_sel['re']}], col=[{_sel['cs']}, {_sel['ce']}]", "info")

    btn_log_pt.on_click(_log_pt)
    btn_log_roi.on_click(_log_roi)

    # ── Create ROI from box-selection ─────────────────────────────────────────
    w_new_name = pn.widgets.TextInput(name="ROI name", placeholder="e.g. bragg_001", width=180)
    w_new_kind = pn.widgets.Select(name="Kind", options=["Bragg", "Pupil"], value="Bragg", width=100)
    btn_mk_roi = pn.widgets.Button(
        name="Create ROI from Selection", button_type="success", width=220)

    def _on_mk_roi(event):
        name = w_new_name.value.strip()
        kind = w_new_kind.value.lower()
        coords = [_sel['rs'], _sel['re'], _sel['cs'], _sel['ce']]
        if not name:
            log("Enter a name for the new ROI.", "warn")
            pn.state.notifications.warning("Enter ROI name.", duration=3000)
            return
        if name in state["rois"]:
            log(f"ROI '{name}' already exists.", "err")
            pn.state.notifications.error(f"Name '{name}' already in use.", duration=3000)
            return
        try:
            from ptyxplore.cbd_loader import ROI
            roi = ROI(scan=scan, kind=kind, coords=coords, centre_pixel=[0.0, 0.0])
            state["rois"][name] = roi
            state["roi_history"][name] = [
                f"[{ts()}] Created from frame selection ({kind}, coords={coords})"]
            if kind == "pupil" and state["pupil_name"] is None:
                state["pupil_name"] = name
            _rebuild_roi_cards()
            _refresh_pipe_roi_select()
            tabs.append((f"{scan_label}: {name}", _make_roi_tab(name)))
            log(f"ROI '{name}' created from selection ✓  coords={coords}", "ok")
            pn.state.notifications.success(f"ROI '{name}' added!", duration=2000)
        except Exception as exc:
            log(f"Failed to create ROI: {exc}", "err")
            pn.state.notifications.error(str(exc), duration=5000)

    btn_mk_roi.on_click(_on_mk_roi)

    # ── Remove scan button ────────────────────────────────────────────────────
    btn_remove = pn.widgets.Button(name="Remove Scan", button_type="danger", width=140)

    def _remove(event):
        if scan_label in state["scans"]:
            del state["scans"][scan_label]
            _refresh_scan_select()
            _rebuild_loaded_scans()
            scan_num = getattr(scan, "scan_num", scan_label)
            tabs.objects = [
                (n, obj) for n, obj in tabs.objects if n != f"Scan {scan_num}"
            ]
            log(f"Scan '{scan_label}' removed.", "warn")

    btn_remove.on_click(_remove)

    return pn.Column(
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Scan Info</p>", sizing_mode="stretch_width"),
            info_html, stats_pane,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            ftable,
            pn.Row(btn_remove),
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Frame Browser</p>", sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Scroll to zoom · Pan · Box-select to mark ROI · "
                "Tap then Log Point for coords</span>"),
            pn.Row(s_file, s_frame),
            pn.Row(cb_log, cb_hot, sel_cmap, align="end"),
            pn.Row(fi_vmin, fi_vmax, btn_auto, align="end"),
            pn.Row(btn_load, btn_avg),
            _bk_fig,
            pn.Row(btn_log_pt, btn_log_roi, coord_info, align="center"),
            fb_info,
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML(
                "<p class='section-label'>Create ROI from Selection</p>",
                sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Draw a box in the frame browser, enter a name, "
                "then click Create.</span>"),
            pn.Row(w_new_name, w_new_kind, align="end"),
            pn.Row(btn_mk_roi),
            sizing_mode="stretch_width", css_classes=["ptycho-card"],
        ),
        sizing_mode="stretch_width", margin=(16, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-ROI tab factory
# ═══════════════════════════════════════════════════════════════════════════════

def _make_roi_tab(roi_name: str) -> pn.Tabs:
    """Return a pn.Tabs with Pipeline + Visualisation sub-tabs scoped to roi_name."""

    # ── Fresh param widget instances (independent per ROI) ────────────────────
    pw = {
        "num_jobs":    pn.widgets.IntInput(  name="num_jobs",              value=32,  step=8,   width=130),
        "kernel_size": pn.widgets.IntInput(  name="kernel_size",           value=2,   step=1,   width=130),
        "stride":      pn.widgets.IntInput(  name="stride",                value=2,   step=1,   width=110),
        "padding":     pn.widgets.IntInput(  name="padding",               value=0,             width=110),
        "mask_val":    pn.widgets.FloatInput(name="mask_val",              value=1.0, step=0.1, width=130),
        "region_rs":   pn.widgets.IntInput(  name="region row start",      value=0,   width=130),
        "region_re":   pn.widgets.IntInput(  name="region row end",        value=10,  width=130),
        "region_cs":   pn.widgets.IntInput(  name="region col start",      value=0,   width=130),
        "region_ce":   pn.widgets.IntInput(  name="region col end",        value=10,  width=130),
        "region_mode": pn.widgets.Select(    name="mode", options=["zeros","median","ones"], value="median", width=120),
        "est_ttheta":  pn.widgets.FloatInput(name="est_ttheta (deg)",      value=1.0, step=0.1, width=160),
        "method":      pn.widgets.Select(    name="method", options=["BFGS","L-BFGS-B","Nelder-Mead"], value="BFGS", width=150),
        "gtol":        pn.widgets.FloatInput(name="gtol",                  value=1e-6, width=130),
        "alpha":       pn.widgets.FloatInput(name="alpha shift",           value=0.0, step=1e-4, width=130),
        "beta":        pn.widgets.FloatInput(name="beta shift",            value=0.0, step=1e-4, width=130),
        "gamma":       pn.widgets.FloatInput(name="gamma shift",           value=0.0, step=1e-4, width=130),
        "var_thresh":  pn.widgets.FloatInput(name="variance threshold",    value=0.1, step=0.01, width=180),
        "sigma":       pn.widgets.FloatInput(name="sigma",                 value=2.0, step=0.5,  width=120),
        "med_thresh":  pn.widgets.FloatInput(name="threshold",             value=3.0, step=0.5,  width=120),
        "det_thresh":  pn.widgets.FloatInput(name="detection threshold",   value=0.1, step=0.01, width=180),
        "det_min":     pn.widgets.FloatInput(name="min object sum",        value=10.0, step=1.0, width=160),
        "det_max":     pn.widgets.FloatInput(name="max object sum",        value=1e6, step=100., width=160),
        "coh_thresh":  pn.widgets.FloatInput(name="threshold value",       value=0.0, step=0.1,  width=160),
        "cmap":        pn.widgets.Select(    name="colormap", options=["viridis","jet","plasma","inferno","gray","hot"], value="viridis", width=140),
        "log_scale":   pn.widgets.Checkbox(  name="log scale", value=False),
        "vmin":        pn.widgets.FloatInput(name="vmin (blank=auto)", width=130),
        "vmax":        pn.widgets.FloatInput(name="vmax (blank=auto)", width=130),
        "4d_px":       pn.widgets.IntInput(  name="detector row (px)", value=0, width=150),
        "4d_py":       pn.widgets.IntInput(  name="detector col (py)", value=0, width=150),
        "save_dir":    pn.widgets.TextInput( name="output directory", placeholder="/path/to/out/", sizing_mode="stretch_width"),
        "save_name":   pn.widgets.TextInput( name="file name (no ext)", placeholder="export", width=200),
    }

    # Only ROI-level operations — the target ROI is implicit
    roi_op_list = [op for op, cfg in _ALL_OPS.items() if cfg["target"] == "roi"]
    w_op       = pn.widgets.Select(name="Operation", options=roi_op_list, value=roi_op_list[0], width=300)
    params_col = pn.Column(sizing_mode="stretch_width")
    queue      = []
    queue_col_r = pn.Column(sizing_mode="stretch_width")

    # ── Snapshot params from local pw dict ────────────────────────────────────
    def snapshot(op):
        p = {}
        if op == "Make 4D dataset":
            p["num_jobs"] = pw["num_jobs"].value
        elif op == "Pool detector":
            p.update(kernel_size=pw["kernel_size"].value, stride=pw["stride"].value, padding=pw["padding"].value)
        elif op in ("Mask region detector", "Mask region cohimgs"):
            p.update(region=[pw["region_rs"].value, pw["region_re"].value,
                             pw["region_cs"].value, pw["region_ce"].value],
                     mode=pw["region_mode"].value)
        elif op == "Make k-outs":
            p["mask_val"] = pw["mask_val"].value
        elif op == "Compute kins":
            p.update(est_ttheta=pw["est_ttheta"].value, method=pw["method"].value, gtol=pw["gtol"].value)
        elif op == "Compute kins w geometry":
            p.update(method=pw["method"].value, gtol=pw["gtol"].value)
        elif op == "Refine kins":
            p.update(alpha=pw["alpha"].value, beta=pw["beta"].value, gamma=pw["gamma"].value)
        elif op == "Filter coherent images":
            p["var_thresh"] = pw["var_thresh"].value
        elif op == "Remove background":
            p["sigma"] = pw["sigma"].value
        elif op == "Median filter (coh)":
            p.update(kernel_size=pw["kernel_size"].value, stride=pw["stride"].value, threshold=pw["med_thresh"].value)
        elif op == "Detect objects":
            p.update(det_thresh=pw["det_thresh"].value, det_min=pw["det_min"].value, det_max=pw["det_max"].value)
        elif op == "Mask cohimgs threshold":
            p["coh_thresh"] = pw["coh_thresh"].value
        elif op.startswith("Plot:"):
            p.update(cmap=pw["cmap"].value, log_scale=pw["log_scale"].value,
                     vmin=pw["vmin"].value or None, vmax=pw["vmax"].value or None)
            if op == "Plot: 4D viewer (single)":
                p.update(px=pw["4d_px"].value, py=pw["4d_py"].value)
        elif op.startswith("Save:"):
            p.update(save_dir=pw["save_dir"].value.strip(),
                     save_name=pw["save_name"].value.strip() or "ptyxplore_export")
        return p

    # ── Refresh param panel when operation changes ────────────────────────────
    def refresh_params(event=None):
        op = w_op.value
        params_col.clear()
        if op == "Make 4D dataset":
            params_col.append(pw["num_jobs"])
        elif op == "Pool detector":
            params_col.append(pn.Row(pw["kernel_size"], pw["stride"], pw["padding"]))
        elif op in ("Mask region detector", "Mask region cohimgs"):
            params_col.append(pn.Row(pw["region_rs"], pw["region_re"], pw["region_cs"], pw["region_ce"]))
            params_col.append(pw["region_mode"])
        elif op == "Make k-outs":
            params_col.append(pw["mask_val"])
        elif op == "Compute kins":
            params_col.append(pn.Row(pw["est_ttheta"], pw["method"], pw["gtol"]))
        elif op == "Compute kins w geometry":
            params_col.append(pn.Row(pw["method"], pw["gtol"]))
        elif op == "Refine kins":
            params_col.append(pn.Row(pw["alpha"], pw["beta"], pw["gamma"]))
        elif op == "Filter coherent images":
            params_col.append(pw["var_thresh"])
        elif op == "Remove background":
            params_col.append(pw["sigma"])
        elif op == "Median filter (coh)":
            params_col.append(pn.Row(pw["kernel_size"], pw["stride"], pw["med_thresh"]))
        elif op == "Detect objects":
            params_col.append(pn.Row(pw["det_thresh"], pw["det_min"], pw["det_max"]))
        elif op == "Mask cohimgs threshold":
            params_col.append(pw["coh_thresh"])
        elif op.startswith("Plot:"):
            params_col.append(pn.Row(pw["cmap"], pw["log_scale"], pw["vmin"], pw["vmax"]))
            if op == "Plot: 4D viewer (single)":
                params_col.append(pn.Row(pw["4d_px"], pw["4d_py"]))
        elif op.startswith("Save:"):
            params_col.append(pw["save_dir"])
            params_col.append(pw["save_name"])
        elif op == "Estimate 2θ":
            params_col.append(pn.pane.HTML(
                "<span style='color:var(--txt-muted);font-size:12px;font-family:JetBrains Mono,monospace'>"
                "Requires a designated Pupil ROI with k-outs computed.</span>"))

    w_op.param.watch(refresh_params, "value")
    refresh_params()

    # ── Per-ROI queue buttons and callbacks ───────────────────────────────────
    btn_add  = pn.widgets.Button(name="Add to Queue", button_type="primary", width=160)
    btn_run  = pn.widgets.Button(name="▶  Run All",   button_type="primary", width=150)
    btn_stop = pn.widgets.Button(name="⏹  Stop",      button_type="danger",  width=120, disabled=True)
    btn_clear = pn.widgets.Button(name="Clear Queue", button_type="default", width=130)

    def on_add(event):
        op = w_op.value
        if _ALL_OPS.get(op, {}).get("needs_pupil") and state["pupil_name"] is None:
            log("This operation requires a designated Pupil ROI.", "warn")
            pn.state.notifications.warning("No Pupil ROI set.", duration=3000)
            return
        queue.append({"op": op, "roi_name": roi_name,
                      "params": snapshot(op), "status": "pending", "enabled": True})
        _rebuild_queue(queue, queue_col_r)
        log(f"[{roi_name}] Queued: {op}", "info")

    def on_run(event):
        if not btn_run.disabled:
            _run_queue(queue, queue_col_r, roi_name, btn_run, btn_stop, btn_clear)

    def on_stop(event):
        _stop_event.set()
        log(f"[{roi_name}] Stop requested — will halt after the current step finishes.", "warn")
        btn_stop.disabled = True

    def on_clear(event):
        queue.clear()
        _rebuild_queue(queue, queue_col_r)
        log(f"[{roi_name}] Queue cleared.", "info")

    btn_add.on_click(on_add)
    btn_run.on_click(on_run)
    btn_stop.on_click(on_stop)
    btn_clear.on_click(on_clear)
    _rebuild_queue(queue, queue_col_r)

    # ── Pipeline sub-tab layout ───────────────────────────────────────────────
    subtab_pipe = pn.Column(
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Add Step</p>", sizing_mode="stretch_width"),
            w_op, params_col,
            pn.Row(btn_add),
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Queue</p>", sizing_mode="stretch_width"),
            queue_col_r,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.Row(btn_run, btn_stop, btn_clear),
            sizing_mode="stretch_width", css_classes=["ptycho-card"],
        ),
        sizing_mode="stretch_width", margin=(16, 0),
    )

    # ── 4D Dataset Viewer — Bokeh (no ROI selector — this tab owns one ROI) ──
    from bokeh.models import (
        ColumnDataSource, LinearColorMapper, ColorBar,
        HoverTool, BasicTicker,
    )
    from bokeh.plotting import figure as bk_figure

    w_prow     = pn.widgets.IntSlider(name="det row",  start=0, end=1, value=0, width=280)
    w_pcol     = pn.widgets.IntSlider(name="det col",  start=0, end=1, value=0, width=280)
    w_lrow     = pn.widgets.IntSlider(name="scan row", start=0, end=1, value=0, width=280)
    w_lcol     = pn.widgets.IntSlider(name="scan col", start=0, end=1, value=0, width=280)
    w_cmap_coh = pn.widgets.Select(
        name="coh cmap", options=["plasma","viridis","jet","gray","hot","turbo"], value="plasma", width=120)
    w_cmap_det = pn.widgets.Select(
        name="det cmap", options=["viridis","plasma","jet","gray","hot","turbo"], value="viridis", width=120)
    fi_coh_vmin = pn.widgets.FloatInput(name="coh vmin", width=100)
    fi_coh_vmax = pn.widgets.FloatInput(name="coh vmax", width=100)
    fi_det_vmin = pn.widgets.FloatInput(name="det vmin", width=100)
    fi_det_vmax = pn.widgets.FloatInput(name="det vmax", width=100)
    btn_coh_auto = pn.widgets.Button(name="Auto", button_type="default", width=60)
    btn_det_auto = pn.widgets.Button(name="Auto", button_type="default", width=60)
    fdmsg        = pn.pane.HTML("", sizing_mode="stretch_width")

    _blank4 = np.zeros((64, 64), dtype=np.float32)

    # Coherent Bokeh figure
    _coh_src = ColumnDataSource({'image': [_blank4.copy()], 'x': [0], 'y': [0], 'dw': [64], 'dh': [64]})
    _coh_map = LinearColorMapper(palette=_bk_palette("plasma"), low=0, high=1, nan_color='black')
    _coh_mark = ColumnDataSource({'x': [32.0], 'y': [32.0], 'w': [2.0], 'h': [2.0]})
    _coh_fig = bk_figure(
        title="Coherent", sizing_mode="stretch_width", height=380,
        tools="pan,wheel_zoom,reset,tap", active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    _coh_fig.y_range.flipped = True
    _coh_fig.image('image', 'x', 'y', 'dw', 'dh', source=_coh_src, color_mapper=_coh_map)
    _coh_fig.rect('x', 'y', 'w', 'h', source=_coh_mark,
                  fill_color=None, line_color='white', line_width=1.5)
    _coh_fig.add_tools(HoverTool(tooltips=[("col", "$x{0}"), ("row", "$y{0}")]))
    _coh_fig.add_layout(ColorBar(color_mapper=_coh_map, ticker=BasicTicker(), location=(0, 0)), 'right')

    # Detector Bokeh figure
    _det_src = ColumnDataSource({'image': [_blank4.copy()], 'x': [0], 'y': [0], 'dw': [64], 'dh': [64]})
    _det_map = LinearColorMapper(palette=_bk_palette("viridis"), low=0, high=1, nan_color='black')
    _det_mark = ColumnDataSource({'x': [32.0], 'y': [32.0], 'w': [4.0], 'h': [4.0]})
    _det_fig = bk_figure(
        title="Detector", sizing_mode="stretch_width", height=380,
        tools="pan,wheel_zoom,reset,tap", active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    _det_fig.y_range.flipped = True
    _det_fig.image('image', 'x', 'y', 'dw', 'dh', source=_det_src, color_mapper=_det_map)
    _det_fig.rect('x', 'y', 'w', 'h', source=_det_mark,
                  fill_color=None, line_color='white', line_width=1.5)
    _det_fig.add_tools(HoverTool(tooltips=[("col", "$x{0}"), ("row", "$y{0}")]))
    _det_fig.add_layout(ColorBar(color_mapper=_det_map, ticker=BasicTicker(), location=(0, 0)), 'right')

    def _apply_coh_scale(event=None):
        d = _coh_src.data['image'][0]
        _coh_map.low  = fi_coh_vmin.value if fi_coh_vmin.value is not None else float(np.nanmin(d))
        _coh_map.high = fi_coh_vmax.value if fi_coh_vmax.value is not None else float(np.nanmax(d))

    def _apply_det_scale(event=None):
        d = _det_src.data['image'][0]
        _det_map.low  = fi_det_vmin.value if fi_det_vmin.value is not None else float(np.nanmin(d))
        _det_map.high = fi_det_vmax.value if fi_det_vmax.value is not None else float(np.nanmax(d))

    def _coh_autoscale(event=None):
        fi_coh_vmin.value = fi_coh_vmax.value = None
        d = _coh_src.data['image'][0]
        _coh_map.low = float(np.nanmin(d));  _coh_map.high = float(np.nanmax(d))

    def _det_autoscale(event=None):
        fi_det_vmin.value = fi_det_vmax.value = None
        d = _det_src.data['image'][0]
        _det_map.low = float(np.nanmin(d));  _det_map.high = float(np.nanmax(d))

    fi_coh_vmin.param.watch(_apply_coh_scale, 'value')
    fi_coh_vmax.param.watch(_apply_coh_scale, 'value')
    fi_det_vmin.param.watch(_apply_det_scale, 'value')
    fi_det_vmax.param.watch(_apply_det_scale, 'value')
    btn_coh_auto.on_click(_coh_autoscale)
    btn_det_auto.on_click(_det_autoscale)

    def update_viewer(*_):
        roi = state["rois"].get(roi_name)
        if roi is None or roi.data_4d is None:
            fdmsg.object = ("<span style='color:var(--txt-muted)'>"
                            "Run <b>Make 4D dataset</b> on this ROI first.</span>")
            return
        fdmsg.object = ""
        from ptyxplore.data_proc import make_coherent_image, make_detector_image
        prow, pcol = w_prow.value, w_pcol.value
        lrow, lcol = w_lrow.value, w_lcol.value
        coh = make_coherent_image(roi.data_4d, np.array([prow, pcol])).astype(float)
        det = make_detector_image(roi.data_4d, np.array([lrow, lcol])).astype(float)
        hc, wc = coh.shape;  hd, wd = det.shape

        # Update coh source + colormap
        _coh_map.palette = _bk_palette(w_cmap_coh.value)
        lo_c = fi_coh_vmin.value if fi_coh_vmin.value is not None else float(np.nanmin(coh))
        hi_c = fi_coh_vmax.value if fi_coh_vmax.value is not None else float(np.nanmax(coh))
        if abs(hi_c - lo_c) < 1e-10: hi_c = lo_c + 1.0
        _coh_map.low = lo_c;  _coh_map.high = hi_c
        _coh_src.data = {'image': [coh], 'x': [0], 'y': [0], 'dw': [wc], 'dh': [hc]}
        _coh_mark.data = {'x': [lcol + 0.5], 'y': [lrow + 0.5], 'w': [2.0], 'h': [2.0]}
        _coh_fig.title.text = f"Coherent  det=({pcol}, {prow})"

        # Update det source + colormap
        _det_map.palette = _bk_palette(w_cmap_det.value)
        lo_d = fi_det_vmin.value if fi_det_vmin.value is not None else float(np.nanmin(det))
        hi_d = fi_det_vmax.value if fi_det_vmax.value is not None else float(np.nanmax(det))
        if abs(hi_d - lo_d) < 1e-10: hi_d = lo_d + 1.0
        _det_map.low = lo_d;  _det_map.high = hi_d
        _det_src.data = {'image': [det], 'x': [0], 'y': [0], 'dw': [wd], 'dh': [hd]}
        _det_mark.data = {'x': [pcol + 0.5], 'y': [prow + 0.5], 'w': [4.0], 'h': [4.0]}
        _det_fig.title.text = f"Detector  scan=({lcol}, {lrow})"

    def init_viewer():
        roi = state["rois"].get(roi_name)
        if roi and roi.data_4d is not None:
            dsh, csh = roi.data_4d.shape[2:], roi.data_4d.shape[:2]
            def _init():
                w_prow.end = dsh[0] - 1;  w_prow.value = dsh[0] // 2
                w_pcol.end = dsh[1] - 1;  w_pcol.value = dsh[1] // 2
                w_lrow.end = csh[0] - 1;  w_lrow.value = csh[0] // 2
                w_lcol.end = csh[1] - 1;  w_lcol.value = csh[1] // 2
                update_viewer()
            pn.state.execute(_init)
        else:
            pn.state.execute(update_viewer)

    roi_4dv_hooks[roi_name] = init_viewer  # called by _execute_step after Make 4D dataset

    for _w in (w_prow, w_pcol, w_lrow, w_lcol, w_cmap_coh, w_cmap_det):
        _w.param.watch(update_viewer, "value")
    init_viewer()

    # ── Per-ROI gallery ───────────────────────────────────────────────────────
    per_gallery     = []
    per_gallery_col = pn.Column(sizing_mode="stretch_width")
    btn_gal_clear   = pn.widgets.Button(name="Clear", button_type="default", width=100)

    def rebuild_gallery():
        per_gallery_col.clear()
        if not per_gallery:
            per_gallery_col.append(pn.pane.HTML(
                "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
                "No plots yet — run pipeline Plot steps.</span>"))
            return
        for i, gitem in enumerate(reversed(per_gallery)):
            actual_idx = len(per_gallery) - 1 - i
            def _make_grm(idx=actual_idx):
                btn = pn.widgets.Button(name="✕", button_type="default", width=40)
                def _rm(event):
                    if idx < len(per_gallery):
                        removed = per_gallery.pop(idx)
                        try: plt.close(removed["fig"])
                        except Exception: pass
                        rebuild_gallery()
                btn.on_click(_rm)
                return btn
            per_gallery_col.append(pn.Column(
                pn.Row(
                    pn.pane.HTML(
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:var(--accent)'>{len(per_gallery)-i}. {gitem['title']}</span>"),
                    _make_grm(), align="center"),
                gitem["pane"],
                pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
            ))

    def add_to_per_gallery(title, fig_or_panel):
        """Add a matplotlib figure or a Panel component to the per-ROI gallery."""
        if isinstance(fig_or_panel, pn.viewable.Viewable):
            pane = fig_or_panel
        else:
            pane = pn.pane.Matplotlib(
                fig_or_panel, tight=True, sizing_mode="stretch_width", height=460)
            plt.close(fig_or_panel)
        per_gallery.append({"title": title, "pane": pane, "fig": None})
        rebuild_gallery()

    def on_gal_clear(event):
        for gitem in per_gallery:
            try: plt.close(gitem["fig"])
            except Exception: pass
        per_gallery.clear()
        rebuild_gallery()

    roi_gallery_hooks[roi_name] = add_to_per_gallery  # called by _add_to_gallery
    btn_gal_clear.on_click(on_gal_clear)
    rebuild_gallery()

    # ── Visualisation sub-tab layout ──────────────────────────────────────────
    subtab_vis = pn.Column(
        pn.Column(
            pn.pane.HTML("<p class='section-label'>4D Dataset Viewer</p>", sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Scroll to zoom · White box = cross-cursor position</span>"),
            pn.Row(
                pn.Column(
                    pn.pane.HTML("<span style='font-size:11px;color:var(--accent2);font-family:JetBrains Mono,monospace'>Detector pixel (coherent view)</span>"),
                    pn.Row(w_prow, w_pcol),
                    pn.Row(w_cmap_coh, fi_coh_vmin, fi_coh_vmax, btn_coh_auto, align="end"),
                    sizing_mode="stretch_width",
                ),
                pn.Column(
                    pn.pane.HTML("<span style='font-size:11px;color:var(--accent2);font-family:JetBrains Mono,monospace'>Scan position (detector view)</span>"),
                    pn.Row(w_lrow, w_lcol),
                    pn.Row(w_cmap_det, fi_det_vmin, fi_det_vmax, btn_det_auto, align="end"),
                    sizing_mode="stretch_width",
                ),
            ),
            fdmsg,
            pn.Row(_coh_fig, _det_fig, sizing_mode="stretch_width"),
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Pipeline Outputs</p>", sizing_mode="stretch_width"),
            pn.Row(btn_gal_clear),
            per_gallery_col,
            sizing_mode="stretch_width", css_classes=["ptycho-card"],
        ),
        sizing_mode="stretch_width", margin=(16, 0),
    )

    return pn.Tabs(
        ("Pipeline",      subtab_pipe),
        ("Visualisation", subtab_vis),
        sizing_mode="stretch_width",
        dynamic=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs + main dashboard
# ═══════════════════════════════════════════════════════════════════════════════

tabs = pn.Tabs(
    ("Scan Loading",     tab_loading),
    ("ROI Manager", tab_roi_manager),
    ("Python",      tab_python),
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
