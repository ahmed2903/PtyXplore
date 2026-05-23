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
    "white_fields":    {},    # label → 2-D float32 ndarray (full-detector H×W)
}
_prev_figs: dict = {}
pipeline_queue: list[dict] = []
plot_gallery:   list[dict] = []
roi_gallery_hooks: dict = {}   # roi_name → callable(title, fig) for per-ROI gallery
roi_4dv_hooks:    dict = {}    # roi_name → callable() to refresh per-ROI 4D viewer
_roi_ref_selects:     list = []   # all per-ROI tab ref_roi Select widgets, kept in sync
_white_field_selects: list = []   # all white-field Select widgets (global + per-ROI), kept in sync
_stop_event = threading.Event()  # set this to request a halt between pipeline steps
mask_library:         dict = {}   # name → list[dict]  (shared across all scan tabs)
_mask_lib_selects:    list = []   # all mask-library Select widgets, kept in sync


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
                          vmin=None, vmax=None, height: int = 440,
                          x_axis_nm=None, y_axis_nm=None) -> pn.Column:
    """
    Return an interactive Panel component wrapping a Bokeh image figure.
    Features: scroll-to-zoom, pan, box-select for ROI, tap for point coords,
              vmin/vmax inputs, autoscale button, Log Point and Log ROI buttons.

    x_axis_nm / y_axis_nm : 1-D arrays of physical motor positions (nm).
        When provided the image is placed on nm axes and axis labels are set.
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

    # ── spatial coordinates ────────────────────────────────────────────────────
    _use_nm = x_axis_nm is not None and y_axis_nm is not None
    if _use_nm:
        img_x  = float(x_axis_nm[0])
        img_y  = float(y_axis_nm[0])
        img_dw = float(x_axis_nm[-1] - x_axis_nm[0])
        img_dh = float(y_axis_nm[-1] - y_axis_nm[0])
        hover_tips = [("Scan X (µm)", "$x{0.000}"), ("Scan Y (µm)", "$y{0.000}"), ("val", "@image")]
    else:
        img_x, img_y, img_dw, img_dh = 0, 0, w, h
        hover_tips = [("col", "$x{0}"), ("row", "$y{0}"), ("val", "@image")]

    bk_src = ColumnDataSource({'image': [disp], 'x': [img_x], 'y': [img_y], 'dw': [img_dw], 'dh': [img_dh]})
    bk_map = LinearColorMapper(palette=_bk_palette(cmap), low=lo, high=hi, nan_color='black')

    fig = bk_figure(
        title=title, sizing_mode="stretch_width", height=height,
        tools="pan,wheel_zoom,box_select,reset,tap", active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    fig.y_range.flipped = True
    fig.image('image', 'x', 'y', 'dw', 'dh', source=bk_src, color_mapper=bk_map)
    fig.add_tools(HoverTool(tooltips=hover_tips))
    fig.add_layout(ColorBar(color_mapper=bk_map, ticker=BasicTicker(), location=(0, 0)), 'right')
    if _use_nm:
        fig.xaxis.axis_label = "X (µm)"
        fig.yaxis.axis_label = "Y (µm)"

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
    btn_r = pn.widgets.Button(name="Remove",       button_type="danger",  width=100)

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
            # Remove the ROI tab (title format: "{scan_label}: {roi_name}")
            tab_suffix = f": {name}"
            for i in reversed([i for i, tname in enumerate(tabs._names) if tname.endswith(tab_suffix)]):
                tabs.pop(i)
            # Remove gallery plots associated with this ROI
            for i in reversed([i for i, item in enumerate(plot_gallery) if item.get("roi_name") == name]):
                removed = plot_gallery.pop(i)
                try:
                    plt.close(removed["fig"])
                except Exception:
                    pass
            _rebuild_gallery()
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
    "Normalise detector":       {"target": "roi", "needs_ref_roi": True},
    "Subtract white field":     {"target": "roi", "needs_white_field": True},
    # Scan-level
    "Sum detector frames":      {"target": "scan"},
    # Plot → Visualisation tab
    "Plot: Average detector":   {"target": "roi"},
    "Plot: Summed detector":    {"target": "roi"},
    "Plot: Averaged coherent":  {"target": "roi"},
    "Plot: Summed coherent":    {"target": "roi"},
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
pw_pos_axes    = pn.widgets.Checkbox(  name="position axes (µm)", value=False)
pw_vmin        = pn.widgets.FloatInput(name="vmin (blank=auto)", width=130)
pw_vmax        = pn.widgets.FloatInput(name="vmax (blank=auto)", width=130)
pw_4d_px       = pn.widgets.IntInput(  name="detector row (px)", value=0, width=150)
pw_4d_py       = pn.widgets.IntInput(  name="detector col (py)", value=0, width=150)
pw_save_dir    = pn.widgets.TextInput( name="output directory", placeholder="/path/to/out/", sizing_mode="stretch_width")
pw_save_name   = pn.widgets.TextInput( name="file name (no ext)", placeholder="export", width=200)
pw_ref_roi        = pn.widgets.Select(name="reference ROI",  options=["— none —"], value="— none —", width=200)
pw_white_field    = pn.widgets.Select(name="white field",     options=["— none —"], value="— none —", width=220)


def _refresh_white_field_selects():
    """Sync all white-field Select widgets to state['white_fields']."""
    labels = list(state["white_fields"].keys())
    opts   = labels if labels else ["— none —"]
    for w in [pw_white_field] + _white_field_selects:
        w.options = opts
        if w.value not in labels:
            w.value = labels[0] if labels else "— none —"


def _refresh_mask_lib_selects():
    """Sync all mask-library Select widgets to mask_library."""
    opts = list(mask_library.keys()) if mask_library else ["— empty —"]
    for w in _mask_lib_selects:
        w.options = opts
        if w.value not in mask_library:
            w.value = opts[0]


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
        _coh_plot_ops = ("Plot: Averaged coherent", "Plot: Summed coherent", "Plot: Coherent sequence")
        if op in _coh_plot_ops:
            pipe_params_col.append(pn.Row(pw_cmap, pw_log_scale, pw_pos_axes, pw_vmin, pw_vmax))
        else:
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
    elif op == "Normalise detector":
        pipe_params_col.append(pw_ref_roi)
    elif op == "Subtract white field":
        pipe_params_col.append(pw_white_field)


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
        _coh_plot_ops = ("Plot: Averaged coherent", "Plot: Summed coherent", "Plot: Coherent sequence")
        if op in _coh_plot_ops:
            p["pos_axes"] = pw_pos_axes.value
        if op == "Plot: 4D viewer (single)":
            p.update(px=pw_4d_px.value, py=pw_4d_py.value)
    elif op.startswith("Save:"):
        p.update(save_dir=pw_save_dir.value.strip(),
                 save_name=pw_save_name.value.strip() or "ptyxplore_export")
    elif op == "Normalise detector":
        p["ref_roi_name"] = pw_ref_roi.value
    elif op == "Subtract white field":
        p["white_field_label"] = pw_white_field.value
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
    pw_ref_roi.options = opts
    if pw_ref_roi.value not in names:
        pw_ref_roi.value = names[0] if names else "— none —"
    # keep stitcher ROI selector in sync (defined later at module level)
    if "_refresh_stitch_select" in globals():
        _refresh_stitch_select()  # noqa: F821
    for w in _roi_ref_selects:
        w.options = opts
        if w.value not in names:
            w.value = names[0] if names else "— none —"
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
    elif op == "Normalise detector":
        ref_roi = state["rois"].get(params["ref_roi_name"])
        if ref_roi is None:
            raise ValueError(f"Reference ROI '{params['ref_roi_name']}' not found.")
        roi_ops.normalise_detector(roi, ref_roi)
        _add_roi_history(roi_name, f"Normalise detector ✓  (ref={params['ref_roi_name']})")
    elif op == "Subtract white field":
        wf_label = params["white_field_label"]
        wf = state["white_fields"].get(wf_label)
        if wf is None:
            raise ValueError(f"White field '{wf_label}' not found. Extract it from a scan first.")
        assert roi is not None
        roi_ops.subtract_white_field(roi, wf)
        _add_roi_history(roi_name, f"Subtract white field ✓  (wf='{wf_label}')")
    elif op == "Plot: Average detector":
        if roi.averaged_det_images is None:
            raise ValueError("Run Make 4D dataset first.")
        panel = _make_bk_image_panel(
            roi.averaged_det_images, f"Avg Detector — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Average Detector — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, "Plot: Average detector → gallery")
    elif op == "Plot: Summed detector":
        if roi.data_4d is None:
            raise ValueError("Run Make 4D dataset first.")
        img = roi.data_4d.sum(axis=(0, 1)).astype(np.float32)
        panel = _make_bk_image_panel(
            img, f"Summed Detector — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"])
        _add_to_gallery(f"Summed Detector — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, f"Plot: Summed detector → gallery (shape={img.shape})")
    elif op == "Plot: Averaged coherent":
        if roi.averaged_coherent_images is None:
            raise ValueError("averaged_coherent_images not available.")
        img = roi.averaged_coherent_images
        _hc, _wc = img.shape
        _pp = _coh_pos_params(roi, _hc, _wc) if params.get("pos_axes") else None
        _xa = _pp["x_axis"] if _pp else None
        _ya = _pp["y_axis"] if _pp else None
        panel = _make_bk_image_panel(
            img, f"Avg Coherent — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"],
            x_axis_nm=_xa, y_axis_nm=_ya)
        _add_to_gallery(f"Averaged Coherent — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, "Plot: Averaged coherent → gallery")
    elif op == "Plot: Summed coherent":
        if roi.data_4d is None:
            raise ValueError("Run Make 4D dataset first.")
        img = roi.data_4d.sum(axis=(2, 3)).astype(np.float32)
        _hc, _wc = img.shape
        _pp = _coh_pos_params(roi, _hc, _wc) if params.get("pos_axes") else None
        _xa = _pp["x_axis"] if _pp else None
        _ya = _pp["y_axis"] if _pp else None
        panel = _make_bk_image_panel(
            img, f"Summed Coherent — {roi_name}",
            params["log_scale"], params["cmap"], params["vmin"], params["vmax"],
            x_axis_nm=_xa, y_axis_nm=_ya)
        _add_to_gallery(f"Summed Coherent — {roi_name}", panel, roi_name=roi_name)
        _add_roi_history(roi_name, f"Plot: Summed coherent → gallery (shape={img.shape})")
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
btn_load_frame = pn.widgets.Button(name="Load Frame",   button_type="primary", width=130)
btn_avg_frame  = pn.widgets.Button(name="Show Average", button_type="default",  width=130)
btn_std_frame  = pn.widgets.Button(name="Show Std",     button_type="default",  width=110)
btn_max_frame  = pn.widgets.Button(name="Show Max",     button_type="default",  width=110)
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


def _on_file_stat_frame(event, stat: str):
    """Shared handler for Show Std / Show Max — computes stat across all frames in the file."""
    scan = state["scan"]
    if scan is None:
        log("Load a scan first.", "warn"); return
    fidx  = w_file_idx.value
    fpath = scan.file_list[fidx]
    log(f"Computing {stat} of all frames in file {fidx+1}…", "info")
    try:
        with h5py.File(fpath, "r") as f:
            data = f["/entry/data/data"][...].astype(np.float32)
        if stat == "Std":
            result = data.std(axis=0)
        else:  # Max
            result = data.max(axis=0)
        state["current_frame"] = result.copy()
        if w_mask_hot.value:
            from ptyxplore.data_proc import mask_hot_pixels
            result = mask_hot_pixels(result)
        state["processed_frame"] = result.copy()
        fig = _make_image_fig(result,
                              f"{stat} — File {fidx+1}/{len(scan.file_list)}",
                              w_log_scale.value, w_cmap_vis.value,
                              w_vmin_vis.value or None, w_vmax_vis.value or None)
        _set_pane_fig(vis_pane, fig, "vis")
        vis_info.object = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted);margin-top:6px'>"
            f"shape: {result.shape} &nbsp;|&nbsp; min: {result.min():.1f} &nbsp;|&nbsp; "
            f"max: {result.max():.1f} &nbsp;|&nbsp; mean: {result.mean():.2f}</div>")
        log(f"{stat} computed ✓  shape={result.shape}", "ok")
    except Exception as exc:
        log(f"Failed to compute {stat}: {exc}", "err")
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
btn_std_frame.on_click(lambda e: _on_file_stat_frame(e, "Std"))
btn_max_frame.on_click(lambda e: _on_file_stat_frame(e, "Max"))
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
cb_4dv_pos     = pn.widgets.Checkbox(name="Position axes (µm)", value=False)

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
    hc, wc = coh.shape
    pp = _coh_pos_params(roi, hc, wc) if cb_4dv_pos.value else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m = np.mean(coh)
    if pp is not None:
        im0 = axes[0].imshow(coh, cmap=w_4dv_cmap_coh.value,
                             vmin=m - 0.05 * abs(m), vmax=m + 0.05 * abs(m),
                             extent=pp["extent"])
        axes[0].set_xlabel(pp["xlabel"])
        axes[0].set_ylabel(pp["ylabel"])
        rect_x = pp["px_to_nm_col"](lcol) - pp["col_nm_size"](1)
        rect_y = pp["px_to_nm_row"](lrow) - pp["row_nm_size"](1)
        rect_w = pp["col_nm_size"](2)
        rect_h = pp["row_nm_size"](2)
    else:
        im0 = axes[0].imshow(coh, cmap=w_4dv_cmap_coh.value,
                             vmin=m - 0.05 * abs(m), vmax=m + 0.05 * abs(m))
        rect_x, rect_y, rect_w, rect_h = lcol - 1, lrow - 1, 2, 2
    axes[0].set_title(f"Coherent  px=({pcol}, {prow})")
    axes[0].add_patch(Rectangle((rect_x, rect_y), rect_w, rect_h,
                                edgecolor='white', facecolor='none', lw=1.5))
    plt.colorbar(im0, ax=axes[0])
    det_disp = det.copy().astype(float)
    if getattr(roi, "detector_mask", None) is not None:
        det_disp[roi.detector_mask == 0] = np.nan
        axes[1].contour(roi.detector_mask, levels=[0.5], colors=[ACCENT], linewidths=0.8, alpha=0.7)
    m2 = np.nanmean(det_disp)
    im1 = axes[1].imshow(det_disp, cmap=w_4dv_cmap_det.value,
                         vmin=m2 - 0.3 * abs(m2), vmax=m2 + 0.2 * abs(m2))
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
    def_prow, def_pcol = _mask_default_pixel(roi)
    w_4dv_prow.end = dsh[0] - 1;  w_4dv_prow.value = def_prow
    w_4dv_pcol.end = dsh[1] - 1;  w_4dv_pcol.value = def_pcol
    w_4dv_lrow.end = csh[0] - 1;  w_4dv_lrow.value = csh[0] // 2
    w_4dv_lcol.end = csh[1] - 1;  w_4dv_lcol.value = csh[1] // 2
    _update_4d_viewer()


w_4dv_roi.param.watch(_on_4dv_roi_change, 'value')
for _w4 in (w_4dv_prow, w_4dv_pcol, w_4dv_lrow, w_4dv_lcol, w_4dv_cmap_coh, w_4dv_cmap_det, cb_4dv_pos):
    _w4.param.watch(_update_4d_viewer, 'value')

tab_vis = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Frame Browser</p>", sizing_mode="stretch_width"),
        pn.Row(w_file_idx, w_frame_idx),
        pn.Row(w_log_scale, w_mask_hot, w_cmap_vis, w_vmin_vis, w_vmax_vis, align="end"),
        pn.Row(btn_load_frame, btn_avg_frame, btn_std_frame, btn_max_frame),
        vis_pane, vis_info,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>4D Dataset Viewer</p>", sizing_mode="stretch_width"),
        pn.Row(w_4dv_roi, w_4dv_cmap_coh, w_4dv_cmap_det, cb_4dv_pos, align="end"),
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
#  Mask-builder helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _coh_pos_params(roi, hc: int, wc: int):
    """Return position-space parameters for the coherent image axes.

    Returns a dict with keys:
      img_x, img_y, img_dw, img_dh  – Bokeh Image source anchor + size in nm
      px_to_nm_col(col)              – converts pixel column index to nm
      px_to_nm_row(row)              – converts pixel row index to nm
      col_nm_size(n)                 – converts n pixels width to nm
      row_nm_size(n)                 – converts n pixels height to nm
      extent                         – matplotlib extent [l, r, bot, top] in nm
      xlabel, ylabel                 – axis labels
    Returns None when position attributes are missing from roi.
    """
    need = ("scan_x_start_point", "scan_x_end_point",
            "scan_y_start_point", "scan_y_end_point")
    if not all(hasattr(roi, a) for a in need):
        return None

    # Convert nm → µm for display.
    # x-axis: scan_x_start (left) → scan_x_end (right)
    # y-axis: scan_y_start (top)  → scan_y_end (bottom)
    img_x  = roi.scan_x_start_point / 1e3
    img_dw = (roi.scan_x_end_point - roi.scan_x_start_point) / 1e3
    img_y  = roi.scan_y_start_point / 1e3
    img_dh = (roi.scan_y_end_point - roi.scan_y_start_point) / 1e3

    # matplotlib extent=[left, right, bottom, top], origin='upper':
    # top = scan_y_start (row 0), bottom = scan_y_end (last row)
    extent = [img_x, img_x + img_dw, img_y + img_dh, img_y]
    xlabel, ylabel = "Scan X (µm)", "Scan Y (µm)"

    def px_to_nm_col(col):
        return img_x + (col + 0.5) * img_dw / wc

    def px_to_nm_row(row):
        return img_y + (row + 0.5) * img_dh / hc

    def col_nm_size(n):
        return n * img_dw / wc

    def row_nm_size(n):
        return n * img_dh / hc

    x_axis = np.linspace(img_x, img_x + img_dw, wc)
    y_axis = np.linspace(img_y, img_y + img_dh, hc)

    return dict(
        img_x=img_x, img_y=img_y, img_dw=img_dw, img_dh=img_dh,
        x_axis=x_axis, y_axis=y_axis,
        px_to_nm_col=px_to_nm_col, px_to_nm_row=px_to_nm_row,
        col_nm_size=col_nm_size, row_nm_size=row_nm_size,
        extent=extent, xlabel=xlabel, ylabel=ylabel,
    )


def _mask_default_pixel(roi) -> tuple[int, int]:
    """Return (prow, pcol) relative to the ROI bounding box.

    If the ROI has a detector_mask, returns the centroid of active pixels so
    that slider initialisation lands inside the mask rather than in the bbox
    centre (which may be in the hole of an annulus).
    Falls back to bbox centre when no mask is present.
    """
    mask = getattr(roi, "detector_mask", None)
    if mask is not None and np.any(mask):
        rows, cols = np.where(mask > 0)
        return int(rows.mean()), int(cols.mean())
    if roi.data_4d is not None:
        dsh = roi.data_4d.shape[2:]
        return dsh[0] // 2, dsh[1] // 2
    return 0, 0


def _build_full_mask(shapes: list, H: int, W: int) -> np.ndarray:
    """Return a float32 (H, W) mask that is the union of all shapes.

    Shape dicts
    -----------
    {"type": "rect",    "rs": int, "re": int, "cs": int, "ce": int}
    {"type": "annulus", "cr": int, "cc": int, "r_in": float, "r_out": float}
    {"type": "disk",    "cr": int, "cc": int, "r_out": float}
    """
    full = np.zeros((H, W), dtype=bool)
    rr, cc = np.mgrid[0:H, 0:W]
    for s in shapes:
        if s["type"] == "rect":
            full[s["rs"]:s["re"], s["cs"]:s["ce"]] = True
        else:  # annulus or disk
            dist = np.sqrt((rr - s["cr"]) ** 2 + (cc - s["cc"]) ** 2)
            r_in = s.get("r_in", 0)
            full |= (dist >= r_in) & (dist <= s["r_out"])
    return full.astype(np.float32)


def _bbox_and_mask(full_mask: np.ndarray):
    """Compute tight bounding box and crop the mask to it.

    Returns (coords [rs,re,cs,ce], mask_2d float32 of shape (re-rs, ce-cs)).
    """
    rows_any = np.any(full_mask, axis=1)
    cols_any = np.any(full_mask, axis=0)
    if not rows_any.any():
        raise ValueError("Mask is empty — no active pixels.")
    rs = int(np.argmax(rows_any))
    re = int(len(rows_any) - np.argmax(rows_any[::-1]))
    cs = int(np.argmax(cols_any))
    ce = int(len(cols_any) - np.argmax(cols_any[::-1]))
    return [rs, re, cs, ce], full_mask[rs:re, cs:ce]


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

    # ── Quick Survey ──────────────────────────────────────────────────────────
    # One global image per statistic, computed across ALL frames in ALL files.
    # Accumulator dict: keys are stat names, values are the final 2D float32 arrays.
    _sv_result  = {}   # type: dict[str, np.ndarray]  stat -> global 2D image
    _sv_thread  = [None]
    _sv_stop    = [False]

    _SV_STATS = ["Mean", "Std", "Max", "Sum"]

    sel_stat = pn.widgets.RadioButtonGroup(
        options=_SV_STATS,
        value="Mean", button_type="default", width=340,
    )
    btn_sv_start = pn.widgets.Button(name="▶ Start", button_type="primary", width=100)
    btn_sv_stop  = pn.widgets.Button(name="■ Stop",  button_type="danger",  width=80, disabled=True)
    sv_status    = pn.pane.HTML(
        "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "color:var(--txt-muted)'>Press Start to compute global stat images across all files.</span>",
        sizing_mode="stretch_width")
    sv_progress  = pn.widgets.Progress(
        active=False, value=0, max=max(1, n_files),
        bar_color="success", sizing_mode="stretch_width", height=8,
    )

    # vmin/vmax + display controls
    fi_sv_vmin  = pn.widgets.FloatInput(name="vmin (blank=auto)", width=140)
    fi_sv_vmax  = pn.widgets.FloatInput(name="vmax (blank=auto)", width=140)
    btn_sv_auto = pn.widgets.Button(name="Autoscale", button_type="default", width=100)
    cb_sv_log   = pn.widgets.Checkbox(name="Log scale", value=False)
    cb_sv_hot   = pn.widgets.Checkbox(name="Mask hot pixels", value=True)
    sel_sv_cmap = pn.widgets.Select(
        name="Colormap",
        options=["viridis", "jet", "plasma", "inferno", "gray", "hot", "turbo"],
        value="viridis", width=130,
    )
    sv_info = pn.pane.HTML("", sizing_mode="stretch_width")

    # Dedicated Bokeh figure for the survey (independent of frame browser)
    _sv_blank = np.zeros((64, 64), dtype=np.float32)
    _sv_src   = ColumnDataSource({"image": [_sv_blank], "x": [0], "y": [0], "dw": [64], "dh": [64]})
    _sv_map   = LinearColorMapper(palette=_bk_palette("viridis"), low=0, high=1, nan_color="black")
    _sv_fig   = bk_figure(
        sizing_mode="stretch_width", height=440,
        tools="pan,wheel_zoom,box_select,reset,tap", active_scroll="wheel_zoom",
        match_aspect=True,
        background_fill_color=BG_DARK, border_fill_color=BG_MID,
    )
    _sv_fig.y_range.flipped = True
    _sv_fig.image("image", "x", "y", "dw", "dh", source=_sv_src, color_mapper=_sv_map)
    _sv_fig.add_tools(HoverTool(tooltips=[("col", "$x{0}"), ("row", "$y{0}"), ("val", "@image")]))
    _sv_fig.add_layout(ColorBar(color_mapper=_sv_map, ticker=BasicTicker(), location=(0, 0)), "right")

    # Tap + box-select tracking for the survey figure
    _sv_cur_src = ColumnDataSource({'x': [0.0], 'y': [0.0]})
    _sv_sel_src = ColumnDataSource({'x0': [0.0], 'y0': [0.0], 'x1': [0.0], 'y1': [0.0]})
    _sv_fig.js_on_event('tap', CustomJS(args={'s': _sv_cur_src}, code="""
        s.data = {x: [cb_obj.x], y: [cb_obj.y]};  s.change.emit();
    """))
    _sv_fig.js_on_event('selectiongeometry', CustomJS(args={'s': _sv_sel_src}, code="""
        if (cb_obj.geometry && cb_obj.geometry.type === 'rect') {
            const g = cb_obj.geometry;
            s.data = {x0: [Math.min(g.x0,g.x1)], y0: [Math.min(g.y0,g.y1)],
                      x1: [Math.max(g.x0,g.x1)], y1: [Math.max(g.y0,g.y1)]};
            s.change.emit();
        }
    """))

    _sv_cursor = {'row': 0, 'col': 0}
    _sv_sel    = {'rs': 0, 're': 0, 'cs': 0, 'ce': 0}
    sv_coord_info = pn.pane.HTML(
        "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "color:var(--txt-muted)'>tap a point · drag to select rect</span>",
        sizing_mode="stretch_width")

    def _fmt_sv_coord():
        sv_coord_info.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted)'>"
            f"cursor → row={_sv_cursor['row']}, col={_sv_cursor['col']}"
            f" &nbsp;|&nbsp; box → row=[{_sv_sel['rs']}, {_sv_sel['re']}]"
            f" col=[{_sv_sel['cs']}, {_sv_sel['ce']}]</span>")

    def _on_sv_cur(attr, old, new):
        _sv_cursor['col'] = max(0, int(new['x'][0]))
        _sv_cursor['row'] = max(0, int(new['y'][0]))
        pn.state.execute(_fmt_sv_coord)

    def _on_sv_sel(attr, old, new):
        _sv_sel['cs'] = max(0, int(new['x0'][0]))
        _sv_sel['ce'] = max(0, int(new['x1'][0]))
        _sv_sel['rs'] = max(0, int(new['y0'][0]))
        _sv_sel['re'] = max(0, int(new['y1'][0]))
        pn.state.execute(_fmt_sv_coord)

    _sv_cur_src.on_change('data', _on_sv_cur)
    _sv_sel_src.on_change('data', _on_sv_sel)

    def _sv_show_image():
        """Push the global stat image for the selected stat into the Bokeh figure."""
        stat = sel_stat.value
        img  = _sv_result.get(stat)
        if img is None:
            sv_info.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--txt-muted)'>{stat} not yet computed — press Start.</span>")
            return
        disp = np.log1p(np.abs(img.astype(float))) if cb_sv_log.value else img.astype(float)
        h, w = disp.shape
        lo = fi_sv_vmin.value if fi_sv_vmin.value is not None else float(np.nanmin(disp))
        hi = fi_sv_vmax.value if fi_sv_vmax.value is not None else float(np.nanmax(disp))
        if abs(hi - lo) < 1e-10:
            hi = lo + 1.0
        _sv_map.palette = _bk_palette(sel_sv_cmap.value)
        _sv_map.low     = lo
        _sv_map.high    = hi
        _sv_src.data    = {"image": [disp], "x": [0], "y": [0], "dw": [w], "dh": [h]}
        sv_info.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted)'>"
            f"{stat} across all {n_files} files &nbsp;|&nbsp; "
            f"shape={img.shape} &nbsp;|&nbsp; max={float(np.nanmax(img)):.1f}</span>")

    def _sv_autoscale(event):
        fi_sv_vmin.value = fi_sv_vmax.value = None
        _sv_show_image()

    btn_sv_auto.on_click(_sv_autoscale)
    fi_sv_vmin.param.watch(lambda e: _sv_show_image(), "value")
    fi_sv_vmax.param.watch(lambda e: _sv_show_image(), "value")
    cb_sv_log.param.watch(lambda e: _sv_show_image(), "value")
    sel_sv_cmap.param.watch(lambda e: _sv_show_image(), "value")
    sel_stat.param.watch(lambda e: _sv_show_image(), "value")

    def _sv_run():
        """
        Background worker: one pass over all files, accumulates running statistics
        for Mean, Std, Max, and Sum simultaneously.

        Uses the parallel-variance formula so only one read per file is needed:
          total_sum, total_sq_sum, total_max, total_count  →  mean, std, max, sum
        """
        total_count  = 0
        total_sum    = None   # float64 accumulator
        total_sq_sum = None   # for variance: E[X²]
        total_max    = None   # running element-wise max

        for fidx, fpath in enumerate(scan.file_list):
            if _sv_stop[0]:
                break
            try:
                with h5py.File(fpath, "r") as f:
                    data = f["/entry/data/data"][...].astype(np.float64)
                # data shape: (M_frames, H, W)
                n_frames = data.shape[0]
                file_sum    = data.sum(axis=0)
                file_sq_sum = (data ** 2).sum(axis=0)
                file_max    = data.max(axis=0)

                if total_sum is None:
                    total_sum    = file_sum
                    total_sq_sum = file_sq_sum
                    total_max    = file_max
                else:
                    total_sum    += file_sum
                    total_sq_sum += file_sq_sum
                    np.maximum(total_max, file_max, out=total_max)

                total_count += n_frames

            except Exception as exc:
                pn.state.execute(
                    lambda e=exc, fi=fidx: log(
                        f"[{scan_label}] Survey: file {fi} failed — {e}", "warn"))

            def _tick(nd=fidx + 1):
                sv_progress.value = nd
                sv_status.object  = (
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                    f"color:var(--txt-muted)'>Computing … {nd}/{n_files} files done</span>")

            pn.state.execute(_tick)

        # Finalise results
        if total_sum is not None and total_count > 0:
            mean = (total_sum / total_count).astype(np.float32)
            var  = np.maximum(total_sq_sum / total_count - (mean.astype(np.float64)) ** 2, 0.0)
            std  = np.sqrt(var).astype(np.float32)
            mx   = total_max.astype(np.float32)
            sm   = total_sum.astype(np.float32)

            if cb_sv_hot.value:
                try:
                    from ptyxplore.data_proc import mask_hot_pixels
                    mean = mask_hot_pixels(mean)
                    std  = mask_hot_pixels(std)
                    mx   = mask_hot_pixels(mx)
                    sm   = mask_hot_pixels(sm)
                except Exception:
                    pass

            _sv_result["Mean"] = mean
            _sv_result["Std"]  = std
            _sv_result["Max"]  = mx
            _sv_result["Sum"]  = sm

        def _finish():
            sv_progress.active    = False
            sv_progress.value     = n_files
            btn_sv_start.disabled = False
            btn_sv_stop.disabled  = True
            if total_sum is not None:
                sv_status.object = (
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                    f"color:var(--accent)'>✓ Survey complete — {n_files} files, "
                    f"{total_count} frames</span>")
                _sv_show_image()
                log(f"[{scan_label}] Survey complete — {n_files} files, {total_count} frames", "ok")
            else:
                sv_status.object = (
                    "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                    "color:var(--warn)'>Survey produced no results.</span>")

        pn.state.execute(_finish)

    def _on_sv_start(event):
        # Stop any running survey first
        _sv_stop[0] = True
        if _sv_thread[0] and _sv_thread[0].is_alive():
            _sv_thread[0].join(timeout=2)
        _sv_stop[0] = False
        _sv_result.clear()
        btn_sv_start.disabled = True
        btn_sv_stop.disabled  = False
        sv_progress.active    = True
        sv_progress.value     = 0
        t = threading.Thread(target=_sv_run, daemon=True)
        _sv_thread[0] = t
        t.start()
        log(f"[{scan_label}] Starting survey over {n_files} files…", "info")

    def _on_sv_stop(event):
        _sv_stop[0] = True
        btn_sv_start.disabled = False
        btn_sv_stop.disabled  = True
        sv_progress.active = False
        sv_status.object = (
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--warn)'>Survey stopped.</span>")

    btn_sv_start.on_click(_on_sv_start)
    btn_sv_stop.on_click(_on_sv_stop)

    # ── White Field Extraction ────────────────────────────────────────────────
    _wf_thread = [None]
    _wf_stop   = [False]

    w_wf_label = pn.widgets.TextInput(
        name="Label", value=f"{scan_label}_wf",
        placeholder="e.g. background_001", width=220,
    )
    w_wf_stat = pn.widgets.RadioButtonGroup(
        options=["Mean", "Max", "Median"], value="Mean",
        button_type="default", width=280,
    )
    btn_wf_start  = pn.widgets.Button(name="▶ Extract", button_type="primary", width=110)
    btn_wf_stop   = pn.widgets.Button(name="■ Stop",    button_type="danger",  width=80, disabled=True)
    btn_wf_delete = pn.widgets.Button(name="Delete stored", button_type="danger", width=130, disabled=True)
    wf_progress   = pn.widgets.Progress(
        active=False, value=0, max=max(1, n_files),
        bar_color="success", sizing_mode="stretch_width", height=8,
    )
    wf_status = pn.pane.HTML(
        "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
        "color:var(--txt-muted)'>Choose a label and statistic, then press Extract.</span>",
        sizing_mode="stretch_width",
    )
    wf_preview_pane = pn.pane.Matplotlib(sizing_mode="stretch_width", height=260, tight=True)
    _wf_preview_fig = [None]

    def _wf_show_preview(img: np.ndarray, label: str):
        if _wf_preview_fig[0] is not None:
            try:
                plt.close(_wf_preview_fig[0])
            except Exception:
                pass
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=BG_DARK)
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TXT_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        disp = np.log1p(img.astype(float))
        im = ax.imshow(disp, cmap="viridis", origin="upper")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log(1+I)", color=TXT_MUTED, fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT_MUTED)
        ax.set_title(f"White field: {label}  shape={img.shape}  max={img.max():.1f}",
                     color=TXT, fontsize=9)
        fig.patch.set_facecolor(BG_DARK)
        plt.tight_layout()
        _wf_preview_fig[0] = fig
        wf_preview_pane.object = fig
        plt.close(fig)

    def _wf_run(label: str, stat: str):
        _wf_stop[0] = False
        acc   = None
        count = 0
        for fidx, fpath in enumerate(scan.file_list):
            if _wf_stop[0]:
                break
            try:
                with h5py.File(fpath, "r") as f:
                    data = f["/entry/data/data"][...].astype(np.float32)
                if stat == "Mean":
                    frame_stat = data.mean(axis=0)
                elif stat == "Max":
                    frame_stat = data.max(axis=0)
                else:  # Median — per-file median, then averaged across files
                    frame_stat = np.median(data, axis=0)
                acc   = frame_stat if acc is None else acc + frame_stat
                count += 1
            except Exception as exc:
                pn.state.execute(lambda e=exc, fi=fidx: log(
                    f"[{scan_label}] WF: file {fi} failed — {e}", "warn"))

            def _tick(fi=fidx, nd=count):
                wf_progress.value = nd
                wf_status.object = (
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                    f"color:var(--txt-muted)'>Computing {stat} … {nd}/{n_files} files done</span>")
            pn.state.execute(_tick)

        if acc is None or count == 0:
            pn.state.execute(lambda: log(f"[{scan_label}] WF extraction failed — no files read.", "err"))
            return

        # For Mean and Median: divide accumulated sum by file count
        if stat in ("Mean", "Median"):
            acc = acc / count
        result = acc.astype(np.float32)

        def _finish(img=result, lbl=label):
            state["white_fields"][lbl] = img
            _refresh_white_field_selects()
            wf_progress.active  = False
            wf_progress.value   = count
            btn_wf_start.disabled  = False
            btn_wf_stop.disabled   = True
            btn_wf_delete.disabled = False
            wf_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--accent)'>✓ '{lbl}' stored — "
                f"shape={img.shape}  max={img.max():.1f}</span>")
            _wf_show_preview(img, lbl)
            log(f"[{scan_label}] White field '{lbl}' extracted ✓  "
                f"({stat}, {count}/{n_files} files, shape={img.shape})", "ok")
            pn.state.notifications.success(f"White field '{lbl}' ready!", duration=3000)

        pn.state.execute(_finish)

    def _on_wf_start(event):
        label = w_wf_label.value.strip()
        if not label:
            pn.state.notifications.warning("Enter a label first.", duration=3000)
            return
        stat = w_wf_stat.value
        _wf_stop[0] = True
        if _wf_thread[0] and _wf_thread[0].is_alive():
            _wf_thread[0].join(timeout=2)
        _wf_stop[0] = False
        btn_wf_start.disabled  = True
        btn_wf_stop.disabled   = False
        btn_wf_delete.disabled = True
        wf_progress.active = True
        wf_progress.value  = 0
        t = threading.Thread(target=_wf_run, args=(label, stat), daemon=True)
        _wf_thread[0] = t
        t.start()
        log(f"[{scan_label}] Extracting white field '{label}' ({stat}, {n_files} files)…", "info")

    def _on_wf_stop(event):
        _wf_stop[0] = True
        btn_wf_start.disabled  = False
        btn_wf_stop.disabled   = True
        wf_progress.active = False
        wf_status.object = (
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--warn)'>Extraction stopped.</span>")

    def _on_wf_delete(event):
        label = w_wf_label.value.strip()
        if label in state["white_fields"]:
            del state["white_fields"][label]
            _refresh_white_field_selects()
            btn_wf_delete.disabled = True
            wf_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--warn)'>'{label}' deleted.</span>")
            wf_preview_pane.object = None
            log(f"White field '{label}' deleted.", "warn")

    btn_wf_start.on_click(_on_wf_start)
    btn_wf_stop.on_click(_on_wf_stop)
    btn_wf_delete.on_click(_on_wf_delete)

    # If a white field for this scan label already exists, show its preview
    _existing_wf = state["white_fields"].get(f"{scan_label}_wf")
    if _existing_wf is not None:
        _wf_show_preview(_existing_wf, f"{scan_label}_wf")
        btn_wf_delete.disabled = False

    # ── Mask Builder ──────────────────────────────────────────────────────────
    # Local shape list for this scan tab
    _shapes: list[dict] = []

    # Shape type selector
    w_shape_type = pn.widgets.RadioButtonGroup(
        name="Shape type", options=["Rect", "Annulus", "Disk"],
        value="Rect", button_type="default", width=300,
    )

    # --- Rect widgets ---
    w_rs = pn.widgets.IntInput(name="row start", value=0,   width=105)
    w_re = pn.widgets.IntInput(name="row end",   value=100, width=105)
    w_cs = pn.widgets.IntInput(name="col start", value=0,   width=105)
    w_ce = pn.widgets.IntInput(name="col end",   value=100, width=105)
    btn_use_sel    = pn.widgets.Button(name="← Frame sel",  button_type="default", width=120)
    btn_use_sv_sel = pn.widgets.Button(name="← Survey sel", button_type="default", width=120)

    rect_params = pn.Column(
        pn.Row(w_rs, w_re, w_cs, w_ce),
        pn.Row(btn_use_sel, btn_use_sv_sel),
    )

    # --- Annulus / Disk widgets (shared center) ---
    w_cr    = pn.widgets.IntInput(  name="center row",    value=256, width=120)
    w_cc    = pn.widgets.IntInput(  name="center col",    value=256, width=120)
    w_r_in  = pn.widgets.FloatInput(name="inner radius",  value=0,   width=115)
    w_r_out = pn.widgets.FloatInput(name="outer radius",  value=100, width=115)
    btn_use_cur = pn.widgets.Button(name="← Use last p", button_type="default", width=120)

    annulus_params = pn.Column(pn.Row(w_cr, w_cc, w_r_in, w_r_out), pn.Row(btn_use_cur))
    disk_params    = pn.Column(pn.Row(w_cr, w_cc, w_r_out),          pn.Row(btn_use_cur))

    param_panel = pn.Column(rect_params, sizing_mode="stretch_width")

    def _on_shape_type(event):
        param_panel.clear()
        if event.new == "Rect":
            param_panel.append(rect_params)
        elif event.new == "Annulus":
            param_panel.append(annulus_params)
        else:
            param_panel.append(disk_params)

    w_shape_type.param.watch(_on_shape_type, "value")

    def _use_sel(event):
        w_rs.value = _sel["rs"]
        w_re.value = _sel["re"]
        w_cs.value = _sel["cs"]
        w_ce.value = _sel["ce"]

    def _use_sv_sel(event):
        w_rs.value = _sv_sel["rs"]
        w_re.value = _sv_sel["re"]
        w_cs.value = _sv_sel["cs"]
        w_ce.value = _sv_sel["ce"]

    def _use_cur(event):
        w_cr.value = _cursor["row"]
        w_cc.value = _cursor["col"]

    btn_use_sel.on_click(_use_sel)
    btn_use_sv_sel.on_click(_use_sv_sel)
    btn_use_cur.on_click(_use_cur)

    # Shape list display + live mask preview
    shapes_col       = pn.Column(sizing_mode="stretch_width")
    mask_preview_pane = pn.pane.Matplotlib(
        sizing_mode="stretch_width", height=300, tight=True)
    _mask_preview_fig = [None]

    def _shape_desc(s: dict) -> str:
        if s["type"] == "rect":
            return (f"<b>Rect</b>  row=[{s['rs']}..{s['re']}]  "
                    f"col=[{s['cs']}..{s['ce']}]")
        elif s["type"] == "annulus":
            return (f"<b>Annulus</b>  r=[{s['r_in']:.0f}..{s['r_out']:.0f}]  "
                    f"ctr=({s['cr']},{s['cc']})")
        else:
            return f"<b>Disk</b>  r=0..{s['r_out']:.0f}  ctr=({s['cr']},{s['cc']})"

    def _get_det_shape():
        """Return (H, W) of the detector.

        Tries the frame already loaded in the Bokeh viewer first; falls back to
        reading just the shape from the first h5 file (fast metadata read).
        Returns (None, None) if both fail.
        """
        imgs = _bk_src.data.get("image", [])
        if imgs:
            img = imgs[0]
            if hasattr(img, "shape") and img.ndim == 2 and min(img.shape) > 1:
                return img.shape
        try:
            with h5py.File(scan.file_list[0], "r") as f:
                sh = f["/entry/data/data"].shape   # (M, H, W)
                return int(sh[1]), int(sh[2])
        except Exception:
            return None, None

    def _update_mask_preview():
        """Render the current shape list as a 2-tone image in mask_preview_pane."""
        prev = _mask_preview_fig[0]
        _mask_preview_fig[0] = None
        if not _shapes:
            mask_preview_pane.object = None
            if prev is not None:
                try:
                    plt.close(prev)
                except Exception:
                    pass
            return

        H, W = _get_det_shape()
        if H is None:
            mask_preview_pane.object = None
            return

        full_mask = _effective_mask(H, W)
        try:
            coords, _ = _bbox_and_mask(full_mask)
        except ValueError:
            mask_preview_pane.object = None
            return
        rs, re, cs, ce = coords
        n_px  = int(full_mask.sum())
        pct   = 100.0 * n_px / (H * W)

        # Two-tone RGB: ACCENT (#0FCFAF) = active, BG_DARK (#0D1117) = inactive
        acc = np.array([0x0F / 255, 0xCF / 255, 0xAF / 255])
        bg  = np.array([0x0D / 255, 0x11 / 255, 0x17 / 255])
        rgb = np.where(full_mask[:, :, None], acc, bg).astype(np.float32)

        fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG_DARK)
        ax.set_facecolor(BG_DARK)
        ax.imshow(rgb, origin="upper", interpolation="nearest", aspect="equal")
        # Bounding box overlay
        from matplotlib.patches import Rectangle as _Rect
        ax.add_patch(_Rect(
            (cs - 0.5, rs - 0.5), ce - cs, re - rs,
            linewidth=1.2, edgecolor=WARN, facecolor="none",
            linestyle="--", alpha=0.85,
        ))
        inv_tag = " [INVERTED]" if cb_invert.value else ""
        ax.set_title(
            f"{len(_shapes)} shape(s){inv_tag} · {n_px:,} px ({pct:.1f}%) · "
            f"bbox row[{rs}:{re}] col[{cs}:{ce}]",
            color=TXT, fontsize=9,
        )
        ax.tick_params(colors=TXT_MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        fig.patch.set_facecolor(BG_DARK)
        plt.tight_layout()
        if prev is not None:
            try:
                plt.close(prev)
            except Exception:
                pass
        _mask_preview_fig[0] = fig
        mask_preview_pane.object = fig

    def _rebuild_shapes():
        shapes_col.clear()
        if not _shapes:
            shapes_col.append(pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>No shapes yet — add at least one.</span>"))
        else:
            for i, s in enumerate(_shapes):
                def _make_rm(idx):
                    btn = pn.widgets.Button(name="×", button_type="danger",
                                            width=34, height=28, margin=(0, 0))
                    def _rm(event, _i=idx):
                        _shapes.pop(_i)
                        _rebuild_shapes()
                    btn.on_click(_rm)
                    return btn
                shapes_col.append(pn.Row(
                    pn.pane.HTML(
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:12px;"
                        f"color:var(--txt)'>"
                        f"<span style='color:var(--txt-muted)'>[{i+1}]</span> "
                        f"{_shape_desc(s)}</span>",
                        sizing_mode="stretch_width"),
                    _make_rm(i),
                    align="center", sizing_mode="stretch_width",
                ))
        _update_mask_preview()

    _rebuild_shapes()

    # Control buttons
    btn_add_shape     = pn.widgets.Button(name="Add shape",      button_type="primary", width=120)
    btn_apply_to_frame = pn.widgets.Button(name="Apply to frame", button_type="default", width=130)
    btn_clear_shapes  = pn.widgets.Button(name="Clear all",       button_type="danger",  width=100)
    cb_invert         = pn.widgets.Checkbox(name="Invert mask",   value=False)

    def _on_add_shape(event):
        stype = w_shape_type.value
        if stype == "Rect":
            if w_re.value <= w_rs.value or w_ce.value <= w_cs.value:
                log("Invalid rect: end must be > start.", "warn")
                pn.state.notifications.warning("row/col end must exceed start.", duration=3000)
                return
            _shapes.append({"type": "rect",
                             "rs": w_rs.value, "re": w_re.value,
                             "cs": w_cs.value, "ce": w_ce.value})
        elif stype == "Annulus":
            if w_r_out.value <= 0 or w_r_in.value < 0 or w_r_in.value >= w_r_out.value:
                log("Invalid annulus: need 0 ≤ inner_r < outer_r.", "warn")
                pn.state.notifications.warning("Need 0 ≤ inner_r < outer_r.", duration=3000)
                return
            _shapes.append({"type": "annulus",
                             "cr": w_cr.value, "cc": w_cc.value,
                             "r_in": w_r_in.value, "r_out": w_r_out.value})
        else:  # Disk
            if w_r_out.value <= 0:
                log("Invalid disk: outer_r must be > 0.", "warn")
                pn.state.notifications.warning("outer_r must be > 0.", duration=3000)
                return
            _shapes.append({"type": "disk",
                             "cr": w_cr.value, "cc": w_cc.value,
                             "r_out": w_r_out.value})
        _rebuild_shapes()
        log(f"[{scan_label}] Shape added ({_shapes[-1]['type']}). Total: {len(_shapes)}", "info")

    def _on_apply_to_frame(event):
        """Multiply the currently loaded frame by the mask and show it in the frame browser."""
        if not _shapes:
            pn.state.notifications.warning("Add at least one shape first.", duration=3000)
            return
        imgs = _bk_src.data.get("image", [])
        if not imgs or not hasattr(imgs[0], "shape") or min(imgs[0].shape) <= 1:
            pn.state.notifications.warning("Load a frame first.", duration=3000)
            return
        current = imgs[0]
        H, W = current.shape
        mask_2d = _effective_mask(H, W)
        _update_bk(current * mask_2d)
        n_px = int(mask_2d.sum())
        log(f"[{scan_label}] Mask applied to frame: {n_px} / {H*W} px active "
            f"({100*n_px/(H*W):.1f}%) — reload frame to restore.", "info")
        pn.state.notifications.info("Mask applied — reload frame to restore.", duration=4000)

    def _on_clear_shapes(event):
        _shapes.clear()
        _rebuild_shapes()

    def _effective_mask(H: int, W: int) -> np.ndarray:
        """Return the combined mask, optionally inverted."""
        m = _build_full_mask(_shapes, H, W)
        if cb_invert.value:
            m = 1.0 - m
        return m

    btn_add_shape.on_click(_on_add_shape)
    btn_apply_to_frame.on_click(_on_apply_to_frame)
    btn_clear_shapes.on_click(_on_clear_shapes)
    cb_invert.param.watch(lambda e: _update_mask_preview(), "value")

    # ── Save / Load mask ─────────────────────────────────────────────────────
    import json as _json

    w_mask_path = pn.widgets.TextInput(
        name="Mask file path (.json)",
        placeholder="/path/to/my_mask.json",
        sizing_mode="stretch_width",
    )
    btn_save_mask = pn.widgets.Button(name="Save mask", button_type="default", width=115)
    btn_load_mask = pn.widgets.Button(name="Load mask", button_type="default", width=115)
    mask_io_status = pn.pane.HTML("", sizing_mode="stretch_width")

    def _on_save_mask(event):
        path = w_mask_path.value.strip()
        if not path:
            mask_io_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>Enter a file path first.</span>")
            return
        if not _shapes:
            mask_io_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>No shapes to save.</span>")
            return
        try:
            if not path.endswith(".json"):
                path += ".json"
                w_mask_path.value = path
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {"ptyxplore_mask_version": 1, "shapes": _shapes}
            with open(path, "w") as fh:
                _json.dump(payload, fh, indent=2)
            mask_io_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--accent)'>Saved {len(_shapes)} shape(s) → {path}</span>")
            log(f"[{scan_label}] Mask saved → {path}", "ok")
        except Exception as exc:
            mask_io_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--warn)'>{exc}</span>")
            log(f"Failed to save mask: {exc}", "err")

    def _on_load_mask(event):
        path = w_mask_path.value.strip()
        if not path:
            mask_io_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>Enter a file path first.</span>")
            return
        try:
            with open(path) as fh:
                payload = _json.load(fh)
            if "shapes" not in payload:
                raise ValueError("File does not contain a 'shapes' key.")
            loaded = payload["shapes"]
            _shapes.clear()
            _shapes.extend(loaded)
            _rebuild_shapes()
            mask_io_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--accent)'>Loaded {len(loaded)} shape(s) from {path}</span>")
            log(f"[{scan_label}] Mask loaded ← {path}  ({len(loaded)} shapes)", "ok")
        except Exception as exc:
            mask_io_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--warn)'>{exc}</span>")
            log(f"Failed to load mask: {exc}", "err")

    btn_save_mask.on_click(_on_save_mask)
    btn_load_mask.on_click(_on_load_mask)

    # ── Mask library (in-memory, shared across all scan tabs) ─────────────────
    w_lib_save_name = pn.widgets.TextInput(
        name="Library entry name",
        placeholder="e.g. central_disk",
        sizing_mode="stretch_width",
    )
    btn_lib_save = pn.widgets.Button(name="Save to library", button_type="default", width=140)

    _lib_opts = list(mask_library.keys()) if mask_library else ["— empty —"]
    w_lib_select = pn.widgets.Select(
        name="Saved mask",
        options=_lib_opts,
        value=_lib_opts[0],
        sizing_mode="stretch_width",
    )
    _mask_lib_selects.append(w_lib_select)

    btn_lib_load   = pn.widgets.Button(name="Load from library", button_type="default", width=145)
    btn_lib_delete = pn.widgets.Button(name="Delete",            button_type="danger",  width=90)
    lib_status     = pn.pane.HTML("", sizing_mode="stretch_width")

    def _on_lib_save(event):
        name = w_lib_save_name.value.strip()
        if not name:
            lib_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>Enter a library entry name first.</span>")
            return
        if not _shapes:
            lib_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>No shapes to save.</span>")
            return
        import copy
        mask_library[name] = copy.deepcopy(_shapes)
        _refresh_mask_lib_selects()
        w_lib_select.value = name
        lib_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--accent)'>Saved {len(_shapes)} shape(s) as '{name}'.</span>")
        log(f"[{scan_label}] Mask '{name}' saved to library ({len(_shapes)} shapes)", "ok")

    def _on_lib_load(event):
        sel = w_lib_select.value
        if sel not in mask_library:
            lib_status.object = (
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--warn)'>No mask selected.</span>")
            return
        import copy
        loaded = copy.deepcopy(mask_library[sel])
        _shapes.clear()
        _shapes.extend(loaded)
        _rebuild_shapes()
        lib_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--accent)'>Loaded '{sel}' ({len(loaded)} shapes).</span>")
        log(f"[{scan_label}] Mask '{sel}' loaded from library", "ok")

    def _on_lib_delete(event):
        sel = w_lib_select.value
        if sel not in mask_library:
            return
        del mask_library[sel]
        _refresh_mask_lib_selects()
        lib_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted)'>Deleted '{sel}' from library.</span>")
        log(f"[{scan_label}] Mask '{sel}' removed from library", "info")

    btn_lib_save.on_click(_on_lib_save)
    btn_lib_load.on_click(_on_lib_load)
    btn_lib_delete.on_click(_on_lib_delete)

    # Final ROI creation
    w_new_name = pn.widgets.TextInput(name="ROI name", placeholder="e.g. bragg_001", width=180)
    w_new_kind = pn.widgets.Select(name="Kind", options=["Bragg", "Pupil"], value="Bragg", width=100)
    btn_mk_roi = pn.widgets.Button(name="Create ROI", button_type="success", width=160)

    def _on_mk_roi(event):
        name = w_new_name.value.strip()
        kind = w_new_kind.value.lower()
        if not name:
            log("Enter a name for the new ROI.", "warn")
            pn.state.notifications.warning("Enter ROI name.", duration=3000)
            return
        if name in state["rois"]:
            log(f"ROI '{name}' already exists.", "err")
            pn.state.notifications.error(f"Name '{name}' already in use.", duration=3000)
            return
        if not _shapes:
            log("Add at least one shape before creating an ROI.", "warn")
            pn.state.notifications.warning("Add at least one shape first.", duration=3000)
            return
        try:
            # Determine detector shape from the first scan file
            with h5py.File(scan.file_list[0], "r") as f:
                det_shape = f["/entry/data/data"].shape[1:]  # (H, W)
            H, W = det_shape
            full_mask = _effective_mask(H, W)
            coords, bbox_mask = _bbox_and_mask(full_mask)
            n_px = int(bbox_mask.sum())
            from ptyxplore.cbd_loader import ROI
            roi = ROI(scan=scan, kind=kind, coords=coords,
                      centre_pixel=[0.0, 0.0],
                      detector_mask=bbox_mask.astype(np.float32))
            state["rois"][name] = roi
            state["roi_history"][name] = [
                f"[{ts()}] Created ({kind}, {len(_shapes)} shape(s), "
                f"bbox={coords}, {n_px} active px)"]
            if kind == "pupil" and state["pupil_name"] is None:
                state["pupil_name"] = name
            _rebuild_roi_cards()
            _refresh_pipe_roi_select()
            tabs.append((f"{scan_label}: {name}", _make_roi_tab(name)))
            log(f"ROI '{name}' created ✓  bbox={coords}  {n_px} px active", "ok")
            pn.state.notifications.success(f"ROI '{name}' added!", duration=2000)
        except Exception as exc:
            log(f"Failed to create ROI: {exc}", "err")
            pn.state.notifications.error(str(exc), duration=5000)

    btn_mk_roi.on_click(_on_mk_roi)

    # ── Refresh scan button ───────────────────────────────────────────────────
    btn_refresh = pn.widgets.Button(name="⟳ Refresh", button_type="default", width=110)

    def _refresh(event):
        import glob as _glob
        import pandas as pd
        try:
            files_h5  = sorted(_glob.glob(os.path.join(scan.data_dir, "*.h5")))
            files_nxs = sorted(_glob.glob(os.path.join(scan.data_dir, "*.nxs")))
            new_files = files_h5 if files_h5 else files_nxs
            scan.file_list = new_files
            s_file.end = max(0, len(new_files) - 1)
            ftable.value = pd.DataFrame(
                [{"#": i + 1, "File": f} for i, f in enumerate(new_files)]
            )
            stats_pane.object = _make_stats_html(scan)
            log(f"Scan '{scan_label}' refreshed ✓  ({len(new_files)} files)", "ok")
            pn.state.notifications.success(f"Scan '{scan_label}' refreshed!", duration=2000)
        except Exception as exc:
            log(f"Failed to refresh scan '{scan_label}': {exc}", "err")
            pn.state.notifications.error(str(exc), duration=5000)

    btn_refresh.on_click(_refresh)

    # ── Remove scan button ────────────────────────────────────────────────────
    btn_remove = pn.widgets.Button(name="Remove Scan", button_type="danger", width=140)

    def _remove(event):
        if scan_label not in state["scans"]:
            return
        # Remove all ROIs belonging to this scan
        rois_to_remove = [
            name for name, roi in state["rois"].items()
            if roi.scan is scan
        ]
        for name in rois_to_remove:
            del state["rois"][name]
            state["roi_history"].pop(name, None)
            if state["pupil_name"] == name:
                state["pupil_name"] = None
            roi_4dv_hooks.pop(name, None)
            roi_gallery_hooks.pop(name, None)
            # Remove ROI tab
            tab_suffix = f": {name}"
            for i in reversed([i for i, tname in enumerate(tabs._names)
                                if tname.endswith(tab_suffix)]):
                tabs.pop(i)
            # Remove gallery plots for this ROI
            for i in reversed([i for i, item in enumerate(plot_gallery)
                                if item.get("roi_name") == name]):
                removed = plot_gallery.pop(i)
                try:
                    plt.close(removed["fig"])
                except Exception:
                    pass
        if rois_to_remove:
            _rebuild_gallery()
            _rebuild_roi_cards()
            _refresh_pipe_roi_select()
        # Remove the scan itself
        del state["scans"][scan_label]
        _refresh_scan_select()
        _rebuild_loaded_scans()
        scan_tab_name = f"Scan {getattr(scan, 'scan_num', scan_label)}"
        for i in reversed([i for i, tname in enumerate(tabs._names)
                           if tname == scan_tab_name]):
            tabs.pop(i)
        log(f"Scan '{scan_label}' removed"
            + (f" (and {len(rois_to_remove)} ROI(s))" if rois_to_remove else "") + ".",
            "warn")

    btn_remove.on_click(_remove)

    return pn.Column(
        pn.Column(
            pn.pane.HTML("<p class='section-label'>Scan Info</p>", sizing_mode="stretch_width"),
            info_html, stats_pane,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            ftable,
            pn.Row(btn_refresh, btn_remove),
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
            pn.pane.HTML("<p class='section-label'>Quick Survey</p>",
                         sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Computes one global image per statistic across "
                "<b>all</b> frames in <b>all</b> files in a single background pass. "
                "Switch between Mean / Std / Max / Sum with the radio buttons.</span>",
                sizing_mode="stretch_width"),
            pn.Row(btn_sv_start, btn_sv_stop, cb_sv_hot, align="end"),
            sv_progress,
            sv_status,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                         "font-family:JetBrains Mono,monospace;margin:0 0 6px 0'>"
                         "STATISTIC</p>", sizing_mode="stretch_width"),
            pn.Row(sel_stat),
            pn.Row(cb_sv_log, sel_sv_cmap, align="end"),
            pn.Row(fi_sv_vmin, fi_sv_vmax, btn_sv_auto, align="end"),
            _sv_fig,
            sv_coord_info,
            sv_info,
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML("<p class='section-label'>White Field</p>",
                         sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Compute a white-field image from this scan "
                "(mean/max/median across all frames of all files). "
                "The result is stored globally and can be subtracted from any ROI "
                "via the <b>Subtract white field</b> pipeline step.</span>",
                sizing_mode="stretch_width"),
            pn.Row(w_wf_label, w_wf_stat, align="end"),
            pn.Row(btn_wf_start, btn_wf_stop, btn_wf_delete, align="center"),
            wf_progress,
            wf_status,
            wf_preview_pane,
            sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
        ),
        pn.Column(
            pn.pane.HTML("<p class='section-label'>ROI Mask Builder</p>",
                         sizing_mode="stretch_width"),
            pn.pane.HTML(
                "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                "color:var(--txt-muted)'>Combine rectangles, annuli, and disks into one ROI. "
                "Each shape is OR-ed together. "
                "The bounding box is computed automatically.</span>",
                sizing_mode="stretch_width"),
            pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                         "font-family:JetBrains Mono,monospace;margin:10px 0 4px 0'>"
                         "SHAPE TYPE</p>", sizing_mode="stretch_width"),
            pn.Row(w_shape_type),
            param_panel,
            pn.Row(btn_add_shape, btn_apply_to_frame, btn_clear_shapes, align="center"),
            pn.Row(cb_invert, align="center"),
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                         "font-family:JetBrains Mono,monospace;margin:0 0 6px 0'>"
                         "SHAPES</p>", sizing_mode="stretch_width"),
            shapes_col,
            mask_preview_pane,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                         "font-family:JetBrains Mono,monospace;margin:0 0 6px 0'>"
                         "SAVE / LOAD MASK</p>", sizing_mode="stretch_width"),
            w_mask_path,
            pn.Row(btn_save_mask, btn_load_mask),
            mask_io_status,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                         "font-family:JetBrains Mono,monospace;margin:0 0 6px 0'>"
                         "MASK LIBRARY</p>", sizing_mode="stretch_width"),
            pn.Row(w_lib_save_name, btn_lib_save, align="end"),
            w_lib_select,
            pn.Row(btn_lib_load, btn_lib_delete),
            lib_status,
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
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
        "pos_axes":    pn.widgets.Checkbox(  name="position axes (µm)", value=False),
        "vmin":        pn.widgets.FloatInput(name="vmin (blank=auto)", width=130),
        "vmax":        pn.widgets.FloatInput(name="vmax (blank=auto)", width=130),
        "4d_px":       pn.widgets.IntInput(  name="detector row (px)", value=0, width=150),
        "4d_py":       pn.widgets.IntInput(  name="detector col (py)", value=0, width=150),
        "save_dir":    pn.widgets.TextInput( name="output directory", placeholder="/path/to/out/", sizing_mode="stretch_width"),
        "save_name":   pn.widgets.TextInput( name="file name (no ext)", placeholder="export", width=200),
        "ref_roi":     pn.widgets.Select(   name="reference ROI",
                                            options=list(state["rois"].keys()) or ["— none —"],
                                            value=list(state["rois"].keys())[0] if state["rois"] else "— none —",
                                            width=200),
        "white_field": pn.widgets.Select(   name="white field",
                                            options=list(state["white_fields"].keys()) or ["— none —"],
                                            value=list(state["white_fields"].keys())[0] if state["white_fields"] else "— none —",
                                            width=220),
    }
    _roi_ref_selects.append(pw["ref_roi"])
    _white_field_selects.append(pw["white_field"])

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
            _coh_plot_ops = ("Plot: Averaged coherent", "Plot: Summed coherent", "Plot: Coherent sequence")
            if op in _coh_plot_ops:
                p["pos_axes"] = pw["pos_axes"].value
            if op == "Plot: 4D viewer (single)":
                p.update(px=pw["4d_px"].value, py=pw["4d_py"].value)
        elif op.startswith("Save:"):
            p.update(save_dir=pw["save_dir"].value.strip(),
                     save_name=pw["save_name"].value.strip() or "ptyxplore_export")
        elif op == "Normalise detector":
            p["ref_roi_name"] = pw["ref_roi"].value
        elif op == "Subtract white field":
            p["white_field_label"] = pw["white_field"].value
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
            _coh_plot_ops = ("Plot: Averaged coherent", "Plot: Summed coherent", "Plot: Coherent sequence")
            if op in _coh_plot_ops:
                params_col.append(pn.Row(pw["cmap"], pw["log_scale"], pw["pos_axes"], pw["vmin"], pw["vmax"]))
            else:
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
        elif op == "Normalise detector":
            params_col.append(pw["ref_roi"])
        elif op == "Subtract white field":
            params_col.append(pw["white_field"])

    w_op.param.watch(refresh_params, "value")
    refresh_params()

    # ── Per-ROI queue buttons and callbacks ───────────────────────────────────
    btn_add   = pn.widgets.Button(name="Add to Queue", button_type="primary", width=160)
    btn_run   = pn.widgets.Button(name="▶  Run All",   button_type="primary", width=150)
    btn_stop  = pn.widgets.Button(name="⏹  Stop",      button_type="danger",  width=120, disabled=True)
    btn_clear = pn.widgets.Button(name="Clear Queue",  button_type="default", width=130)
    btn_stxm  = pn.widgets.Button(name="Make STXM",   button_type="success", width=130)

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

    def on_stxm(event):
        queue.append({"op": "Make 4D dataset", "roi_name": roi_name,
                      "params": {"num_jobs": 96}, "status": "pending", "enabled": True})
        queue.append({"op": "Plot: Summed coherent", "roi_name": roi_name,
                      "params": {"cmap": "viridis", "log_scale": False,
                                 "vmin": None, "vmax": None, "pos_axes": False},
                      "status": "pending", "enabled": True})
        _rebuild_queue(queue, queue_col_r)
        log(f"[{roi_name}] STXM steps queued (Make 4D dataset × 96 + Plot: Summed coherent)", "info")
        _run_queue(queue, queue_col_r, roi_name, btn_run, btn_stop, btn_clear)

    btn_add.on_click(on_add)
    btn_run.on_click(on_run)
    btn_stop.on_click(on_stop)
    btn_clear.on_click(on_clear)
    btn_stxm.on_click(on_stxm)
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
            pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
            pn.Row(btn_stxm),
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
    cb_pos       = pn.widgets.Checkbox(name="Position axes (µm)", value=False)

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
        pp = _coh_pos_params(roi, hc, wc) if cb_pos.value else None
        _coh_map.palette = _bk_palette(w_cmap_coh.value)
        lo_c = fi_coh_vmin.value if fi_coh_vmin.value is not None else float(np.nanmin(coh))
        hi_c = fi_coh_vmax.value if fi_coh_vmax.value is not None else float(np.nanmax(coh))
        if abs(hi_c - lo_c) < 1e-10: hi_c = lo_c + 1.0
        _coh_map.low = lo_c;  _coh_map.high = hi_c
        if pp is not None:
            _coh_src.data = {'image': [coh],
                             'x': [pp["img_x"]], 'y': [pp["img_y"]],
                             'dw': [pp["img_dw"]], 'dh': [pp["img_dh"]]}
            _coh_mark.data = {'x': [pp["px_to_nm_col"](lcol)],
                              'y': [pp["px_to_nm_row"](lrow)],
                              'w': [pp["col_nm_size"](2)],
                              'h': [pp["row_nm_size"](2)]}
            _coh_fig.xaxis.axis_label = pp["xlabel"]
            _coh_fig.yaxis.axis_label = pp["ylabel"]
        else:
            _coh_src.data = {'image': [coh], 'x': [0], 'y': [0], 'dw': [wc], 'dh': [hc]}
            _coh_mark.data = {'x': [lcol + 0.5], 'y': [lrow + 0.5], 'w': [2.0], 'h': [2.0]}
            _coh_fig.xaxis.axis_label = ""
            _coh_fig.yaxis.axis_label = ""
        _coh_fig.title.text = f"Coherent  det=({pcol}, {prow})"

        # Update det source + colormap
        # Apply mask: set pixels outside the ROI mask to NaN so they render
        # as the figure nan_color (black), making the active region clear.
        det_disp = det.copy().astype(float)
        if getattr(roi, "detector_mask", None) is not None:
            det_disp[roi.detector_mask == 0] = np.nan
        _det_map.palette = _bk_palette(w_cmap_det.value)
        lo_d = fi_det_vmin.value if fi_det_vmin.value is not None else float(np.nanmin(det_disp))
        hi_d = fi_det_vmax.value if fi_det_vmax.value is not None else float(np.nanmax(det_disp))
        if abs(hi_d - lo_d) < 1e-10: hi_d = lo_d + 1.0
        _det_map.low = lo_d;  _det_map.high = hi_d
        _det_src.data = {'image': [det_disp], 'x': [0], 'y': [0], 'dw': [wd], 'dh': [hd]}
        _det_mark.data = {'x': [pcol + 0.5], 'y': [prow + 0.5], 'w': [4.0], 'h': [4.0]}
        _det_fig.title.text = f"Detector  scan=({lcol}, {lrow})"

    def init_viewer():
        roi = state["rois"].get(roi_name)
        if roi and roi.data_4d is not None:
            dsh, csh = roi.data_4d.shape[2:], roi.data_4d.shape[:2]
            def_prow, def_pcol = _mask_default_pixel(roi)
            def _init():
                w_prow.end = dsh[0] - 1;  w_prow.value = def_prow
                w_pcol.end = dsh[1] - 1;  w_pcol.value = def_pcol
                w_lrow.end = csh[0] - 1;  w_lrow.value = csh[0] // 2
                w_lcol.end = csh[1] - 1;  w_lcol.value = csh[1] // 2
                update_viewer()
            pn.state.execute(_init)
        else:
            pn.state.execute(update_viewer)

    roi_4dv_hooks[roi_name] = init_viewer  # called by _execute_step after Make 4D dataset

    for _w in (w_prow, w_pcol, w_lrow, w_lcol, w_cmap_coh, w_cmap_det, cb_pos):
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
                    pn.Row(w_cmap_coh, fi_coh_vmin, fi_coh_vmax, btn_coh_auto, cb_pos, align="end"),
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
#  TAB — STITCHER
# ═══════════════════════════════════════════════════════════════════════════════

_stitch_items: list[str] = []   # ordered list of roi_names queued for stitching
_stitch_result: dict = {}       # keys: 'img', 'x_axis', 'y_axis' after a successful stitch

# ── ROI selector & list ───────────────────────────────────────────────────────
sel_stitch_roi  = pn.widgets.Select(
    name="ROI", options=["— none —"], value="— none —", width=260)
btn_stitch_add  = pn.widgets.Button(name="+ Add", button_type="primary", width=90)
stitch_list_col = pn.Column(sizing_mode="stretch_width")

def _refresh_stitch_select():
    names = ["— none —"] + list(state["rois"].keys())
    sel_stitch_roi.options = names
    if sel_stitch_roi.value not in names:
        sel_stitch_roi.value = names[0]

def _rebuild_stitch_list():
    stitch_list_col.clear()
    if not _stitch_items:
        stitch_list_col.append(pn.pane.HTML(
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--txt-muted)'>No ROIs added yet.</span>"))
        return
    for name in list(_stitch_items):
        roi = state["rois"].get(name)
        has_img = roi is not None and roi.averaged_coherent_images is not None
        shape_str = str(roi.averaged_coherent_images.shape) if has_img else "no image"
        flag = "" if has_img else " ⚠ no avg image"
        btn_rm = pn.widgets.Button(name="✕", button_type="danger", width=36, height=28)
        def _rm(e, n=name):
            if n in _stitch_items:
                _stitch_items.remove(n)
            _rebuild_stitch_list()
        btn_rm.on_click(_rm)
        stitch_list_col.append(pn.Row(
            btn_rm,
            pn.pane.HTML(
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--txt-base)'>{name}</span>"
                f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;"
                f"color:var(--txt-muted)'> {shape_str}{flag}</span>"),
            align="center"))

def _on_stitch_add(event):
    name = sel_stitch_roi.value
    if name and name != "— none —" and name not in _stitch_items:
        _stitch_items.append(name)
        _rebuild_stitch_list()

btn_stitch_add.on_click(_on_stitch_add)

# ── options ───────────────────────────────────────────────────────────────────
fi_stitch_psize  = pn.widgets.FloatInput(
    name="Pixel size (µm, blank=auto)", width=200)
sel_stitch_mode  = pn.widgets.RadioButtonGroup(
    options=["mean", "max", "sum"], value="mean", button_type="default", width=220)
sel_stitch_cmap  = pn.widgets.Select(
    name="Colormap", options=["viridis","jet","plasma","inferno","gray","hot","turbo"],
    value="viridis", width=130)
cb_stitch_log    = pn.widgets.Checkbox(name="Log scale", value=False)
fi_stitch_vmin   = pn.widgets.FloatInput(name="vmin (blank=auto)", width=130)
fi_stitch_vmax   = pn.widgets.FloatInput(name="vmax (blank=auto)", width=130)
btn_stitch_auto  = pn.widgets.Button(name="Autoscale", button_type="default", width=100)

# ── run & status ──────────────────────────────────────────────────────────────
btn_stitch_run   = pn.widgets.Button(name="▶ Stitch", button_type="primary", width=120)
stitch_status    = pn.pane.HTML(
    "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
    "color:var(--txt-muted)'>Add ROIs above then press Stitch.</span>",
    sizing_mode="stretch_width")

# ── result Bokeh figure ───────────────────────────────────────────────────────
from bokeh.models import (
    ColumnDataSource, LinearColorMapper, ColorBar,
    HoverTool, BasicTicker,
)
from bokeh.plotting import figure as bk_figure

_st_blank  = np.zeros((64, 64), dtype=np.float32)
_st_src    = ColumnDataSource({"image": [_st_blank], "x": [0], "y": [0], "dw": [64], "dh": [64]})
_st_map    = LinearColorMapper(palette=_bk_palette("viridis"), low=0, high=1, nan_color="black")
_st_fig    = bk_figure(
    sizing_mode="stretch_width", height=520,
    tools="pan,wheel_zoom,reset,save", active_scroll="wheel_zoom",
    match_aspect=True,
    background_fill_color=BG_DARK, border_fill_color=BG_MID,
)
_st_fig.y_range.flipped = True
_st_fig.image("image", "x", "y", "dw", "dh", source=_st_src, color_mapper=_st_map)
_st_fig.add_tools(HoverTool(tooltips=[("X (µm)", "$x{0.000}"), ("Y (µm)", "$y{0.000}"), ("val", "@image{0.00}")]))
_st_fig.add_layout(ColorBar(color_mapper=_st_map, ticker=BasicTicker(), location=(0, 0)), "right")
_st_fig.xaxis.axis_label = "X (µm)"
_st_fig.yaxis.axis_label = "Y (µm)"
stitch_info = pn.pane.HTML("", sizing_mode="stretch_width")

def _st_refresh_display():
    """Re-render the stitched image with current scale / cmap settings."""
    img = _stitch_result.get("img")
    if img is None:
        return
    x_axis = _stitch_result["x_axis"]
    y_axis = _stitch_result["y_axis"]
    disp = np.log1p(np.abs(img.astype(float))) if cb_stitch_log.value else img.astype(float)
    lo = fi_stitch_vmin.value if fi_stitch_vmin.value is not None else float(np.nanmin(disp))
    hi = fi_stitch_vmax.value if fi_stitch_vmax.value is not None else float(np.nanmax(disp))
    if abs(hi - lo) < 1e-10:
        hi = lo + 1.0
    x0, y0 = float(x_axis[0]), float(y_axis[0])
    dw = float(x_axis[-1] - x_axis[0])
    dh = float(y_axis[-1] - y_axis[0])
    _st_map.palette = _bk_palette(sel_stitch_cmap.value)
    _st_map.low = lo
    _st_map.high = hi
    _st_src.data = {"image": [disp], "x": [x0], "y": [y0], "dw": [dw], "dh": [dh]}

def _on_stitch_run(event):
    if not _stitch_items:
        stitch_status.object = (
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--warn)'>No ROIs in the stitch list.</span>")
        return
    rois = []
    for name in _stitch_items:
        roi = state["rois"].get(name)
        if roi is None:
            stitch_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--warn)'>ROI '{name}' not found.</span>")
            return
        if roi.averaged_coherent_images is None:
            stitch_status.object = (
                f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
                f"color:var(--warn)'>ROI '{name}' has no averaged coherent images. "
                f"Run Make coherent images + Align/Filter first.</span>")
            return
        rois.append(roi)
    try:
        from ptyxplore import scan_stitcher
        psize = fi_stitch_psize.value * 1e3 if fi_stitch_psize.value else None  # µm → nm for API
        stitched, x_axis_nm, y_axis_nm = scan_stitcher.stitch(
            rois, pixel_size=psize, mode=sel_stitch_mode.value)
        x_axis = x_axis_nm / 1e3   # nm → µm for display
        y_axis = y_axis_nm / 1e3
        _stitch_result.clear()
        _stitch_result.update({"img": stitched, "x_axis": x_axis, "y_axis": y_axis})
        _st_refresh_display()
        n_valid = int(np.sum(~np.isnan(stitched)))
        stitch_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--accent)'>✓ Stitched {len(rois)} ROIs — "
            f"output shape {stitched.shape}, {n_valid} filled pixels, "
            f"X:[{x_axis[0]:.3f}, {x_axis[-1]:.3f}] µm  "
            f"Y:[{y_axis[0]:.3f}, {y_axis[-1]:.3f}] µm</span>")
        log(f"Stitch complete: {len(rois)} ROIs → shape {stitched.shape}", "ok")
        stitch_info.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--txt-muted)'>"
            f"X: [{x_axis[0]:.3f}, {x_axis[-1]:.3f}] µm  ·  "
            f"Y: [{y_axis[0]:.3f}, {y_axis[-1]:.3f}] µm  ·  "
            f"pixel size: {x_axis[1]-x_axis[0]:.4f} µm  ·  "
            f"shape: {stitched.shape}</span>")
    except Exception as exc:
        import traceback as _tb
        stitch_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--warn)'>Error: {exc}</span>")
        log(f"Stitch failed: {exc}", "err")

btn_stitch_run.on_click(_on_stitch_run)

def _on_stitch_auto(event):
    fi_stitch_vmin.value = fi_stitch_vmax.value = None
    _st_refresh_display()

btn_stitch_auto.on_click(_on_stitch_auto)
for _sw in (fi_stitch_vmin, fi_stitch_vmax, sel_stitch_cmap, cb_stitch_log):
    _sw.param.watch(lambda e: _st_refresh_display(), "value")

# ── save ──────────────────────────────────────────────────────────────────────
w_stitch_save_path = pn.widgets.TextInput(
    name="Save path (.h5)", placeholder="/path/to/stitched.h5",
    sizing_mode="stretch_width")
btn_stitch_save = pn.widgets.Button(name="Save HDF5", button_type="default", width=120)
stitch_save_status = pn.pane.HTML("", sizing_mode="stretch_width")

def _on_stitch_save(event):
    if not _stitch_result:
        stitch_save_status.object = (
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--warn)'>No result to save — run Stitch first.</span>")
        return
    path = w_stitch_save_path.value.strip()
    if not path:
        stitch_save_status.object = (
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--warn)'>Enter a file path.</span>")
        return
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset("stitched", data=_stitch_result["img"], compression="gzip")
            f.create_dataset("x_axis_nm", data=_stitch_result["x_axis"])
            f.create_dataset("y_axis_nm", data=_stitch_result["y_axis"])
            f["stitched"].attrs["rois"] = str(_stitch_items)
            f["stitched"].attrs["mode"] = sel_stitch_mode.value
        stitch_save_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--accent)'>✓ Saved to {path}</span>")
        log(f"Stitched image saved → {path}", "ok")
    except Exception as exc:
        stitch_save_status.object = (
            f"<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            f"color:var(--warn)'>Save failed: {exc}</span>")

btn_stitch_save.on_click(_on_stitch_save)

# ── tab layout ────────────────────────────────────────────────────────────────
tab_stitcher = pn.Column(
    pn.Column(
        pn.pane.HTML("<p class='section-label'>ROI Selection</p>", sizing_mode="stretch_width"),
        pn.pane.HTML(
            "<span style='font-family:JetBrains Mono,monospace;font-size:11px;"
            "color:var(--txt-muted)'>Add ROIs whose <b>averaged_coherent_images</b> "
            "should be stitched. Positions are read from scan motor coordinates (nm).</span>",
            sizing_mode="stretch_width"),
        pn.Row(sel_stitch_roi, btn_stitch_add, align="end"),
        pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                     "font-family:JetBrains Mono,monospace;margin:8px 0 4px 0'>"
                     "STITCH LIST</p>", sizing_mode="stretch_width"),
        stitch_list_col,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Options</p>", sizing_mode="stretch_width"),
        pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                     "font-family:JetBrains Mono,monospace;margin:4px 0'>OVERLAP MODE</p>",
                     sizing_mode="stretch_width"),
        pn.Row(sel_stitch_mode),
        fi_stitch_psize,
        pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
        pn.Row(btn_stitch_run),
        stitch_status,
        sizing_mode="stretch_width", css_classes=["ptycho-card"], margin=(0, 0, 16, 0),
    ),
    pn.Column(
        pn.pane.HTML("<p class='section-label'>Result</p>", sizing_mode="stretch_width"),
        pn.Row(sel_stitch_cmap, cb_stitch_log, fi_stitch_vmin, fi_stitch_vmax,
               btn_stitch_auto, align="end"),
        _st_fig,
        stitch_info,
        pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
        pn.pane.HTML("<p style='font-size:11px;color:var(--txt-muted);"
                     "font-family:JetBrains Mono,monospace;margin:4px 0'>SAVE</p>",
                     sizing_mode="stretch_width"),
        pn.Row(w_stitch_save_path, btn_stitch_save),
        stitch_save_status,
        sizing_mode="stretch_width", css_classes=["ptycho-card"],
    ),
    sizing_mode="stretch_width", margin=(16, 0),
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Tabs + main dashboard
# ═══════════════════════════════════════════════════════════════════════════════

tabs = pn.Tabs(
    ("Scan Loading",     tab_loading),
    ("Viewer",           tab_vis),
    ("ROI Manager",      tab_roi_manager),
    ("Stitcher",         tab_stitcher),
    ("Python",           tab_python),
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
