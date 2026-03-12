"""
Ptychography GUI Dashboard — Experiment Setup
=============================================
Run in JupyterHub:
    import panel as pn
    from ptycho_dashboard import dashboard
    pn.extension()
    dashboard.servable()

Or serve standalone on Maxwell:
    panel serve ptycho_dashboard.py --show
"""

import panel as pn
import param
import numpy as np
from datetime import datetime

pn.extension(
    "tabulator",
    sizing_mode="stretch_width",
    notifications=True,
)

# ── Colour tokens ──────────────────────────────────────────────────────────────
ACCENT   = "#0FCFAF"   # teal-green
ACCENT2  = "#845EC2"   # purple
WARN     = "#FF6B6B"   # red-ish
BG_DARK  = "#0D1117"
BG_MID   = "#161B22"
BG_CARD  = "#1C2333"
BORDER   = "#30363D"
TXT      = "#E6EDF3"
TXT_MUTED= "#8B949E"

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

/* ── Cards ── */
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

/* ── Header ── */
.dash-header {{
  background: linear-gradient(135deg, {BG_MID} 0%, #0A1628 100%);
  border-bottom: 1px solid var(--border);
  padding: 20px 32px;
  display: flex;
  align-items: center;
  gap: 16px;
}}
.dash-title {{
  font-size: 22px;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}}
.dash-sub {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--txt-muted);
  margin: 2px 0 0 0;
}}

/* ── Section labels ── */
.section-label {{
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 14px 0;
  font-family: 'JetBrains Mono', monospace;
}}

/* ── Status badge ── */
.badge {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  letter-spacing: 0.5px;
}}
.badge-ok  {{ background: rgba(0,201,167,0.15); color: var(--accent); border: 1px solid rgba(0,201,167,0.3); }}
.badge-err {{ background: rgba(255,107,107,0.15); color: var(--warn);   border: 1px solid rgba(255,107,107,0.3); }}
.badge-idle{{ background: rgba(139,148,158,0.15); color: var(--txt-muted); border: 1px solid rgba(139,148,158,0.3); }}

/* ── Stat chips ── */
.stat-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 12px;
}}
.stat-chip {{
  background: var(--bg-mid);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
}}
.stat-chip .val {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 600;
  color: var(--accent);
  line-height: 1;
}}
.stat-chip .lbl {{
  font-size: 11px;
  color: var(--txt-muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 1px;
}}

/* ── Log panel ── */
.log-box {{
  background: var(--bg-dark);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  line-height: 1.7;
  color: var(--txt-muted);
  max-height: 280px;
  overflow-y: auto;
}}
.log-ok   {{ color: {ACCENT}; }}
.log-err  {{ color: {WARN}; }}
.log-info {{ color: #79C0FF; }}
.log-warn {{ color: #E3B341; }}

/* ── Widget overrides ── */
.bk-input, .bk-select {{
  background: var(--bg-mid) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt) !important;
  border-radius: 6px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 13px !important;
}}
.bk-input:focus {{ border-color: var(--accent) !important; outline: none !important; }}

.bk-btn-primary {{
  background: linear-gradient(135deg, var(--accent), #00A88A) !important;
  border: none !important;
  color: #000 !important;
  font-weight: 700 !important;
  font-family: 'Syne', sans-serif !important;
  letter-spacing: 0.5px !important;
  border-radius: 8px !important;
  transition: transform 0.15s, box-shadow 0.15s !important;
}}
.bk-btn-primary:hover {{
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 20px rgba(0,201,167,0.35) !important;
}}
.bk-btn-default {{
  background: var(--bg-mid) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt) !important;
  font-family: 'Syne', sans-serif !important;
  border-radius: 8px !important;
}}

.bk-widget-box {{ background: transparent !important; }}
label.bk {{ color: var(--txt-muted) !important; font-size: 12px !important; letter-spacing: 0.3px; }}

/* divider */
.divider {{
  border: none;
  border-top: 1px solid var(--border);
  margin: 18px 0;
}}
"""

pn.config.raw_css.append(CSS)


# ═══════════════════════════════════════════════════════════════════════════════
#  State helpers
# ═══════════════════════════════════════════════════════════════════════════════

log_lines: list[str] = []

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log(msg: str, kind: str = "info") -> None:
    """kind: info | ok | err | warn"""
    log_lines.append(f'<span class="log-{kind}">[{ts()}] {msg}</span>')
    _refresh_log()

def _refresh_log():
    log_pane.object = '<br>'.join(log_lines[-80:])   # keep last 80 lines


# ═══════════════════════════════════════════════════════════════════════════════
#  Widgets — Experiment Block
# ═══════════════════════════════════════════════════════════════════════════════

w_year      = pn.widgets.IntInput(
    name="Year", value=datetime.now().year, step=1,
    width=120,
)
w_btid      = pn.widgets.TextInput(
    name="Beamtime ID", placeholder="e.g. 11018232", width=200,
)
w_beamline  = pn.widgets.Select(
    name="Beamline", options=["p11", "p06", "p10", "p23"], value="p11", width=140,
)
btn_create_exp = pn.widgets.Button(
    name="Create Experiment", button_type="primary", width=200,
)

# derived path preview
exp_path_md = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;font-size:12px'>"
    "Fill fields above to preview path…</span>",
    sizing_mode="stretch_width",
)

# ── live path preview ──────────────────────────────────────────────────────────
def _update_path_preview(event=None):
    y  = w_year.value or "YYYY"
    bt = w_btid.value.strip() or "<beamtime_id>"
    bl = w_beamline.value
    path = f"/asap3/petra3/gpfs/{bl}/{y}/data/{bt}/raw"
    exp_path_md.object = (
        f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;"
        f"color:var(--txt-muted)'>"
        f"<span style='color:var(--accent)'>scan_frames:</span> {path}/scan_frames<br>"
        f"<span style='color:var(--accent)'>logs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:</span> {path}/server_log/Scan_logs"
        f"</div>"
    )

for w in [w_year, w_btid, w_beamline]:
    w.param.watch(_update_path_preview, "value")

_update_path_preview()


# ═══════════════════════════════════════════════════════════════════════════════
#  Widgets — Scan Block
# ═══════════════════════════════════════════════════════════════════════════════

w_scan_num   = pn.widgets.IntInput(
    name="Scan Number", value=1, step=1, width=140,
)
w_detector   = pn.widgets.Select(
    name="Detector", options=["Eiger", "Lambda"], value="Eiger", width=140,
)
w_sample     = pn.widgets.TextInput(
    name="Sample Name", placeholder="e.g. SampleA_001", width=220,
)
w_det_dist   = pn.widgets.FloatInput(
    name="Detector Distance (m)", value=2.0, step=0.05, width=200,
)
btn_create_scan = pn.widgets.Button(
    name="Create Scan", button_type="primary", width=180,
    disabled=True,   # enabled once Exp exists
)
btn_clear_scan  = pn.widgets.Button(
    name="Clear", button_type="default", width=100,
)

# ── Scan info display ─────────────────────────────────────────────────────────
scan_info_html = pn.pane.HTML(
    "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;"
    "font-size:12px'>No scan loaded yet.</span>",
    sizing_mode="stretch_width",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Widgets — File List Block
# ═══════════════════════════════════════════════════════════════════════════════

file_table = pn.widgets.Tabulator(
    value=None,
    pagination="local",
    page_size=12,
    sizing_mode="stretch_width",
    height=260,
    theme="midnight",
    stylesheets=[
        """:host .tabulator { background: var(--bg-dark); color: var(--txt);
           font-family: 'JetBrains Mono', monospace; font-size: 12px; }"""
    ],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Widgets — Status bar + Log
# ═══════════════════════════════════════════════════════════════════════════════

status_badge = pn.pane.HTML(
    "<span class='badge badge-idle'>IDLE</span>",
    width=120,
)

log_pane = pn.pane.HTML(
    "<span class='log-info'>[--:--:--] Dashboard initialised.</span>",
    sizing_mode="stretch_width",
)

# ── stat chips ────────────────────────────────────────────────────────────────
stats_html = pn.pane.HTML(
    "<div class='stat-grid'>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Scan #</div></div>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Files</div></div>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Energy (keV)</div></div>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Step X (nm)</div></div>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Step Y (nm)</div></div>"
    "<div class='stat-chip'><div class='val'>—</div><div class='lbl'>Points X×Y</div></div>"
    "</div>",
    sizing_mode="stretch_width",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  State holders  (real objects go here once created)
# ═══════════════════════════════════════════════════════════════════════════════
state = {"exp": None, "scan": None}


# ═══════════════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

def _set_status(ok: bool | None):
    if ok is None:
        status_badge.object = "<span class='badge badge-idle'>IDLE</span>"
    elif ok:
        status_badge.object = "<span class='badge badge-ok'>READY</span>"
    else:
        status_badge.object = "<span class='badge badge-err'>ERROR</span>"


def on_create_exp(event):
    year  = w_year.value
    btid  = w_btid.value.strip()
    bl    = w_beamline.value

    if not btid:
        log("Beamtime ID is required.", "err")
        _set_status(False)
        return

    log(f"Creating Exp(year={year}, beamtime_id='{btid}', beamline='{bl}')…", "info")
    try:
        from explorer.cbd_loader import Exp          # ← uncomment when integrated
        state["exp"] = Exp(year=year, beamtime_id=btid, beamline=bl)
        
        # ─────────────────────────────────────────────────────────────────────

        log(f"Exp created ✓  scan_frames → {state['exp'].scan_frames_dir}", "ok")
        _set_status(True)
        btn_create_scan.disabled = False
        pn.state.notifications.success("Experiment created!", duration=3000)

    except Exception as exc:
        log(f"Failed to create Exp: {exc}", "err")
        _set_status(False)
        pn.state.notifications.error(str(exc), duration=5000)


def _update_stats(scan):
    """Render stat chips from a Scan-like object."""
    def _v(attr, fmt="{}", fallback="—"):
        v = getattr(scan, attr, None)
        return fmt.format(v) if v is not None else fallback

    n_files = len(getattr(scan, "file_list", []) or [])
    energy  = getattr(scan, "energy", None)
    e_str   = f"{energy*1e-3:.4f}" if energy else "—"   # eV → keV

    sx = getattr(scan, "step_size_x", None)
    sy = getattr(scan, "step_size_y", None)
    sx_str = f"{sx*1e9:.2f}" if sx else "—"   # m → nm
    sy_str = f"{sy*1e9:.2f}" if sy else "—"

    px = getattr(scan, "scan_x_steps_count", None)
    py = getattr(scan, "scan_y_steps_count", None)
    pts_str = f"{px}×{py}" if (px and py) else "—"

    stats_html.object = (
        "<div class='stat-grid'>"
        f"<div class='stat-chip'><div class='val'>{scan.scan_num}</div><div class='lbl'>Scan #</div></div>"
        f"<div class='stat-chip'><div class='val'>{n_files}</div><div class='lbl'>Files</div></div>"
        f"<div class='stat-chip'><div class='val'>{e_str}</div><div class='lbl'>Energy (keV)</div></div>"
        f"<div class='stat-chip'><div class='val'>{sx_str}</div><div class='lbl'>Step X (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{sy_str}</div><div class='lbl'>Step Y (nm)</div></div>"
        f"<div class='stat-chip'><div class='val'>{pts_str}</div><div class='lbl'>Points X×Y</div></div>"
        "</div>"
    )


def on_create_scan(event):
    if state["exp"] is None:
        log("Create an Experiment first.", "warn")
        return

    snum   = w_scan_num.value
    det    = w_detector.value
    sample = w_sample.value.strip() or None
    ddist  = w_det_dist.value

    log(f"Loading Scan {snum} (det={det}, dist={ddist} m)…", "info")
    try:
        from cbd_loader import Scan
        state["scan"] = Scan(
             exp=state["exp"], scan_num=snum,
             detector=det, sample_name=sample, det_distance=ddist,
         )

        scan = state["scan"]
        log(f"Scan {snum} loaded ✓  ({len(scan.file_list)} files found)", "ok")
        _set_status(True)

        # ── info pane ─────────────────────────────────────────────────────────
        scan_info_html.object = (
            f"<div style='font-family:JetBrains Mono,monospace;font-size:12px;line-height:1.9'>"
            f"<span style='color:var(--txt-muted)'>data_dir &nbsp;&nbsp;:</span> "
            f"<span style='color:var(--txt)'>{scan.data_dir}</span><br>"
            f"<span style='color:var(--txt-muted)'>slow_axis  :</span> "
            f"<span style='color:{ACCENT2}'>{scan.slow_axis}</span><br>"
            f"<span style='color:var(--txt-muted)'>wavelength :</span> "
            f"<span style='color:{ACCENT}'>{scan.wavelength*1e10:.4f} Å</span>&nbsp;&nbsp;"
            f"<span style='color:var(--txt-muted)'>energy :</span> "
            f"<span style='color:{ACCENT}'>{scan.energy*1e-3:.4f} keV</span><br>"
            f"<span style='color:var(--txt-muted)'>det psize  :</span> "
            f"<span style='color:var(--txt)'>{scan.det_psize*1e6:.0f} µm</span>"
            f"</div>"
        )

        # ── file table ────────────────────────────────────────────────────────
        import pandas as pd
        rows = [{"#": i+1, "File": f} for i, f in enumerate(scan.file_list)]
        file_table.value = pd.DataFrame(rows)

        _update_stats(scan)
        pn.state.notifications.success(f"Scan {snum} loaded!", duration=3000)

    except Exception as exc:
        log(f"Failed to create Scan: {exc}", "err")
        _set_status(False)
        pn.state.notifications.error(str(exc), duration=5000)


def on_clear_scan(event):
    state["scan"] = None
    scan_info_html.object = (
        "<span style='color:var(--txt-muted);font-family:JetBrains Mono,monospace;"
        "font-size:12px'>No scan loaded yet.</span>"
    )
    file_table.value = None
    stats_html.object = (
        "<div class='stat-grid'>"
        + "".join(
            f"<div class='stat-chip'><div class='val'>—</div><div class='lbl'>{l}</div></div>"
            for l in ["Scan #","Files","Energy (keV)","Step X (nm)","Step Y (nm)","Points X×Y"]
        )
        + "</div>"
    )
    _set_status(None)
    log("Scan cleared.", "warn")


btn_create_exp.on_click(on_create_exp)
btn_create_scan.on_click(on_create_scan)
btn_clear_scan.on_click(on_clear_scan)


# ═══════════════════════════════════════════════════════════════════════════════
#  Layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _card(title: str, *children):
    return pn.pane.HTML(
        f"<div class='ptycho-card'>"
        f"<p class='section-label'>{title}</p>",
        sizing_mode="stretch_width",
        height=0,   # placeholder — real content via Column below
    )


def _section(label: str, *body_items) -> pn.Column:
    header = pn.pane.HTML(
        f"<div class='ptycho-card' style='padding-bottom:0'>"
        f"<p class='section-label'>{label}</p>"
        f"</div>",
        sizing_mode="stretch_width",
    )
    return pn.Column(header, *body_items, sizing_mode="stretch_width")


# ── Header ────────────────────────────────────────────────────────────────────
header = pn.pane.HTML(
    f"""
    <div class="dash-header">
      <div>
        <p class="dash-title">⬡ PTYCHO DASHBOARD</p>
        <p class="dash-sub">CID · Beamline P11 · PETRA III · DESY</p>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:12px">
        <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--txt-muted)">status</span>
    """,
    sizing_mode="stretch_width",
    height=72,
)

# ─── Panel cards as Column groups ─────────────────────────────────────────────

# Card 1 — Experiment setup
card_exp = pn.Column(
    pn.pane.HTML("<div class='ptycho-card' style='padding-bottom:8px'><p class='section-label'>01 — Experiment</p></div>",
                 sizing_mode="stretch_width"),
    pn.Row(w_year, w_btid, w_beamline, align="end"),
    exp_path_md,
    pn.Row(btn_create_exp, status_badge, align="center"),
    pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    css_classes=["ptycho-card"],
    margin=(0, 0, 16, 0),
)

# Card 2 — Scan setup
card_scan = pn.Column(
    pn.pane.HTML("<p class='section-label'>02 — Scan</p>", sizing_mode="stretch_width"),
    pn.Row(w_scan_num, w_detector, w_sample, w_det_dist, align="end"),
    pn.Row(btn_create_scan, btn_clear_scan, align="center"),
    pn.pane.HTML("<hr class='divider'>", sizing_mode="stretch_width"),
    scan_info_html,
    sizing_mode="stretch_width",
    css_classes=["ptycho-card"],
    margin=(0, 0, 16, 0),
)

# Card 3 — Stats
card_stats = pn.Column(
    pn.pane.HTML("<p class='section-label'>03 — Scan Summary</p>", sizing_mode="stretch_width"),
    stats_html,
    sizing_mode="stretch_width",
    css_classes=["ptycho-card"],
    margin=(0, 0, 16, 0),
)

# Card 4 — File list
card_files = pn.Column(
    pn.pane.HTML("<p class='section-label'>04 — Detector Files</p>", sizing_mode="stretch_width"),
    file_table,
    sizing_mode="stretch_width",
    css_classes=["ptycho-card"],
    margin=(0, 0, 16, 0),
)

# Card 5 — Log
card_log = pn.Column(
    pn.pane.HTML("<p class='section-label'>05 — Activity Log</p>", sizing_mode="stretch_width"),
    pn.Column(log_pane, css_classes=["log-box"], sizing_mode="stretch_width"),
    sizing_mode="stretch_width",
    css_classes=["ptycho-card"],
    margin=(0, 0, 0, 0),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main dashboard
# ═══════════════════════════════════════════════════════════════════════════════

dashboard = pn.template.FastGridTemplate(
    title="Ptycho Dashboard",
    theme="dark",
    accent_base_color=ACCENT,
    header_background=BG_MID,
    header_color=TXT,
    sidebar_width=0,
    prevent_collision=True,
    save_layout=True,
    raw_css=[CSS],
    main=[
        pn.Column(
            pn.pane.HTML(
                f"""<div class="dash-header">
                  <div>
                    <p class="dash-title">⬡ PTYCHO DASHBOARD</p>
                    <p class="dash-sub">CID · Beamline P11 · PETRA III · DESY</p>
                  </div>
                </div>""",
                sizing_mode="stretch_width",
            ),
            card_exp,
            card_scan,
            card_stats,
            card_files,
            card_log,
            sizing_mode="stretch_width",
            margin=(0, 24),
        )
    ],
)


# ── Notebook-friendly entry point ─────────────────────────────────────────────
def show():
    """Call this inside a JupyterHub notebook cell."""
    pn.extension(
        "tabulator",
        sizing_mode="stretch_width",
        notifications=True,
        raw_css=[CSS],
    )
    return pn.Column(
        pn.pane.HTML(
            f"""<div class="dash-header" style="margin-bottom:16px">
              <div>
                <p class="dash-title">⬡ PTYCHO DASHBOARD</p>
                <p class="dash-sub">CID · Beamline P11 · PETRA III · DESY</p>
              </div>
            </div>""",
            sizing_mode="stretch_width",
        ),
        card_exp,
        card_scan,
        card_stats,
        card_files,
        card_log,
        sizing_mode="stretch_width",
    )


dashboard.servable()

if __name__ == "__main__":
    dashboard.show()
