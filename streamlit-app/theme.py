"""Design system: light theme CSS and small HTML helpers shared by all pages."""

import html

import streamlit as st

# Palette (kept in one place; mirrored in .streamlit/config.toml)
BG = "#F6F7F9"
SURFACE = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"
ACCENT = "#2F5FD0"
OK = "#0F8A6D"
ALERT = "#C8403D"
NEUTRAL = "#94A3B8"

TONES = {
    "ok": OK,
    "alert": ALERT,
    "neutral": NEUTRAL,
    "accent": ACCENT,
}

_CSS = f"""
<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --text:{TEXT}; --muted:{MUTED};
  --border:{BORDER}; --accent:{ACCENT}; --ok:{OK}; --alert:{ALERT}; --neutral:{NEUTRAL};
}}
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea, .stApp li {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}
.stApp {{ background: var(--bg); color: var(--text); }}
footer, #MainMenu {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ right: 1rem; }}
.block-container {{ padding-top: 2.25rem; padding-bottom: 3rem; max-width: 1280px; }}

/* typography */
.stApp h1 {{ font-size: 1.7rem; font-weight: 650; letter-spacing: -0.015em; color: var(--text); padding: 0; }}
.stApp h2 {{ font-size: 1.15rem; font-weight: 600; letter-spacing: -0.005em; color: var(--text); }}
.stApp h3 {{ font-size: 1rem; font-weight: 600; color: var(--text); }}
.stApp p, .stApp li {{ color: var(--text); line-height: 1.6; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.25rem; }}
.dds-brand {{ display:flex; align-items:center; gap:.65rem; padding:.25rem .25rem 1.1rem .25rem; }}
.dds-brand .name {{ font-weight:650; font-size:.98rem; line-height:1.2; letter-spacing:-.005em; color:var(--text); }}
.dds-brand .sub {{ font-size:.72rem; color:var(--muted); letter-spacing:.02em; }}
.dds-side-label {{ font-size:.68rem; font-weight:600; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); padding:.25rem .25rem .4rem .25rem; }}
.dds-side-foot {{ border-top:1px solid var(--border); margin-top:1.25rem; padding-top:.9rem;
  font-size:.78rem; color:var(--muted); line-height:1.5; }}
.dds-side-foot b {{ color:var(--text); font-weight:600; }}

[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
  justify-content:flex-start; text-align:left; border:none; box-shadow:none;
  border-radius:8px; padding:.5rem .75rem; min-height:0; font-weight:500; font-size:.92rem;
}}
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{ background:transparent; color:var(--muted); }}
[data-testid="stSidebar"] button[kind="secondary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {{ background:#F1F5F9; color:var(--text); }}
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{ background:#EAF0FD; color:var(--accent); }}
[data-testid="stSidebar"] button[kind="primary"]:hover,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {{ background:#E0E9FB; color:var(--accent); }}
[data-testid="stSidebar"] .stButton {{ margin-bottom:-.35rem; }}
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button [data-testid="stMarkdownContainer"] {{
  width:100%; justify-content:flex-start; text-align:left; }}
[data-testid="stSidebar"] .stButton button p {{ text-align:left; width:100%; margin:0; }}
[data-testid="stSidebar"] button[kind="secondary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] p {{ color:var(--muted); }}
[data-testid="stSidebar"] button[kind="secondary"]:hover p,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover p {{ color:var(--text); }}
[data-testid="stSidebar"] button[kind="primary"] p,
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p {{ color:var(--accent); font-weight:600; }}

/* page header */
.dds-header {{ margin-bottom:1.5rem; }}
.dds-header .title {{ font-size:1.7rem; font-weight:650; letter-spacing:-.015em; color:var(--text); line-height:1.2; }}
.dds-header .sub {{ color:var(--muted); font-size:.95rem; margin-top:.35rem; max-width:62ch; }}

/* cards */
.dds-card {{ background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:1.1rem 1.25rem; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.dds-card + .dds-card {{ margin-top:.9rem; }}
.dds-label {{ font-size:.7rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }}
.dds-value {{ font-size:1.55rem; font-weight:650; letter-spacing:-.015em; color:var(--text); line-height:1.25; margin-top:.2rem; }}
.dds-value.sm {{ font-size:1.25rem; }}
.dds-value .unit {{ font-size:.82rem; font-weight:500; color:var(--muted); margin-left:.2rem; letter-spacing:0; }}
.dds-note {{ font-size:.8rem; color:var(--muted); margin-top:.25rem; }}

.dds-chip {{ display:inline-flex; align-items:center; gap:.4rem; font-size:.78rem; font-weight:550;
  padding:.18rem .6rem; border-radius:999px; border:1px solid var(--border); background:#F8FAFC; color:var(--muted); }}
.dds-chip .dot {{ width:.45rem; height:.45rem; border-radius:50%; background:var(--neutral); }}
.dds-chip.ok {{ color:var(--ok); background:#EEF8F5; border-color:#CFE9E1; }}
.dds-chip.ok .dot {{ background:var(--ok); }}
.dds-chip.alert {{ color:var(--alert); background:#FDF1F0; border-color:#F3D1CF; }}
.dds-chip.alert .dot {{ background:var(--alert); }}
.dds-chip.accent {{ color:var(--accent); background:#EEF3FD; border-color:#D5E0F7; }}
.dds-chip.accent .dot {{ background:var(--accent); }}

.dds-bar {{ height:6px; border-radius:999px; background:#EDF1F5; overflow:hidden; margin-top:.55rem; }}
.dds-bar > span {{ display:block; height:100%; border-radius:999px; background:var(--accent); }}

.dds-grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:.9rem; margin-top:.9rem; }}
.dds-grid2 .dds-card {{ margin-top:0; padding:1rem 1.1rem; }}
.dds-grid2 .dds-label {{ white-space:nowrap; }}

.dds-kv {{ display:flex; justify-content:space-between; font-size:.85rem; padding:.3rem 0;
  border-bottom:1px solid #F1F5F9; color:var(--muted); }}
.dds-kv:last-child {{ border-bottom:none; }}
.dds-kv b {{ color:var(--text); font-weight:550; font-variant-numeric:tabular-nums; }}

.dds-placeholder {{ background:var(--surface); border:1px dashed #CBD5E1; border-radius:12px;
  aspect-ratio:4/3; max-height:58vh; min-height:320px; width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; padding:2rem; color:var(--muted); }}
.dds-placeholder .t {{ color:var(--text); font-weight:600; font-size:1.02rem; margin-bottom:.3rem; }}
.dds-placeholder .d {{ font-size:.88rem; max-width:38ch; }}

.dds-step {{ display:flex; gap:.9rem; padding:.85rem 0; border-bottom:1px solid var(--border); }}
.dds-step:last-child {{ border-bottom:none; }}
.dds-step .n {{ flex:0 0 1.6rem; height:1.6rem; border-radius:50%; background:#EEF3FD; color:var(--accent);
  font-size:.8rem; font-weight:650; display:flex; align-items:center; justify-content:center; margin-top:.1rem; }}
.dds-step .h {{ font-weight:600; font-size:.95rem; color:var(--text); }}
.dds-step .p {{ font-size:.88rem; color:var(--muted); margin-top:.1rem; line-height:1.5; }}

.dds-row {{ display:flex; flex-direction:column; padding:.2rem 0; }}
.dds-row .h {{ font-weight:600; font-size:.98rem; }}
.dds-row .p {{ font-size:.88rem; color:var(--muted); line-height:1.5; }}

/* native widgets */
.stApp .stButton > button, .stApp .stDownloadButton > button {{ border-radius:8px; font-weight:550; }}
[data-testid="stMain"] .stButton > button[kind="primary"],
[data-testid="stMain"] [data-testid="stBaseButton-primary"] {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
[data-testid="stMain"] button[kind="primary"] p,
[data-testid="stMain"] [data-testid="stBaseButton-primary"] p {{ color:#fff; }}
[data-testid="stFileUploader"] section {{ background:var(--surface); border:1px dashed #CBD5E1; border-radius:12px; padding:1.5rem; }}
[data-testid="stExpander"] {{ background:var(--surface); border:1px solid var(--border); border-radius:10px; }}
[data-testid="stProgress"] > div > div > div > div {{ background-color:var(--accent); }}
[data-testid="stAlert"] {{ border-radius:10px; }}
hr {{ border-color:var(--border); }}
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def _e(text):
    return html.escape(str(text))


def page_header(title, subtitle=None):
    sub = f'<div class="sub">{_e(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="dds-header"><div class="title">{_e(title)}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def chip(text, tone="neutral"):
    return f'<span class="dds-chip {tone}"><span class="dot"></span>{_e(text)}</span>'


def metric_card(label, value, unit="", note=""):
    unit_html = f'<span class="unit">{_e(unit)}</span>' if unit else ""
    note_html = f'<div class="dds-note">{_e(note)}</div>' if note else ""
    return (
        f'<div class="dds-card"><div class="dds-label">{_e(label)}</div>'
        f'<div class="dds-value">{_e(value)}{unit_html}</div>{note_html}</div>'
    )


def kv_rows(rows):
    """rows: list of (label, value-string)."""
    return "".join(f'<div class="dds-kv"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in rows)


def bar(fraction, tone="accent"):
    pct = max(0.0, min(1.0, fraction)) * 100
    return f'<div class="dds-bar"><span style="width:{pct:.1f}%;background:{TONES.get(tone, ACCENT)}"></span></div>'


def placeholder(title, description):
    return (
        f'<div class="dds-placeholder"><div class="t">{_e(title)}</div>'
        f'<div class="d">{_e(description)}</div></div>'
    )


def show_image(image_rgb, caption=None):
    """st.image across Streamlit versions (use_container_width replaced use_column_width)."""
    try:
        st.image(image_rgb, caption=caption, use_container_width=True)
    except TypeError:
        st.image(image_rgb, caption=caption, use_column_width=True)


def fmt_ms(value):
    return "—" if value is None else f"{value:.0f}"


def prediction_card(label, tone, chip_text, prob_drowsy=None, confidence=None):
    """Shared prediction summary used by the webcam and image pages."""
    conf = "—" if confidence is None else f"{confidence * 100:.1f}%"
    bar_html = ""
    if prob_drowsy is not None:
        bar_html = (
            f'<div class="dds-note" style="margin-top:.9rem">Drowsy probability · {prob_drowsy * 100:.1f}%</div>'
            + bar(prob_drowsy, tone if tone in ("ok", "alert") else "accent")
        )
    return (
        '<div class="dds-card">'
        '<div style="display:flex;justify-content:space-between;align-items:center">'
        '<div class="dds-label">Prediction</div>' + chip(chip_text, tone) + "</div>"
        f'<div class="dds-value" style="margin-top:.5rem">{_e(label)}</div>'
        f'<div class="dds-note">Confidence {conf}</div>' + bar_html + "</div>"
    )
