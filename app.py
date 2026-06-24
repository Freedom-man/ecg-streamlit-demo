from datetime import datetime
from pathlib import Path
import html
import io
import json

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from inference import ECGPredictor, predict_ensemble
from interpretation import interpret_profile


APP_DIR = Path(__file__).resolve().parent
CLASS_NAMES = {
    "CD": "Нарушения проводимости",
    "HYP": "Гипертрофия",
    "MI": "Инфаркт миокарда",
    "NORM": "Нормальная ЭКГ",
    "STTC": "Изменения ST/T",
}
CLASS_EXPLANATIONS = {
    "NORM": "Нормальная ЭКГ в рамках рассматриваемых групп.",
    "MI": "Признаки инфаркта миокарда или постинфарктных изменений.",
    "STTC": "Изменения сегмента ST и зубца T.",
    "CD": "Нарушения проводимости.",
    "HYP": "Признаки гипертрофии или перегрузки отделов сердца.",
}
CLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


st.set_page_config(
    page_title="Сервис предварительного анализа ЭКГ",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    :root {
        --teal: #078b82;
        --teal-dark: #06746e;
        --teal-soft: #e5f4f1;
        --ink: #121b2a;
        --muted: #687383;
        --line: #dfe5e8;
        --surface: #ffffff;
        --page: #f6f8f9;
    }
    .stApp { background: var(--page); color: var(--ink); }
    .block-container { max-width: 1420px; padding: 0.85rem 1.35rem 1.5rem; }
    [data-testid="stHeader"] { height: 2.8rem; background: rgba(246, 248, 249, 0.96); }
    [data-testid="stToolbar"] { display: flex !important; background: transparent !important; }
    [data-testid="stHeaderActionElements"],
    [data-testid="stAppDeployButton"],
    [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        position: fixed !important;
        top: 0.42rem !important;
        left: 0.55rem !important;
        z-index: 1000002 !important;
        width: 36px !important;
        height: 36px !important;
        align-items: center !important;
        justify-content: center !important;
        border: 1px solid #cfdadd !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        color: var(--teal-dark) !important;
        box-shadow: 0 2px 8px rgba(20, 36, 48, 0.12) !important;
    }
    [data-testid="stExpandSidebarButton"] svg { color: var(--teal-dark) !important; }
    [data-testid="stSidebarCollapseButton"] { display: flex !important; }
    [data-testid="stSidebar"] {
        background: #fbfdfd;
        border-right: 1px solid #dde5e6;
        min-width: 330px;
        max-width: 330px;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 1.1rem 1.25rem 1.5rem; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--ink) !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] input {
        background: #ffffff !important;
        color: var(--ink) !important;
        border-color: #cfd8dc !important;
    }
    [data-testid="stSidebar"] h2 { font-size: 1.25rem; letter-spacing: 0; }
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(20, 36, 48, 0.03);
    }
    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #65b5ae;
        border-radius: 7px;
        background: #f8fdfc;
        min-height: 130px;
    }
    [data-testid="stFileUploader"] section button { color: var(--teal-dark); }
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        border-radius: 6px;
        font-weight: 700;
        min-height: 46px;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
    }
    div[data-testid="stDownloadButton"] button {
        border-color: #68b8b0;
        color: var(--teal-dark);
        background: #fff;
    }
    .brand { display: flex; align-items: center; gap: 11px; margin: 0 0 15px; }
    .brand-mark {
        width: 48px; height: 48px; border: 2px solid var(--teal); border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: var(--teal); font-weight: 800; font-size: 13px;
    }
    .brand-line { color: var(--teal); font-size: 14px; font-weight: 750; line-height: 1.2; }
    .page-title { margin: 0 0 4px; font-size: 30px; line-height: 1.1; letter-spacing: 0; }
    .page-subtitle { color: var(--muted); font-size: 14px; margin: 0 0 12px; }
    .mobile-hint { display: none; }
    .summary-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 16px;
        margin: 15px 0 18px;
        padding: 0 0 12px;
        border-bottom: 1px solid #dfe6e8;
    }
    .summary-item {
        min-height: 54px;
        padding: 0;
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .summary-icon {
        flex: 0 0 34px; width: 34px; height: 34px; border-radius: 50%;
        background: var(--teal-soft); color: var(--teal-dark);
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 850;
    }
    .summary-kicker { color: #586472; font-size: 12px; margin-bottom: 2px; }
    .summary-main { color: var(--teal-dark); font-size: 15px; font-weight: 800; line-height: 1.2; overflow-wrap: anywhere; }
    .summary-extra { color: #56616e; font-size: 11px; margin-top: 3px; line-height: 1.3; overflow-wrap: anywhere; }
    .metric-card {
        height: 106px; background: #fff; border: 1px solid var(--line); border-radius: 7px;
        padding: 14px; display: flex; gap: 12px; align-items: flex-start;
        box-shadow: 0 1px 3px rgba(20, 36, 48, 0.035);
    }
    .metric-icon {
        flex: 0 0 42px; width: 42px; height: 42px; border-radius: 50%;
        background: var(--teal-soft); color: var(--teal-dark); display: flex;
        align-items: center; justify-content: center; font-size: 13px; font-weight: 800;
    }
    .metric-label { color: #586472; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: var(--teal-dark); font-size: 17px; font-weight: 750; line-height: 1.2; }
    .metric-value, .metric-note { overflow-wrap: anywhere; }
    .metric-note { color: #56616e; font-size: 11px; margin-top: 5px; line-height: 1.3; }
    .panel-title { font-size: 17px; font-weight: 750; margin: 0 0 12px; }
    .result-badge {
        display: inline-flex; align-items: center; gap: 7px;
        border-radius: 999px; padding: 6px 11px; margin: 0 0 10px;
        font-size: 12px; font-weight: 800; letter-spacing: 0;
    }
    .result-badge.normal { background: #e8f6ed; color: #276141; border: 1px solid #c7e8d2; }
    .result-badge.pathology { background: #fff7e8; color: #705017; border: 1px solid #ead5a6; }
    .result-badge.conflict { background: #fff0f0; color: #772027; border: 1px solid #e6b4b4; }
    .result-badge.uncertain { background: #f1f3f4; color: #4f5b64; border: 1px solid #d9dfe2; }
    .profile { padding: 4px 2px 2px; }
    .profile-head {
        border: 0; border-radius: 0; background: transparent;
        padding: 0; margin: 4px 0 16px;
    }
    .profile-head-label { color: #65707d; font-size: 12px; margin-bottom: 3px; }
    .profile-head-value { color: var(--teal-dark); font-size: 20px; font-weight: 850; line-height: 1.18; }
    .profile-head-note { color: #65707d; font-size: 11px; margin-top: 4px; }
    .profile-row { display: grid; grid-template-columns: 52px 1fr 92px; gap: 12px; align-items: center; margin: 16px 0; }
    .profile-class { font-weight: 750; }
    .profile-track { height: 13px; background: #edf0f2; border-radius: 4px; overflow: hidden; }
    .profile-fill { height: 100%; border-radius: 4px; background: #c7cdd1; }
    .profile-row.detected .profile-class, .profile-row.detected .profile-value { color: var(--teal-dark); }
    .profile-row.detected .profile-fill { background: var(--teal); }
    .profile-value { text-align: right; color: #65707d; font-variant-numeric: tabular-nums; }
    .legend { color: #65707d; font-size: 11px; display: flex; gap: 15px; margin-top: 20px; flex-wrap: wrap; }
    .legend i { width: 11px; height: 11px; display: inline-block; border-radius: 2px; margin-right: 6px; }
    .legend .high { background: var(--teal); } .legend .low { background: #cfd4d8; }
    .summary-label { color: #66717d; font-size: 13px; margin-bottom: 7px; }
    .summary-value { color: var(--teal-dark); font-size: 17px; font-weight: 750; }
    .summary-note { color: #5f6975; font-size: 13px; line-height: 1.5; margin-top: 9px; }
    .interpretation-alert {
        margin: 8px 0 10px;
        padding: 10px 12px;
        border: 1px solid #d9e1e4;
        border-left-width: 4px;
        border-radius: 5px;
        font-size: 12px;
        line-height: 1.5;
    }
    .interpretation-alert strong { display: block; margin-bottom: 4px; font-size: 13px; }
    .interpretation-alert.conflict { background: #fff0f0; border-color: #e6b4b4; border-left-color: #b4232c; color: #772027; }
    .interpretation-alert.pathology { background: #fff7e8; border-color: #ead5a6; border-left-color: #b97913; color: #705017; }
    .interpretation-alert.uncertain { background: #f4f5f6; border-color: #d9dfe2; border-left-color: #6e7c86; color: #4f5b64; }
    .interpretation-alert.normal { background: #ebf7ef; border-color: #c8e5d2; border-left-color: #318457; color: #285f42; }
    .rule-note { color: #66717d; font-size: 11px; line-height: 1.45; }
    .threshold-value { color: var(--teal-dark); font-size: 31px; font-weight: 800; }
    .decision-pill {
        background: #e8f6ed; border: 1px solid #c7e8d2; border-radius: 6px;
        color: #276141; padding: 8px 12px; margin: 6px 0; font-size: 13px; font-weight: 650;
    }
    .empty-pill { background: #f1f3f4; border-color: #e0e4e6; color: #67717b; }
    .support-note { color: #626e79; font-size: 12px; margin-top: 12px; line-height: 1.4; }
    .sidebar-info {
        border: 1px solid #dbe7e7; border-radius: 7px; background: #f8fdfc;
        padding: 10px 11px; color: #51616d; font-size: 12px; line-height: 1.45;
        margin: 8px 0 13px;
    }
    .sidebar-info strong { color: var(--teal-dark); }
    .section-divider {
        height: 1px;
        background: #dfe6e8;
        margin: 24px 0 18px;
    }
    [data-testid="stImage"], [data-testid="stImage"] img,
    [data-testid="stPyplotGlobalUse"] { max-width: 100% !important; }
    hr { border-color: #e2e7e9 !important; }

    /* Tablet and small laptop layouts. */
    @media (min-width: 769px) and (max-width: 1199px) {
        [data-testid="stSidebar"] { min-width: 280px; max-width: 280px; }
        [data-testid="stSidebar"] > div:first-child { padding: 0.95rem 1rem 1.3rem; }
        .block-container { padding: 0.75rem 1rem 1.25rem; }
        .page-title { font-size: 29px; }
        .page-subtitle { font-size: 14px; }
        .summary-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.75rem !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 230px !important;
            width: auto !important;
            min-width: 230px !important;
        }
        .metric-card { height: auto; min-height: 96px; }
        .profile-row { grid-template-columns: 46px minmax(90px, 1fr) 72px; gap: 8px; }
        .profile-value { font-size: 12px; }
    }

    /* Phone layout: one content column and an overlay sidebar. */
    @media (max-width: 768px) {
        [data-testid="stHeader"] { height: 3rem; background: #f6f8f9; }
        [data-testid="stExpandSidebarButton"] {
            top: auto !important;
            bottom: 1rem !important;
            left: 1rem !important;
            width: 148px !important;
            height: 46px !important;
            justify-content: flex-start !important;
            gap: 7px !important;
            padding: 0 12px !important;
            background: var(--teal) !important;
            border-color: var(--teal) !important;
            color: #ffffff !important;
            box-shadow: 0 3px 12px rgba(7, 139, 130, 0.28) !important;
            animation: menu-attention 0.8s ease-out 2;
        }
        [data-testid="stExpandSidebarButton"] svg { color: #ffffff !important; }
        [data-testid="stExpandSidebarButton"]::after {
            content: "Параметры";
            color: #ffffff;
            font-size: 13px;
            font-weight: 750;
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            min-width: min(88vw, 330px) !important;
            max-width: min(88vw, 330px) !important;
            box-shadow: 0 8px 28px rgba(20, 36, 48, 0.18) !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
            min-width: 0 !important;
            max-width: 0 !important;
            width: 0 !important;
            transform: translateX(-110vw) !important;
            border-right: 0 !important;
            box-shadow: none !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"] * {
            display: none !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding: 0.8rem 0.9rem 1.2rem; }
        .block-container { padding: 0.85rem 0.7rem 5rem; }
        .page-title { font-size: 25px; line-height: 1.14; }
        .page-subtitle { font-size: 13px; line-height: 1.4; margin-bottom: 10px; }
        .summary-strip {
            grid-template-columns: 1fr;
            margin: 10px 0 12px;
            gap: 10px;
        }
        .summary-item {
            min-height: 48px;
            padding: 0;
        }
        .mobile-hint {
            display: block;
            margin: 0 0 12px;
            padding: 10px 12px;
            border-left: 3px solid var(--teal);
            border-radius: 0 5px 5px 0;
            background: #eaf6f4;
            color: #305b58;
            font-size: 12px;
            line-height: 1.45;
        }
        .brand { margin-bottom: 10px; }
        .brand-mark { width: 40px; height: 40px; flex-basis: 40px; }

        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: nowrap !important;
            align-items: stretch !important;
            gap: 0.65rem !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 auto !important;
            width: 100% !important;
            min-width: 0 !important;
        }
        .metric-card { height: auto; min-height: 88px; padding: 12px; }
        .metric-icon { width: 38px; height: 38px; flex-basis: 38px; font-size: 11px; }
        .metric-value { font-size: 16px; }
        .panel-title { font-size: 16px; }
        .profile-row {
            grid-template-columns: 42px minmax(0, 1fr) 66px;
            gap: 7px;
            margin: 14px 0;
        }
        .profile-class { font-size: 13px; }
        .profile-value { font-size: 11px; }
        .legend { flex-direction: column; gap: 6px; margin-top: 14px; }
        .threshold-value { font-size: 27px; }
        .decision-pill { font-size: 12px; padding: 7px 9px; }
        [data-testid="stFileUploader"] section { min-height: 105px; }
        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button { min-height: 43px; }
    }

    @media (max-width: 420px) {
        .page-title { font-size: 22px; }
        .metric-card { min-height: 82px; }
        .profile-row { grid-template-columns: 38px minmax(0, 1fr) 62px; }
        .profile-value { font-size: 10px; }
    }
    @keyframes menu-attention {
        0% { transform: scale(1); }
        45% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor(model_label):
    return ECGPredictor(model_label)


@st.cache_data(show_spinner=False)
def load_demo_data(demo_samples_mtime):
    data = np.load(APP_DIR / "artifacts" / "demo_samples.npz", allow_pickle=True)
    return {
        "signals": data["signals"],
        "ecg_ids": data["ecg_ids"],
        "labels": data["labels"],
        "classes": data["classes"].tolist(),
    }


def load_uploaded_signal(uploaded_file):
    signal = np.load(io.BytesIO(uploaded_file.getvalue()), allow_pickle=False)
    return ECGPredictor.validate_signal(signal)


def collapse_sidebar_once_on_mobile():
    components.html(
        """
        <script>
        (() => {
          const parentWindow = window.parent;
          if (parentWindow.innerWidth > 768) return;

          const closeSidebar = () => {
            const collapseControl = parentWindow.document.querySelector(
              '[data-testid="stSidebarCollapseButton"] button'
            );
            const sidebar = parentWindow.document.querySelector('[data-testid="stSidebar"]');
            if (!collapseControl || !sidebar) return;

            const expanded = sidebar.getAttribute('aria-expanded');
            if (expanded === 'false') return;
            collapseControl.click();
          };

          setTimeout(closeSidebar, 80);
          setTimeout(closeSidebar, 220);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def plot_ecg(signal):
    duration = len(signal) / 100.0
    time = np.arange(len(signal)) / 100.0
    offsets = np.arange(11, -1, -1) * 2.15

    fig, ax = plt.subplots(figsize=(13.2, 5.25))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fffdfd")

    for lead_idx, offset in enumerate(offsets):
        lead = signal[:, lead_idx].astype(float)
        lead = lead - np.median(lead)
        scale = np.percentile(np.abs(lead), 98)
        if scale > 0:
            lead = lead / scale * 0.72
        ax.plot(time, lead + offset, color="#3f474c", linewidth=0.75)

    ax.set_xlim(0, duration)
    ax.set_ylim(-1.15, offsets[0] + 1.15)
    ax.set_yticks(offsets, labels=LEADS)
    ax.set_xlabel("Время, с", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.43))
    ax.grid(which="major", color="#e9b7ba", linewidth=0.65, alpha=0.72)
    ax.grid(which="minor", color="#f4d9db", linewidth=0.42, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ead6d8")
    fig.tight_layout(pad=0.8)
    return fig


def metric_card(icon, label, value, note):
    return f"""
    <div class="metric-card">
      <div class="metric-icon">{html.escape(icon)}</div>
      <div>
        <div class="metric-label">{html.escape(label)}</div>
        <div class="metric-value">{html.escape(value)}</div>
        <div class="metric-note">{html.escape(note)}</div>
      </div>
    </div>
    """


def summary_strip_html(items):
    blocks = []
    for icon, label, value, note in items:
        blocks.append(
            '<div class="summary-item">'
            f'<div class="summary-icon">{html.escape(icon)}</div>'
            '<div>'
            f'<div class="summary-kicker">{html.escape(label)}</div>'
            f'<div class="summary-main">{html.escape(value)}</div>'
            f'<div class="summary-extra">{html.escape(note)}</div>'
            '</div>'
            '</div>'
        )
    return '<div class="summary-strip">' + "".join(blocks) + '</div>'


def profile_html(probability_by_class, threshold):
    top_class = max(CLASS_ORDER, key=lambda class_name: probability_by_class.get(class_name, 0.0))
    top_probability = float(probability_by_class.get(top_class, 0.0))
    rows = []
    for class_name in CLASS_ORDER:
        probability = float(probability_by_class.get(class_name, 0.0))
        detected = probability >= threshold
        rows.append(
            f'<div class="profile-row {"detected" if detected else ""}">'
            f'<div class="profile-class">{class_name}</div>'
            f'<div class="profile-track"><div class="profile-fill" style="width:{probability * 100:.1f}%"></div></div>'
            f'<div class="profile-value">{probability:.2f} ({probability:.0%})</div>'
            '</div>'
        )
    return (
        '<div class="profile-head">'
        '<div class="profile-head-label">Наиболее вероятная группа</div>'
        f'<div class="profile-head-value">{top_class} — {html.escape(CLASS_NAMES[top_class])}</div>'
        f'<div class="profile-head-note">Вероятность: {top_probability:.2f} ({top_probability:.0%})</div>'
        '</div>'
        + "".join(rows)
        + (
        '<div class="legend">'
        f'<span><i class="high"></i>Выше порога (&ge; {threshold:.2f})</span>'
        f'<span><i class="low"></i>Ниже порога (&lt; {threshold:.2f})</span>'
        '</div>'
        '<div class="rule-note">Классы выше порога считаются значимыми. '
        'Результат не заменяет заключение врача.</div>'
        )
    )


def result_badge_html(interpretation):
    labels = {
        "normal": "Норма по выбранным группам",
        "pathology": "Обнаружены признаки патологии",
        "conflict": "Неоднозначный результат",
        "uncertain": "Требуется проверка",
    }
    level = interpretation["level"]
    return (
        f'<div class="result-badge {html.escape(level)}">'
        f'{html.escape(labels.get(level, "Требуется проверка"))}'
        "</div>"
    )


def result_help_markdown(threshold, borderline_margin=0.05):
    low = max(0.0, threshold - borderline_margin)
    high = min(1.0, threshold + borderline_margin)
    return f"""
Модель считает независимые вероятности для пяти групп. Несколько групп могут быть
отмечены одновременно.

| Код | Смысл |
|---|---|
| NORM | нормальная ЭКГ в рамках пяти групп |
| MI | инфаркт или постинфарктные изменения |
| STTC | изменения сегмента ST и зубца T |
| CD | нарушения проводимости |
| HYP | признаки гипертрофии или перегрузки |

**Правила:** значимый класс — **p ≥ {threshold:.2f}**; пограничная зона —
**{low:.2f}–{high:.2f}**; `NORM` вместе с патологией — неоднозначный профиль.

Результат предварительный и не заменяет заключение врача.
"""


demo_samples_path = APP_DIR / "artifacts" / "demo_samples.npz"
demo = load_demo_data(demo_samples_path.stat().st_mtime_ns)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">ECG</div><div class="brand-line">CardioAI<br>Research</div></div>',
        unsafe_allow_html=True,
    )
    st.header("Параметры анализа")
    st.write("**Загрузка ЭКГ**")

    use_demo = st.toggle("Использовать пример из PTB-XL", value=True)

    uploaded = None
    sample_options = []
    selected_demo = None
    if use_demo:
        for record_id, labels in zip(demo["ecg_ids"], demo["labels"]):
            label_names = ", ".join(
                class_name
                for class_name, value in zip(demo["classes"], labels)
                if value == 1
            )
            sample_options.append(f"ECG {record_id}: {label_names}")
        selected_demo = st.selectbox("Тестовая запись", sample_options)
        st.markdown(
            '<div class="sidebar-info"><strong>Источник:</strong> PTB-XL<br>'
            '<strong>Формат:</strong> демонстрационная запись загружается автоматически<br>'
            '<strong>Частота:</strong> 100 Гц</div>',
            unsafe_allow_html=True,
        )
    else:
        uploaded = st.file_uploader(
            "Выберите файл ЭКГ",
            type=["npy"],
            label_visibility="visible",
        )
        detected_format = Path(uploaded.name).suffix.upper().lstrip(".") if uploaded else "не выбран"
        st.markdown(
            '<div class="sidebar-info">'
            f'<strong>Формат файла:</strong> {html.escape(detected_format)}<br>'
            '<strong>Частота:</strong> ожидается 100 Гц<br>'
            '<strong>Размерность:</strong> отсчеты × 12 отведений</div>',
            unsafe_allow_html=True,
        )

    model_display = st.selectbox(
        "Модель",
        [
            "ResNet-1D — основная",
            "Ансамбль — максимальное качество",
            "FCN-1D — сравнительная",
        ],
        help="ResNet-1D показала лучший результат среди одиночных моделей. Ансамбль усредняет вероятности FCN и ResNet.",
    )
    model_choice = {
        "ResNet-1D — основная": "ResNet-1D",
        "Ансамбль — максимальное качество": "Ансамбль FCN + ResNet",
        "FCN-1D — сравнительная": "FCN-1D",
    }[model_display]
    st.caption(
        "ResNet-1D: macro AUC 0,9317 · Ансамбль: 0,9324 · FCN-1D: 0,9288"
    )
    threshold = st.slider("Порог вероятности", 0.10, 0.90, 0.50, 0.01)
    run_analysis = st.button(
        "Запустить анализ",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
    )
    st.markdown(
        '<div class="support-note">Поддерживаются 12-канальные ЭКГ длительностью не менее 2,5 секунды.</div>',
        unsafe_allow_html=True,
    )

signal = None
true_labels = []
record_id = None
source_label = "пользователь"
input_name = "Загруженный файл"

if use_demo:
    selected_idx = sample_options.index(selected_demo)
    signal = demo["signals"][selected_idx]
    record_id = int(demo["ecg_ids"][selected_idx])
    input_name = f"ECG {record_id}"
    source_label = "PTB-XL"
    true_labels = [
        class_name
        for class_name, value in zip(demo["classes"], demo["labels"][selected_idx])
        if value == 1
    ]
elif uploaded is not None:
    try:
        signal = load_uploaded_signal(uploaded)
        input_name = Path(uploaded.name).stem
    except Exception as exc:
        st.error(str(exc))

signature = f"{input_name}|{model_choice}"
analysis = st.session_state.get("analysis")
cached_analysis_is_actual = bool(analysis and analysis["signature"] == signature)

if run_analysis and signal is not None:
    if not cached_analysis_is_actual:
        with st.spinner("Выполняется анализ ЭКГ..."):
            if model_choice == "Ансамбль FCN + ResNet":
                computed = predict_ensemble(
                    [load_predictor("FCN-1D"), load_predictor("ResNet-1D")],
                    signal,
                )
            else:
                computed = load_predictor(model_choice).predict(signal)
        st.session_state["analysis"] = {
            "signature": signature,
            "result": computed,
            "time": datetime.now().strftime("%H:%M:%S"),
            "model": model_choice,
        }
        analysis = st.session_state["analysis"]
    collapse_sidebar_once_on_mobile()

if analysis and analysis["signature"] != signature:
    analysis = None

if analysis:
    result = analysis["result"]
    probability_by_class = dict(zip(result["classes"], result["probabilities"]))
else:
    probability_by_class = {class_name: 0.0 for class_name in CLASS_ORDER}

detected = [
    class_name
    for class_name in CLASS_ORDER
    if probability_by_class.get(class_name, 0.0) >= threshold
]
likely = sorted(probability_by_class, key=probability_by_class.get, reverse=True)[:2]
interpretation = interpret_profile(probability_by_class, threshold)

st.markdown('<h1 class="page-title">Сервис предварительного анализа ЭКГ</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Расчет вероятностей диагностических групп для 12-канальной электрокардиограммы</p>',
    unsafe_allow_html=True,
)
if not analysis:
    st.markdown(
        '<div class="mobile-hint"><strong>Начало работы:</strong> нажмите «Параметры», выберите ЭКГ и модель, затем запустите анализ.</div>',
        unsafe_allow_html=True,
    )

duration = len(signal) / 100 if signal is not None else 0
status_value = "Анализ завершен" if analysis else "Готов к анализу"
status_note = f"Сегодня, {analysis['time']}" if analysis else "Выберите запись и запустите расчет"

summary_items = [
    ("ID", "Запись", input_name, f"Источник: {source_label}"),
    ("SEC", "Длительность", f"{duration:.1f} сек", f"{len(signal) if signal is not None else 0} отсчетов"),
    ("12", "Каналы", "12 отведений", "I, II, III, aVR, aVL, aVF, V1-V6"),
    ("OK", "Статус", status_value, status_note),
]
st.markdown(summary_strip_html(summary_items), unsafe_allow_html=True)

main_left, main_right = st.columns([1.72, 1.08], gap="medium")

with main_left:
    st.markdown('<div class="panel-title">Предпросмотр сигнала</div>', unsafe_allow_html=True)
    if signal is None:
        st.info("Загрузите NPY-файл или включите демонстрационный пример PTB-XL.")
    else:
        ecg_figure = plot_ecg(signal)
        st.pyplot(ecg_figure, width="stretch")
        plt.close(ecg_figure)
        st.caption("Масштаб отображения нормализован отдельно для каждого отведения · 100 Гц")

with main_right:
    st.markdown('<div class="panel-title">Предварительная оценка модели</div>', unsafe_allow_html=True)
    if analysis:
        st.markdown(result_badge_html(interpretation), unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="result-badge uncertain">Ожидание анализа</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="profile">' + profile_html(probability_by_class, threshold) + "</div>",
        unsafe_allow_html=True,
    )
    with st.popover("Как читать результат?", icon=":material/help:"):
        st.markdown(result_help_markdown(threshold))

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
bottom_left, bottom_middle, bottom_right = st.columns([1.18, 1.12, 1], gap="medium")

with bottom_left:
    st.markdown('<div class="panel-title">Интерпретация</div>', unsafe_allow_html=True)
    if analysis:
        st.markdown(
            f'<div class="interpretation-alert {interpretation["level"]}">'
            f'<strong>{html.escape(interpretation["title"])}</strong>'
            f'{html.escape(interpretation["message"])}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="summary-label">Наиболее вероятные группы</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="summary-value">'
            + "<br>".join(
                f"{class_name} — {html.escape(CLASS_NAMES[class_name])}"
                for class_name in likely
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        if true_labels:
            st.markdown(
                '<div class="summary-note">Разметка PTB-XL: '
                + html.escape(", ".join(true_labels))
                + "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="summary-value">Ожидание анализа</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="summary-note">Результат является предварительной оценкой и не заменяет заключение врача.</div>',
        unsafe_allow_html=True,
    )

with bottom_middle:
    st.markdown('<div class="panel-title">Порог принятия решения</div>', unsafe_allow_html=True)
    threshold_col, decisions_col = st.columns([0.72, 1.28])
    with threshold_col:
        st.markdown('<div class="summary-label">Порог вероятности</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="threshold-value">{threshold:.2f}</div>', unsafe_allow_html=True)
    with decisions_col:
        if detected:
            for class_name in detected:
                st.markdown(
                    f'<div class="decision-pill">{class_name} — {html.escape(CLASS_NAMES[class_name])}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="decision-pill empty-pill">Классы выше порога отсутствуют</div>',
                unsafe_allow_html=True,
            )

with bottom_right:
    st.markdown('<div class="panel-title">Экспорт результата</div>', unsafe_allow_html=True)
    if analysis:
        report = {
            "record_id": record_id,
            "source": source_label,
            "model": analysis["model"],
            "threshold": threshold,
            "probabilities": {
                class_name: float(probability_by_class[class_name]) for class_name in CLASS_ORDER
            },
            "detected_classes": detected,
            "interpretation": interpretation,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        report_table = pd.DataFrame(
            {
                "class": CLASS_ORDER,
                "description": [CLASS_NAMES[class_name] for class_name in CLASS_ORDER],
                "probability": [probability_by_class[class_name] for class_name in CLASS_ORDER],
                "detected": [class_name in detected for class_name in CLASS_ORDER],
            }
        )
        st.download_button(
            "Скачать CSV",
            report_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{input_name}_ecg_report.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
        st.download_button(
            "Экспорт JSON",
            json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{input_name}_ecg_report.json",
            mime="application/json",
            icon=":material/data_object:",
            width="stretch",
        )
    else:
        st.caption("Экспорт станет доступен после анализа.")
