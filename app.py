"""
EcoML — Carbon and Energy-Aware Machine Learning Model Selector.

A production-grade Green AI experimentation platform and interactive dashboard.
Evaluates machine learning models across a scientific dual axis: predictive accuracy
versus empirical computational footprint (execution latency, RAM, energy, and carbon).
"""

from typing import Any, Dict, List, Optional
import io
import time
import sys
from pathlib import Path

# Ensure project root is prioritized in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# Optional Plotly import with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from config.settings import (
        PROJECT_NAME,
        PROJECT_TAGLINE,
        PROJECT_STATUS,
        PROJECT_DESCRIPTION,
        PROJECT_VERSION,
        PLANNED_FEATURES,
        FUTURE_ROADMAP_FEATURES,
        ROADMAP_STEPS,
        THEME_COLORS,
        RANDOM_STATE,
        TEST_SIZE_DEFAULT,
    )
except ImportError:
    # Handle live Streamlit hot-reload when config.settings is cached in sys.modules
    import importlib
    import config.settings
    config.settings = importlib.reload(config.settings)
    from config.settings import (
        PROJECT_NAME,
        PROJECT_TAGLINE,
        PROJECT_STATUS,
        PROJECT_DESCRIPTION,
        PROJECT_VERSION,
        PLANNED_FEATURES,
        ROADMAP_STEPS,
        THEME_COLORS,
        RANDOM_STATE,
        TEST_SIZE_DEFAULT,
    )
    FUTURE_ROADMAP_FEATURES = getattr(
        config.settings,
        "FUTURE_ROADMAP_FEATURES",
        [
            {
                "id": "hyperparameter_eco_tuning",
                "title": "Green Hyperparameter Tuning",
                "icon": "🎛️",
                "description": "Energy-budgeted Bayesian optimization that dynamically halts exploration when accuracy gains plateau relative to electrical power consumed.",
                "status": "Planned",
                "stage": "Future Evolution",
            },
            {
                "id": "deep_learning_gpu",
                "title": "Deep Learning & GPU Telemetry",
                "icon": "⚡",
                "description": "Hardware NVML power telemetry and FLOPs-per-watt profiling for PyTorch and TensorFlow deep neural network architectures.",
                "status": "Planned",
                "stage": "Future Evolution",
            },
            {
                "id": "grid_carbon_scheduler",
                "title": "Carbon-Aware Workload Dispatcher",
                "icon": "🌐",
                "description": "Automated training dispatch targeting cloud regions and local grid time windows with peak renewable energy generation.",
                "status": "Planned",
                "stage": "Future Evolution",
            },
            {
                "id": "quantization_export",
                "title": "Green Model Quantization & Export",
                "icon": "📦",
                "description": "Post-training 8-bit integer quantization and ONNX export optimized for low-power edge and microcontroller deployment.",
                "status": "Planned",
                "stage": "Future Evolution",
            },
        ],
    )
from src.data_preprocessing import (
    PreprocessingResult,
    load_dataset,
    validate_dataset,
    get_dataset_summary,
    identify_column_types,
    validate_target_column,
    separate_features_and_target,
    split_and_preprocess_data,
)
from src.benchmark_engine import (
    BenchmarkSuiteResult,
    get_supported_models,
    run_benchmark_suite,
    generate_benchmark_interpretation,
)
from src.telemetry import TelemetryCategory
from src.recommender import WeightProfile, calculate_model_recommendations

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title=f"{PROJECT_NAME} — {PROJECT_TAGLINE}",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Refined Custom Styling (Deep Midnight Navy & Restrained Emerald Identity)
# ==============================================================================
st.markdown(
    f"""
    <style>
        /* Base Application Canvas */
        .stApp {{
            background-color: {THEME_COLORS["navy_darkest"]};
            color: {THEME_COLORS["text_primary"]};
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            letter-spacing: -0.01em;
        }}

        /* Streamlit Native Header & Toolbar (Cohesive Dark Navy Identity) */
        header[data-testid="stHeader"] {{
            background-color: {THEME_COLORS["navy_darkest"]} !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.08);
            z-index: 99;
        }}

        [data-testid="stToolbar"] {{
            right: 1.5rem;
        }}

        /* Main Content Container & Spacing Optimization */
        .main .block-container,
        div[data-testid="stMainBlockContainer"],
        .block-container {{
            padding-top: 4.85rem !important;
            padding-bottom: 3.5rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }}

        /* Responsive spacing for smaller viewports */
        @media (max-width: 992px) {{
            .main .block-container,
            div[data-testid="stMainBlockContainer"],
            .block-container {{
                padding-top: 4.5rem !important;
                padding-left: 1.25rem !important;
                padding-right: 1.25rem !important;
            }}
        }}

        /* Eliminate phantom spacing from hidden markdown style injection */
        div[data-testid="stElementContainer"]:has(style:only-child) {{
            display: none !important;
        }}

        /* Sidebar Container */
        section[data-testid="stSidebar"] {{
            background-color: #0A0F1D;
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }}

        /* Custom Modern Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 6px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.14);
            padding-bottom: 2px;
            margin-bottom: 1.2rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border: none;
            color: {THEME_COLORS["text_secondary"]};
            font-weight: 500;
            font-size: 0.92rem;
            padding: 8px 18px;
            border-radius: 6px 6px 0 0;
            transition: color 0.15s ease, background-color 0.15s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: #FFFFFF;
            background-color: rgba(255, 255, 255, 0.03);
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            color: {THEME_COLORS["eco_green_light"]};
            font-weight: 600;
            border-bottom: 2px solid {THEME_COLORS["eco_green"]};
            background-color: rgba(16, 185, 129, 0.06);
        }}

        /* Buttons Styling */
        .stButton > button {{
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.88rem;
            padding: 0.45rem 1rem;
            transition: all 0.15s ease;
            border: 1px solid rgba(148, 163, 184, 0.2);
            background-color: #131B2E;
            color: {THEME_COLORS["text_primary"]};
        }}

        .stButton > button:hover {{
            border-color: {THEME_COLORS["eco_green"]};
            color: {THEME_COLORS["eco_green_light"]};
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
        }}

        .stButton > button[kind="primary"] {{
            background: linear-gradient(180deg, #10B981 0%, #059669 100%);
            border: 1px solid #10B981;
            color: #FFFFFF;
            font-weight: 600;
            box-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);
        }}

        .stButton > button[kind="primary"]:hover {{
            background: linear-gradient(180deg, #059669 0%, #047857 100%);
            border-color: #34D399;
            color: #FFFFFF;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
        }}

        /* Clean Panel / Hero Container */
        .eco-hero {{
            background: linear-gradient(180deg, #0F172A 0%, #0B132B 100%);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-left: 4px solid {THEME_COLORS["eco_green"]};
            border-radius: 10px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.3);
        }}

        /* Stat Card */
        .stat-card {{
            background-color: #0F172A;
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            transition: border-color 0.2s ease;
        }}

        .stat-card:hover {{
            border-color: rgba(16, 185, 129, 0.35);
        }}

        .stat-val {{
            color: #FFFFFF;
            font-size: 1.45rem;
            font-weight: 700;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }}

        .stat-lbl {{
            color: {THEME_COLORS["text_secondary"]};
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }}

        /* Architecture & Roadmap Cards */
        .arch-card {{
            background-color: #0F172A;
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            padding: 1.15rem;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }}

        .arch-card:hover {{
            border-color: rgba(16, 185, 129, 0.35);
            transform: translateY(-2px);
        }}

        .roadmap-card {{
            background-color: #0F172A;
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .roadmap-card-done {{
            border-left: 3px solid {THEME_COLORS["eco_green"]};
        }}

        .roadmap-card-planned {{
            border-left: 3px solid #38BDF8;
        }}

        /* Subtle Badges & Chips */
        .pill-badge {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 9px;
            border-radius: 9999px;
            font-size: 0.74rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}

        .pill-green {{
            background-color: rgba(16, 185, 129, 0.12);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .pill-sky {{
            background-color: rgba(56, 189, 248, 0.12);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        .pill-amber {{
            background-color: rgba(245, 158, 11, 0.12);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}

        .pill-muted {{
            background-color: rgba(148, 163, 184, 0.08);
            color: #94A3B8;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }}

        /* Notice Panel */
        .notice-panel {{
            background-color: #0F172A;
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 8px;
            padding: 1.25rem;
            margin-top: 1.2rem;
        }}

        .notice-title {{
            color: #FFFFFF;
            font-weight: 600;
            font-size: 0.98rem;
            margin-bottom: 0.4rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Enhanced Dataframe Styling for High Readability */
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 8px !important;
            background-color: #0F172A !important;
            margin-top: 0.4rem;
            margin-bottom: 0.8rem;
        }}

        div[data-testid="stDataFrame"] div[role="grid"] {{
            font-size: 0.95rem !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }}

        div[data-testid="stDataFrame"] div[role="columnheader"] {{
            background-color: #131E36 !important;
            color: #F8FAFC !important;
            font-weight: 700 !important;
            font-size: 0.88rem !important;
            letter-spacing: 0.02em !important;
        }}

        div[data-testid="stDataFrame"] div[role="gridcell"] {{
            font-size: 0.92rem !important;
            color: #E2E8F0 !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def format_plotly_theme(fig: Any, title_text: str = "") -> Any:
    """Apply refined dark midnight navy and emerald styling to a Plotly figure."""
    if not HAS_PLOTLY or fig is None:
        return fig
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color="#F8FAFC", size=14, family="Inter, -apple-system, sans-serif"),
        ),
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B132B",
        font=dict(color="#94A3B8", family="Inter, -apple-system, sans-serif", size=12),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1,
            font=dict(color="#E2E8F0", size=12),
        ),
        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12),
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.12)",
            zerolinecolor="rgba(148, 163, 184, 0.25)",
            tickfont=dict(color="#94A3B8", size=11),
            title_font=dict(color="#CBD5E1", size=12),
            showgrid=True,
        ),
        margin=dict(l=50, r=35, t=55, b=45),
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="rgba(16, 185, 129, 0.5)",
            font=dict(color="#F8FAFC", family="Inter, sans-serif", size=12),
        ),
    )
    return fig


# ==============================================================================
# Helper Utilities & Session State Initialization
# ==============================================================================
def build_config_signature(
    source_label: str,
    target: str,
    task: str,
    test_size: float,
    seed: int,
    scale: bool,
) -> str:
    """Build a deterministic configuration fingerprint string for state invalidation."""
    return f"{source_label}|{target}|{str(task).lower()}|{float(test_size):.2f}|{int(seed)}|{bool(scale)}"


# Ensure stateful keys persist across Streamlit reruns
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "data_source_label" not in st.session_state:
    st.session_state["data_source_label"] = ""
if "dataset_source_type" not in st.session_state:
    st.session_state["dataset_source_type"] = None
if "active_dataset_id" not in st.session_state:
    st.session_state["active_dataset_id"] = None
if "preprocessed_data" not in st.session_state:
    st.session_state["preprocessed_data"] = None
if "prep_signature" not in st.session_state:
    st.session_state["prep_signature"] = None
if "benchmark_suite_result" not in st.session_state:
    st.session_state["benchmark_suite_result"] = None
if "benchmark_prep_sig" not in st.session_state:
    st.session_state["benchmark_prep_sig"] = None

# ==============================================================================
# Sidebar (Workspace Context & System Telemetry Info)
# ==============================================================================
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 0.2rem 0 0.8rem 0;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.5rem;">🌱</span>
                <span style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">{PROJECT_NAME}</span>
                <span class="pill-badge pill-green" style="font-size: 0.68rem; padding: 2px 6px;">v{PROJECT_VERSION}</span>
            </div>
            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 3px;">
                Carbon & Energy-Aware ML Selector
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Current Session Context
    active_df_obj: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    active_prep_obj: Optional[PreprocessingResult] = st.session_state.get("preprocessed_data")
    active_suite_obj: Optional[BenchmarkSuiteResult] = st.session_state.get("benchmark_suite_result")

    st.markdown("##### 📌 Session Status")
    ds_status = "Loaded" if active_df_obj is not None else "None"
    ds_class = "pill-green" if active_df_obj is not None else "pill-muted"
    prep_status = "Configured" if active_prep_obj is not None else "Pending"
    prep_class = "pill-green" if active_prep_obj is not None else "pill-muted"
    bench_count = len(active_suite_obj.models_evaluated) if active_suite_obj is not None else 0
    bench_class = "pill-green" if bench_count > 0 else "pill-muted"

    st.markdown(
        f"""
        <div style="font-size: 0.84rem; line-height: 1.8; color: #CBD5E1;">
            <div>Dataset: <span class="pill-badge {ds_class}">{ds_status}</span></div>
            <div>Pipeline: <span class="pill-badge {prep_class}">{prep_status}</span></div>
            <div>Benchmarks: <span class="pill-badge {bench_class}">{bench_count} Models</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### 🎓 Academic Track")
    st.markdown(
        """
        - **Degree:** B.Tech CSE (AI & ML)
        - **Target Domain:** Green AI & Sustainable Computing
        - **Framework:** Empirical Multi-Criteria Decision Making
        """
    )

    st.markdown("---")
    with st.expander("📁 Project Structure", expanded=False):
        st.caption("EcoML Directory Layout:")
        st.code(
            """ECO-ML/
├── app.py
├── config/
├── data/
├── src/
└── tests/""",
            language="text",
        )

# ==============================================================================
# Refined Hero Header
# ==============================================================================
st.markdown(
    f"""
    <div class="eco-hero">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px; margin-bottom: 0.5rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 1.85rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">{PROJECT_NAME}</span>
                    <span class="pill-badge pill-green">● Operational Pipeline Active</span>
                </div>
                <div style="color: {THEME_COLORS['eco_green_light']}; font-size: 0.96rem; font-weight: 500; margin-top: 2px;">
                    {PROJECT_TAGLINE}
                </div>
            </div>
            <div>
                <span class="pill-badge pill-sky">Academic Capstone — AI & ML</span>
            </div>
        </div>
        <p style="color: {THEME_COLORS['text_secondary']}; font-size: 0.88rem; line-height: 1.55; margin: 0; max-width: 950px;">
            {PROJECT_DESCRIPTION}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Production Navigation Architecture (5 Coherent Sections)
# ==============================================================================
tab_overview, tab_preprocess, tab_benchmark, tab_recommender, tab_methodology = st.tabs([
    "🔭 Overview",
    "📂 Dataset Studio",
    "⚡ Benchmark Lab",
    "🌱 Green Intelligence",
    "📐 Project & Methodology",
])

# ------------------------------------------------------------------------------
# SECTION 1: OVERVIEW & EXPERIMENT DASHBOARD
# ------------------------------------------------------------------------------
with tab_overview:
    st.markdown("### 🔭 Experimentation Overview")
    st.caption("High-level pipeline status, active session summary, and dual-axis evaluation workflow.")

    # High-level System Metrics Row
    active_raw: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    active_prep: Optional[PreprocessingResult] = st.session_state.get("preprocessed_data")
    active_suite: Optional[BenchmarkSuiteResult] = st.session_state.get("benchmark_suite_result")
    current_sig: str = st.session_state.get("prep_signature", "")
    cached_bench_sig: str = st.session_state.get("benchmark_prep_sig", "")
    is_stale_bench: bool = bool(cached_bench_sig and cached_bench_sig != current_sig)

    ov1, ov2, ov3, ov4 = st.columns(4)
    with ov1:
        if active_raw is not None:
            val_txt = f"{active_raw.shape[0]} rows"
            lbl_txt = f"{active_raw.shape[1]} cols · Active Data"
        else:
            val_txt = "None"
            lbl_txt = "Awaiting Ingestion"
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-val">{val_txt}</div>
                <div class="stat-lbl">{lbl_txt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with ov2:
        if active_prep is not None:
            val_txt = f"{active_prep.train_shape[0]}/{active_prep.test_shape[0]}"
            lbl_txt = f"Train/Test · {active_prep.task_type.capitalize()}"
        else:
            val_txt = "Pending"
            lbl_txt = "Pipeline Unconfigured"
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-val">{val_txt}</div>
                <div class="stat-lbl">{lbl_txt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with ov3:
        if active_suite is not None and not is_stale_bench:
            val_txt = f"{len(active_suite.models_evaluated)}"
            lbl_txt = "Evaluated Algorithms"
        else:
            val_txt = "0"
            lbl_txt = "Benchmarks Idle"
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-val">{val_txt}</div>
                <div class="stat-lbl">{lbl_txt}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with ov4:
        if active_suite is not None and not is_stale_bench:
            try:
                ov_rep = calculate_model_recommendations(active_suite, weight_profile=WeightProfile.balanced())
                top_m = ov_rep.top_recommended_model
                top_name = top_m.model_name if top_m else "N/A"
                top_score = f"{top_m.composite_score:.1f}/100" if top_m else "Ready"
            except Exception:
                top_name = "Ready"
                top_score = "Ready"
        else:
            top_name = "Pending"
            top_score = "Awaiting Run"
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-val" style="color: {THEME_COLORS['eco_green_light']}; font-size: 1.25rem;">{top_name}</div>
                <div class="stat-lbl">Top Green Pick ({top_score})</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Display Active Experiment Snapshot or Clean Empty State
    if active_suite is not None and not is_stale_bench:
        interp = generate_benchmark_interpretation(active_suite)
        st.markdown("#### 📊 Active Benchmark Executive Summary")
        st.markdown(
            f"""
            <div class="eco-hero" style="border-left: 4px solid {THEME_COLORS['eco_green']}; padding: 1.25rem 1.5rem; margin-bottom: 1.2rem;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 1.05rem; margin-bottom: 0.35rem;">
                    Key Trade-Off Observation
                </div>
                <div style="color: {THEME_COLORS['text_primary']}; font-size: 0.94rem; line-height: 1.55; margin-bottom: 0.8rem;">
                    {interp['tradeoff_observation']}
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.85rem; color: {THEME_COLORS['text_secondary']};">
                    <div>🏆 Top Accuracy/Utility: <strong style="color: {THEME_COLORS['eco_green_light']};">{interp['best_predictive_model']}</strong> ({interp['best_predictive_metric']})</div>
                    <div>⚡ Fastest Training: <strong style="color: #38BDF8;">{interp['fastest_training_model']}</strong></div>
                    <div>⏱️ Lowest Latency: <strong style="color: #FBBF24;">{interp['lowest_latency_model']}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(active_suite.to_dataframe(), use_container_width=True)
        st.info("👉 Switch to **'🌱 Green Intelligence'** for interactive Pareto frontiers and radar trade-off analysis, or **'⚡ Benchmark Lab'** for full telemetry logs.")
    else:
        st.markdown("#### 🚀 Guided Experimentation Workflow")
        st.markdown(
            "EcoML evaluates machine learning models through a reproducible 3-step experimental workflow. "
            "Get started by ingesting data in **Dataset Studio** or exploring our offline benchmark datasets."
        )

        w1, w2, w3 = st.columns(3)
        with w1:
            st.markdown(
                f"""
                <div class="arch-card">
                    <div>
                        <div style="font-size: 1.6rem; margin-bottom: 0.4rem;">📂</div>
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 0.3rem;">1. Dataset Studio</div>
                        <p style="color: {THEME_COLORS['text_secondary']}; font-size: 0.85rem; line-height: 1.5;">
                            Upload tabular CSV files or load local datasets (Breast Cancer for classification, Diabetes for regression).
                            Perform automated schema diagnostics and configure zero-leakage preprocessing pipelines.
                        </p>
                    </div>
                    <span class="pill-badge pill-sky" style="align-self: flex-start; margin-top: 0.6rem;">Step 1: Ingestion</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with w2:
            st.markdown(
                f"""
                <div class="arch-card">
                    <div>
                        <div style="font-size: 1.6rem; margin-bottom: 0.4rem;">⚡</div>
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 0.3rem;">2. Benchmark Lab</div>
                        <p style="color: {THEME_COLORS['text_secondary']}; font-size: 0.85rem; line-height: 1.5;">
                            Train diverse scikit-learn models on strictly identical partitions.
                            Measure predictive scores alongside high-resolution execution duration, per-sample latency, and psutil RSS memory.
                        </p>
                    </div>
                    <span class="pill-badge pill-green" style="align-self: flex-start; margin-top: 0.6rem;">Step 2: Empirical Profiling</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with w3:
            st.markdown(
                f"""
                <div class="arch-card">
                    <div>
                        <div style="font-size: 1.6rem; margin-bottom: 0.4rem;">🌱</div>
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 1rem; margin-bottom: 0.3rem;">3. Green Intelligence</div>
                        <p style="color: {THEME_COLORS['text_secondary']}; font-size: 0.85rem; line-height: 1.5;">
                            Apply Multi-Criteria Decision Making (WSM) balancing utility against carbon, memory, and latency.
                            Analyze the dual-axis Pareto frontier to select the optimal model for your operational constraints.
                        </p>
                    </div>
                    <span class="pill-badge pill-green" style="align-self: flex-start; margin-top: 0.6rem;">Step 3: Recommendation</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------------------------
# SECTION 2: DATASET STUDIO & PREPROCESSING
# ------------------------------------------------------------------------------
with tab_preprocess:
    st.markdown("### 📂 Dataset Studio")
    st.caption("Tabular data intake, automated schema diagnostics, and leakage-free preprocessing pipeline configuration.")

    col_upload, col_sample = st.columns([3, 2])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload your tabular CSV dataset",
            type=["csv"],
            help="Select any local CSV file containing features and a target column.",
            key="csv_file_uploader",
        )

    with col_sample:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            load_sample = st.button("🧪 Demo CSV (15 rows)", use_container_width=True, help="Load minimal synthetic dataset for quick testing.")
        with btn_c2:
            load_breast_cancer_btn = st.button("🏥 Breast Cancer (569 rows)", use_container_width=True, help="Load local offline Wisconsin Breast Cancer dataset for classification.")
        with btn_c3:
            load_diabetes_btn = st.button("🩺 Diabetes (442 rows)", use_container_width=True, help="Load local offline scikit-learn Diabetes dataset for regression.")

    # Dataset Ingestion with robust cross-rerun session_state caching
    if uploaded_file is not None:
        upload_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("active_dataset_id") != upload_id:
            try:
                uploaded_df = load_dataset(uploaded_file)
                st.session_state["raw_df"] = uploaded_df
                st.session_state["data_source_label"] = f"Uploaded File: {uploaded_file.name}"
                st.session_state["dataset_source_type"] = "uploaded"
                st.session_state["active_dataset_id"] = upload_id
                # Invalidate previous preprocessing and benchmarks
                st.session_state["preprocessed_data"] = None
                st.session_state["prep_signature"] = None
                st.session_state["benchmark_suite_result"] = None
                st.session_state["benchmark_prep_sig"] = None
                st.session_state.pop("target_selector", None)
            except ValueError as err:
                st.error(f"❌ Upload Error: {err}")
    elif load_sample:
        demo_csv = io.StringIO(
            """age,income,department,tenure,previous_eval,promoted
25,52000,Sales,1,3.2,0
30,64000,Engineering,3,4.1,1
35,78000,Engineering,5,4.5,1
40,,Marketing,8,3.8,0
22,46000,Sales,1,,0
45,92000,,12,4.8,1
32,69000,Engineering,4,4.0,1
28,54000,Marketing,2,3.5,0
38,84000,Sales,7,4.6,1
50,99000,Engineering,15,4.9,1
29,59000,Sales,2,3.6,0
42,87000,Marketing,10,4.2,1
31,65000,Engineering,3,3.9,0
27,51000,Sales,1,3.4,0
48,95000,Marketing,13,4.7,1
"""
        )
        st.session_state["raw_df"] = load_dataset(demo_csv)
        st.session_state["data_source_label"] = "Synthetic Demo Dataset (Demonstration Only)"
        st.session_state["dataset_source_type"] = "synthetic_demo"
        st.session_state["active_dataset_id"] = "synthetic_demo"
        st.session_state["selected_task_type"] = "Classification"
        # Invalidate previous preprocessing and benchmarks
        st.session_state["preprocessed_data"] = None
        st.session_state["prep_signature"] = None
        st.session_state["benchmark_suite_result"] = None
        st.session_state["benchmark_prep_sig"] = None
        st.session_state.pop("target_selector", None)
    elif load_breast_cancer_btn:
        from sklearn.datasets import load_breast_cancer
        bc_data = load_breast_cancer(as_frame=True)
        st.session_state["raw_df"] = bc_data.frame
        st.session_state["data_source_label"] = "Local Public Dataset: Breast Cancer Wisconsin (569 samples)"
        st.session_state["dataset_source_type"] = "breast_cancer"
        st.session_state["active_dataset_id"] = "breast_cancer"
        st.session_state["selected_task_type"] = "Classification"
        # Invalidate previous preprocessing and benchmarks
        st.session_state["preprocessed_data"] = None
        st.session_state["prep_signature"] = None
        st.session_state["benchmark_suite_result"] = None
        st.session_state["benchmark_prep_sig"] = None
        st.session_state.pop("target_selector", None)
    elif load_diabetes_btn:
        from sklearn.datasets import load_diabetes
        diab_data = load_diabetes(as_frame=True)
        st.session_state["raw_df"] = diab_data.frame
        st.session_state["data_source_label"] = "Local Public Dataset: Diabetes Progression (442 samples)"
        st.session_state["dataset_source_type"] = "diabetes"
        st.session_state["active_dataset_id"] = "diabetes"
        st.session_state["selected_task_type"] = "Regression"
        # Invalidate previous preprocessing and benchmarks
        st.session_state["preprocessed_data"] = None
        st.session_state["prep_signature"] = None
        st.session_state["benchmark_suite_result"] = None
        st.session_state["benchmark_prep_sig"] = None
        st.session_state.pop("target_selector", None)

    df: Optional[pd.DataFrame] = st.session_state.get("raw_df")
    data_source_label: str = st.session_state.get("data_source_label", "")

    if df is None:
        st.info("👆 Please upload a tabular CSV file or select one of the local benchmark datasets above to begin.")
    else:
        val_result = validate_dataset(df, min_rows=10, min_cols=2)
        if not val_result["is_valid"]:
            st.error(f"⚠️ Validation Failed: {val_result['error_message']}")
        else:
            st.success(f"✅ {data_source_label} successfully loaded and validated.")
            if "Demonstration Only" in data_source_label:
                st.warning(
                    "⚠️ **Demonstration Mode:** This small synthetic dataset (15 samples) is intended for functional interface verification only. "
                    "Metrics obtained from this dataset must not be interpreted as empirical, real-world machine learning evidence."
                )
            summary = get_dataset_summary(df)

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(
                    f"""<div class="stat-card">
                        <div class="stat-val">{summary['rows']}</div>
                        <div class="stat-lbl">Total Samples</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""<div class="stat-card">
                        <div class="stat-val">{summary['columns']}</div>
                        <div class="stat-lbl">Total Columns</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""<div class="stat-card">
                        <div class="stat-val">{len(summary['numerical_columns'])}</div>
                        <div class="stat-lbl">Numerical Features</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""<div class="stat-card">
                        <div class="stat-val">{len(summary['categorical_columns'])}</div>
                        <div class="stat-lbl">Categorical Features</div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with m5:
                missing_color = THEME_COLORS["eco_green_light"] if summary['total_missing'] == 0 else "#FBBF24"
                st.markdown(
                    f"""<div class="stat-card">
                        <div class="stat-val" style="color: {missing_color};">{summary['total_missing']}</div>
                        <div class="stat-lbl">Missing Cells ({summary['duplicate_rows']} Dups)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            with st.expander("🔍 Dataset Preview & Detailed Column Inspection", expanded=True):
                st.markdown("##### 📄 Head of Dataset (First 10 Rows)")
                st.dataframe(df.head(10), use_container_width=True)

                st.markdown("##### 📊 Column Types, Non-Null Counts & Missing Rates")
                col_diag = []
                for col in df.columns:
                    col_diag.append({
                        "Column Name": col,
                        "Inferred Type": "Numerical" if col in summary["numerical_columns"] else "Categorical",
                        "Data Type": summary["dtypes"].get(col, ""),
                        "Missing Count": summary["missing_counts"].get(col, 0),
                        "Missing (%)": f"{summary['missing_percentages'].get(col, 0.0)}%",
                        "Unique Values": int(df[col].nunique()),
                    })
                st.dataframe(pd.DataFrame(col_diag), use_container_width=True)

            st.markdown("---")
            st.markdown("#### ⚙️ Task & Target Variable Configuration")

            col_target, col_task = st.columns(2)

            target_options = ["-- Select Target Column --"] + list(df.columns)
            default_idx = 0
            prev_target = st.session_state.get("selected_target_col")
            if prev_target and prev_target in target_options:
                default_idx = target_options.index(prev_target)
            else:
                for idx, col_name in enumerate(target_options):
                    if col_name.lower() in ["target", "label", "promoted", "price", "class"]:
                        default_idx = idx
                        break

            with col_target:
                selected_target = st.selectbox(
                    "🎯 Select Target Variable (y)",
                    options=target_options,
                    index=default_idx,
                    help="The column that models will be trained to predict.",
                    key="target_selector",
                )
                st.session_state["selected_target_col"] = selected_target

            with col_task:
                task_options = ["Classification", "Regression"]
                prev_task = st.session_state.get("selected_task_type", "Classification")
                task_idx = task_options.index(prev_task) if prev_task in task_options else 0
                task_type = st.radio(
                    "🏷️ Machine Learning Task Type",
                    options=task_options,
                    index=task_idx,
                    horizontal=True,
                    help="Classification predicts discrete classes; Regression predicts continuous values.",
                    key="task_type_radio",
                )
                st.session_state["selected_task_type"] = task_type

            with st.expander("🛠️ Advanced Partition & Scaling Parameters", expanded=False):
                p1, p2, p3 = st.columns(3)
                with p1:
                    test_size_val = st.slider(
                        "Test Partition Ratio",
                        min_value=0.10,
                        max_value=0.40,
                        value=float(st.session_state.get("selected_test_size", TEST_SIZE_DEFAULT)),
                        step=0.05,
                        help="Proportion of the dataset reserved for unbiased testing.",
                        key="test_size_slider",
                    )
                    st.session_state["selected_test_size"] = test_size_val
                with p2:
                    random_seed_val = st.number_input(
                        "Reproducibility Seed",
                        value=int(st.session_state.get("selected_random_seed", RANDOM_STATE)),
                        step=1,
                        help="Random seed for deterministic train/test splitting.",
                        key="random_seed_input",
                    )
                    st.session_state["selected_random_seed"] = int(random_seed_val)
                with p3:
                    scale_num_features = st.checkbox(
                        "Standardize Numerical Features (StandardScaler)",
                        value=bool(st.session_state.get("selected_scale_num", True)),
                        help="Scales features to zero mean and unit variance using training statistics.",
                        key="scale_num_checkbox",
                    )
                    st.session_state["selected_scale_num"] = scale_num_features

            if selected_target == "-- Select Target Column --":
                st.warning("👉 Please select a target variable above to configure and validate the pipeline.")
                st.session_state["preprocessed_data"] = None
                st.session_state["prep_signature"] = None
                st.session_state["benchmark_suite_result"] = None
                st.session_state["benchmark_prep_sig"] = None
            else:
                target_check = validate_target_column(
                    df=df,
                    target_column=selected_target,
                    task_type=task_type.lower(),
                )

                if not target_check["is_valid"]:
                    st.error(f"❌ Target Error: {target_check['error_message']}")
                    st.session_state["preprocessed_data"] = None
                    st.session_state["prep_signature"] = None
                    st.session_state["benchmark_suite_result"] = None
                    st.session_state["benchmark_prep_sig"] = None
                else:
                    st.success(
                        f"🎯 Target **'{selected_target}'** is valid for {task_type}. "
                        f"Unique values: {target_check['unique_classes']} | Missing target values: {target_check['null_count']}"
                    )

                    current_config_sig = build_config_signature(
                        data_source_label,
                        selected_target,
                        task_type,
                        test_size_val,
                        int(random_seed_val),
                        scale_num_features,
                    )

                    cached_prep = st.session_state.get("preprocessed_data")
                    cached_sig = st.session_state.get("prep_signature")

                    if cached_prep is not None and cached_sig != current_config_sig:
                        st.session_state["preprocessed_data"] = None
                        st.session_state["prep_signature"] = None
                        st.session_state["benchmark_suite_result"] = None
                        st.session_state["benchmark_prep_sig"] = None
                        st.info("ℹ️ Pipeline parameters were modified. Please click **'Configure & Run Preprocessing Pipeline'** below to build the updated pipeline.")

                    if st.button("🚀 Configure & Run Preprocessing Pipeline", type="primary"):
                        try:
                            X, y = separate_features_and_target(df, selected_target)
                            prep_result = split_and_preprocess_data(
                                X=X,
                                y=y,
                                task_type=task_type.lower(),
                                test_size=test_size_val,
                                random_state=int(random_seed_val),
                                scale_numerical=scale_num_features,
                            )

                            st.session_state["preprocessed_data"] = prep_result
                            st.session_state["dataset_name"] = data_source_label
                            st.session_state["prep_signature"] = current_config_sig
                            # Invalidate stale benchmark results
                            st.session_state["benchmark_suite_result"] = None
                            st.session_state["benchmark_prep_sig"] = None

                        except Exception as e:
                            st.error(f"❌ Preprocessing Failed: {str(e)}")

                    # Persistently render Preprocessing Architecture Summary whenever active prep matches current signature
                    active_prep: Optional[PreprocessingResult] = st.session_state.get("preprocessed_data")
                    if active_prep is not None and st.session_state.get("prep_signature") == current_config_sig:
                        st.markdown(
                            """
                            <div style="margin-top: 1rem; margin-bottom: 1rem;">
                                <span class="pill-badge pill-green">🛡️ Zero Data Leakage Guaranteed</span>
                                <span style="color: #94A3B8; font-size: 0.88rem; margin-left: 8px;">
                                    Imputation and scaling statistics were fitted strictly on the training partition.
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("##### 📋 Preprocessing Architecture Summary")
                        sum_col1, sum_col2, sum_col3 = st.columns(3)

                        with sum_col1:
                            st.markdown(
                                f"""
                                - **Training Set (X_train):** `{active_prep.train_shape[0]} samples × {active_prep.train_shape[1]} features`
                                - **Testing Set (X_test):** `{active_prep.test_shape[0]} samples × {active_prep.test_shape[1]} features`
                                - **Task Type:** `{active_prep.task_type.capitalize()}`
                                """
                            )

                        with sum_col2:
                            st.markdown(
                                f"""
                                - **Numerical Features ({len(active_prep.numerical_cols)}):** Median Imputation + {"StandardScaler" if scale_num_features else "No Scaling"}
                                - **Categorical Features ({len(active_prep.categorical_cols)}):** Most Frequent Imputation + OneHotEncoder (ignoring unseen categories)
                                """
                            )

                        with sum_col3:
                            st.markdown(
                                f"""
                                - **Target Variable:** `{active_prep.target_column}`
                                - **Classes / Range:** `{len(np.unique(active_prep.y_train))} distinct outcomes`
                                - **Reproducibility Seed:** `{random_seed_val}`
                                """
                            )

                        with st.expander("🔬 Inspect Transformed Training Matrix (First 5 Rows)", expanded=False):
                            preview_df = pd.DataFrame(
                                active_prep.X_train_processed[:5],
                                columns=active_prep.feature_names_out,
                            )
                            st.dataframe(preview_df, use_container_width=True)

                        st.success(
                            "✅ Preprocessing pipeline configured successfully! "
                            "Switch to the **'⚡ Benchmark Lab'** tab to run model benchmarks."
                        )

# ------------------------------------------------------------------------------
# SECTION 3: BENCHMARK LAB
# ------------------------------------------------------------------------------
with tab_benchmark:
    st.markdown("### ⚡ Benchmark Lab")
    st.caption(
        "Execute empirical benchmarks across candidate machine learning algorithms on preprocessed data. "
        "Measures predictive scores alongside training duration, per-sample inference latency, and memory footprint."
    )

    prep_data: Optional[PreprocessingResult] = st.session_state.get("preprocessed_data")

    if prep_data is None:
        st.warning(
            "⚠️ **No preprocessed data available.** Please complete dataset configuration in the "
            "**'📂 Dataset Studio'** tab before executing model benchmarks."
        )
    else:
        # Check for stale benchmark cache
        current_sig = st.session_state.get("prep_signature", "")
        cached_bench_sig = st.session_state.get("benchmark_prep_sig", "")
        is_stale_cache = bool(cached_bench_sig and cached_bench_sig != current_sig)

        # Active Dataset Summary Card
        st.markdown(
            f"""
            <div style="background-color: #0F172A; border: 1px solid rgba(148, 163, 184, 0.12);
                        border-radius: 8px; padding: 0.85rem 1.15rem; margin-bottom: 1.2rem; display: flex;
                        justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <span style="color: {THEME_COLORS['text_secondary']}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">ACTIVE DATASET:</span>
                    <strong style="color: #FFFFFF; margin-left: 6px;">{st.session_state.get('dataset_name', 'Loaded Data')}</strong>
                </div>
                <div>
                    <span style="color: {THEME_COLORS['text_secondary']}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">TASK:</span>
                    <span class="pill-badge pill-sky" style="margin-left: 6px;">{prep_data.task_type.upper()}</span>
                </div>
                <div>
                    <span style="color: {THEME_COLORS['text_secondary']}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">TARGET:</span>
                    <strong style="color: {THEME_COLORS['eco_green_light']}; margin-left: 6px;">{prep_data.target_column}</strong>
                </div>
                <div>
                    <span style="color: {THEME_COLORS['text_secondary']}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;">PARTITION:</span>
                    <span style="color: #FFFFFF; margin-left: 6px;">{prep_data.train_shape[0]} train / {prep_data.test_shape[0]} test</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "Demonstration Only" in st.session_state.get("dataset_name", ""):
            st.warning(
                "⚠️ **Demonstration Mode Active:** The current benchmark results are derived from a minimal synthetic demonstration dataset (15 samples). "
                "These metrics illustrate UI workflow and computational telemetry mechanics only, and must not be used as empirical scientific evidence."
            )

        # Model Selection Area
        st.markdown("#### 🛠️ Benchmark Configuration & Model Selection")
        supported = get_supported_models(prep_data.task_type)

        col_models, col_telemetry = st.columns([3, 2])

        with col_models:
            selected_models = st.multiselect(
                "Select algorithms to benchmark:",
                options=list(supported.keys()),
                default=list(supported.keys()),
                format_func=lambda m_id: supported[m_id],
                help="Choose two or more models to evaluate side-by-side on identical partitions.",
                key="benchmark_selected_models",
            )

        with col_telemetry:
            st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
            track_carbon = st.checkbox(
                "Enable Experimental CodeCarbon Telemetry",
                value=False,
                help=(
                    "Attempts hardware energy tracking during training. Note: Training runs under 1-2 seconds "
                    "typically finish below physical hardware energy counter sampling intervals."
                ),
                key="track_carbon_checkbox",
            )
            bench_seed = st.number_input(
                "Benchmark Seed",
                value=int(RANDOM_STATE),
                step=1,
                help="Ensures reproducible stochastic model initialization.",
                key="benchmark_seed_input",
            )

        suite_res_check: Optional[BenchmarkSuiteResult] = st.session_state.get("benchmark_suite_result")
        models_mismatch = False
        if suite_res_check is not None:
            cached_models = sorted([r.model_id for r in suite_res_check.models_evaluated])
            current_selected = sorted(selected_models)
            if cached_models != current_selected:
                models_mismatch = True

        if is_stale_cache or models_mismatch:
            if models_mismatch:
                st.warning(
                    "⚠️ Selected algorithms have changed since the last benchmark run. "
                    "Click **'Run Empirical Benchmark Suite'** below to generate updated benchmarks."
                )
            else:
                st.warning(
                    "⚠️ The dataset or preprocessing parameters were modified after the last benchmark run. "
                    "Click **'Run Empirical Benchmark Suite'** below to generate updated benchmarks."
                )

        # Execution Action Button
        run_benchmark_btn = st.button("🚀 Run Empirical Benchmark Suite", type="primary", use_container_width=True)

        if run_benchmark_btn:
            if not selected_models:
                st.error("❌ Please select at least one algorithm to benchmark.")
            else:
                with st.spinner("Executing empirical model training, latency measurement, and resource profiling..."):
                    suite_run = run_benchmark_suite(
                        prep_result=prep_data,
                        selected_model_ids=selected_models,
                        random_state=int(bench_seed),
                        track_emissions=track_carbon,
                        dataset_name=st.session_state.get("dataset_name", "Dataset"),
                    )
                    # Cache benchmark results tied to current preprocessing signature
                    st.session_state["benchmark_suite_result"] = suite_run
                    st.session_state["benchmark_prep_sig"] = current_sig

        # Display Live Benchmark Results
        suite_res: Optional[BenchmarkSuiteResult] = st.session_state.get("benchmark_suite_result")

        if suite_res is not None and not is_stale_cache and not models_mismatch:
            interpretation = generate_benchmark_interpretation(suite_res)

            # Insights & Trade-Offs Banner
            st.markdown(
                f"""
                <div class="eco-hero" style="padding: 1.35rem 1.6rem; margin-top: 1.4rem; margin-bottom: 1.2rem;">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 1.1rem; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 8px;">
                        <span>📊</span>
                        <span>Empirical Benchmark Insights & Trade-Offs</span>
                    </div>
                    <div style="color: {THEME_COLORS['text_primary']}; font-size: 0.94rem; line-height: 1.55; margin-bottom: 0.8rem;">
                        {interpretation['tradeoff_observation']}
                    </div>
                    <div style="display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.86rem; color: {THEME_COLORS['text_secondary']};">
                        <div>🏆 <strong>Top Predictive Performer:</strong> <span style="color: {THEME_COLORS['eco_green_light']};">{interpretation['best_predictive_model']}</span> ({interpretation['best_predictive_metric']})</div>
                        <div>⚡ <strong>Fastest Training:</strong> <span style="color: #38BDF8;">{interpretation['fastest_training_model']}</span></div>
                        <div>⏱️ <strong>Lowest Inference Latency:</strong> <span style="color: #FBBF24;">{interpretation['lowest_latency_model']}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Empirical Comparison Table
            st.markdown("#### 📋 Empirical Comparison Matrix")
            st.caption(
                "Predictive scores, execution durations, and process memory are measured directly on your host machine. "
                "Energy measurements are reported as estimated or unavailable according to hardware telemetry status."
            )
            df_results = suite_res.to_dataframe()
            st.dataframe(df_results, use_container_width=True)

            # Interactive Multi-View Visualizations
            st.markdown("#### 📈 Multi-Dimensional Visualizations")
            v_tab1, v_tab2, v_tab3 = st.tabs([
                "🎯 Predictive Performance",
                "⏱️ Training Time & Latency",
                "💾 Green AI Trade-Offs (Accuracy vs Cost)",
            ])

            successful_runs = [r for r in suite_res.models_evaluated if r.status == "SUCCESS"]

            with v_tab1:
                if not successful_runs:
                    st.info("No successful runs available to chart.")
                elif not HAS_PLOTLY:
                    st.info("Interactive Plotly charts require the `plotly` library.")
                else:
                    if suite_res.task_type == "classification":
                        chart_data = pd.DataFrame([
                            {
                                "Model": r.model_name,
                                "Accuracy": round(r.accuracy, 4) if r.accuracy is not None else None,
                                "F1-Score": round(r.f1, 4) if r.f1 is not None else None,
                                "Precision": round(r.precision, 4) if r.precision is not None else None,
                                "Recall": round(r.recall, 4) if r.recall is not None else None,
                            }
                            for r in successful_runs
                        ])
                        fig_pred = px.bar(
                            chart_data,
                            x="Model",
                            y=["F1-Score", "Accuracy"],
                            barmode="group",
                            color_discrete_sequence=[THEME_COLORS["eco_green"], THEME_COLORS["eco_green_light"]],
                        )
                        fig_pred = format_plotly_theme(fig_pred, "Classification Metric Scores (F1 & Accuracy)")
                        fig_pred.update_yaxes(range=[0, 1.05])
                        st.plotly_chart(fig_pred, use_container_width=True)
                    else:
                        chart_data = pd.DataFrame([
                            {
                                "Model": r.model_name,
                                "R² Score": round(r.r2, 4) if r.r2 is not None else None,
                                "RMSE": round(r.rmse, 4) if r.rmse is not None else None,
                                "MAE": round(r.mae, 4) if r.mae is not None else None,
                            }
                            for r in successful_runs
                        ])
                        fig_r2 = px.bar(
                            chart_data,
                            x="Model",
                            y="R² Score",
                            color="R² Score",
                            color_continuous_scale=["#0B132B", THEME_COLORS["eco_green"]],
                        )
                        fig_r2 = format_plotly_theme(fig_r2, "Regression Goodness-of-Fit (R² Score)")
                        st.plotly_chart(fig_r2, use_container_width=True)

            with v_tab2:
                if not successful_runs:
                    st.info("No successful runs available to chart.")
                elif not HAS_PLOTLY:
                    st.info("Plotly is required for interactive charts.")
                else:
                    col_t1, col_t2 = st.columns(2)
                    time_data = pd.DataFrame([
                        {
                            "Model": r.model_name,
                            "Training Duration (s)": round(r.training_time_sec, 5) if r.training_time_sec is not None else 0.0,
                            "Latency (ms/sample)": round(r.latency_per_sample_ms, 4) if r.latency_per_sample_ms is not None else 0.0,
                        }
                        for r in successful_runs
                    ])

                    with col_t1:
                        fig_train = px.bar(
                            time_data,
                            x="Model",
                            y="Training Duration (s)",
                            color_discrete_sequence=["#38BDF8"],
                        )
                        fig_train = format_plotly_theme(fig_train, "Model Training Duration (Seconds)")
                        st.plotly_chart(fig_train, use_container_width=True)

                    with col_t2:
                        fig_lat = px.bar(
                            time_data,
                            x="Model",
                            y="Latency (ms/sample)",
                            color_discrete_sequence=["#FBBF24"],
                        )
                        fig_lat = format_plotly_theme(fig_lat, "Per-Sample Inference Latency (Milliseconds)")
                        st.plotly_chart(fig_lat, use_container_width=True)

            with v_tab3:
                if not successful_runs:
                    st.info("No successful runs available to chart.")
                elif not HAS_PLOTLY:
                    st.info("Plotly is required for interactive charts.")
                else:
                    y_metric_key = "F1-Score" if suite_res.task_type == "classification" else "R² Score"
                    tradeoff_data = pd.DataFrame([
                        {
                            "Model": r.model_name,
                            "Training Time (s)": r.training_time_sec,
                            y_metric_key: r.f1 if suite_res.task_type == "classification" else r.r2,
                            "Peak Process RAM (MB)": r.ram_current_mb or 10.0,
                        }
                        for r in successful_runs
                    ])

                    fig_scatter = px.scatter(
                        tradeoff_data,
                        x="Training Time (s)",
                        y=y_metric_key,
                        size="Peak Process RAM (MB)",
                        color="Model",
                        hover_name="Model",
                        size_max=24,
                        color_discrete_sequence=[
                            THEME_COLORS["eco_green"],
                            "#38BDF8",
                            "#FBBF24",
                            "#A78BFA",
                            "#F43F5E",
                        ],
                    )
                    fig_scatter = format_plotly_theme(
                        fig_scatter,
                        f"Green AI Frontier: {y_metric_key} vs Training Time (Bubble Size = Process RAM MB)",
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

            # Scientific Measurement Transparency Callout
            st.markdown(
                f"""
                <div class="notice-panel">
                    <div class="notice-title">
                        <span>🛡️</span>
                        <span>Scientific Measurement Transparency</span>
                    </div>
                    <ul style="color: {THEME_COLORS['text_secondary']}; font-size: 0.88rem; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                        <li><strong>Training & Prediction Latency:</strong> Captured directly using high-resolution monotonic clocks (<code>time.perf_counter()</code>).</li>
                        <li><strong>Memory Footprint:</strong> Observed via OS process telemetry (<code>psutil.Process().memory_info().rss</code>).</li>
                        <li><strong>Energy & Carbon Emissions:</strong> Handled via CodeCarbon. If model training takes milliseconds on small datasets, hardware sampling intervals report energy as <em>Unavailable</em> rather than generating fake or fabricated figures.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.info(
                "👉 Ready for recommendation analysis! Switch to the **'🌱 Green Intelligence'** tab to "
                "configure criteria weights, analyze Pareto trade-offs, and inspect multi-criteria rankings."
            )

# ------------------------------------------------------------------------------
# SECTION 4: GREEN INTELLIGENCE (RECOMMENDER & PARETO ANALYSIS)
# ------------------------------------------------------------------------------
with tab_recommender:
    st.markdown("### 🌱 Green Intelligence")
    st.caption(
        "Multi-Criteria Decision Making (MCDM) balancing task performance against compute duration, "
        "inference latency, memory footprint, and carbon emissions. Identifies Pareto-optimal models with transparent scoring."
    )

    suite_res: Optional[BenchmarkSuiteResult] = st.session_state.get("benchmark_suite_result")
    current_sig = st.session_state.get("prep_signature", "")
    cached_bench_sig = st.session_state.get("benchmark_prep_sig", "")
    is_stale_cache = bool(cached_bench_sig and cached_bench_sig != current_sig)

    selected_bench_models = st.session_state.get("benchmark_selected_models")
    bench_models_mismatch = False
    if suite_res is not None and selected_bench_models is not None:
        cached_model_ids = sorted([r.model_id for r in suite_res.models_evaluated])
        current_selected_ids = sorted(selected_bench_models)
        if cached_model_ids != current_selected_ids:
            bench_models_mismatch = True

    if suite_res is None:
        st.warning(
            "⚠️ **No benchmark results available.** Please execute model benchmarks in the "
            "**'⚡ Benchmark Lab'** tab before viewing green model recommendations."
        )
    elif is_stale_cache or bench_models_mismatch:
        if bench_models_mismatch:
            st.warning(
                "⚠️ **Benchmark results are out of date.** Selected algorithms were modified. "
                "Please rerun model benchmarks in the '**⚡ Benchmark Lab**' tab before viewing recommendations."
            )
        else:
            st.warning(
                "⚠️ **Benchmark results are out of date.** Preprocessing parameters or dataset was modified. "
                "Please rerun model benchmarks in the '**⚡ Benchmark Lab**' tab before viewing recommendations."
            )
    else:
        # Profile Selection & Weight Configuration
        st.markdown("#### ⚖️ Evaluation Weight Profile")
        st.caption("Select an operational profile or define custom weights to prioritize specific deployment constraints:")

        profile_choice = st.radio(
            "Select Multi-Criteria Evaluation Profile",
            options=[
                "⚖️ Balanced (40% Performance, 60% System Efficiency)",
                "🌿 Green-First (25% Performance, 75% Eco-Efficiency)",
                "🎯 Accuracy-First (70% Performance, 30% System Efficiency)",
                "📱 Edge-Deployment (30% Performance, 70% Latency & Memory)",
                "⚙️ Custom Weight Profile",
            ],
            index=0,
            horizontal=False,
        )

        selected_profile: WeightProfile
        if "Balanced" in profile_choice:
            selected_profile = WeightProfile.balanced()
        elif "Green-First" in profile_choice:
            selected_profile = WeightProfile.green_first()
        elif "Accuracy-First" in profile_choice:
            selected_profile = WeightProfile.accuracy_first()
        elif "Edge-Deployment" in profile_choice:
            selected_profile = WeightProfile.edge_deployment()
        else:
            st.markdown("##### 🎛️ Custom Criteria Weights")
            cw1, cw2, cw3, cw4, cw5 = st.columns(5)
            with cw1:
                w_perf = st.slider("Predictive Utility (F1/R²)", 0.0, 1.0, 0.40, 0.05)
            with cw2:
                w_train = st.slider("Training Speed", 0.0, 1.0, 0.15, 0.05)
            with cw3:
                w_lat = st.slider("Inference Latency", 0.0, 1.0, 0.15, 0.05)
            with cw4:
                w_ram = st.slider("RAM Compactness", 0.0, 1.0, 0.15, 0.05)
            with cw5:
                w_carb = st.slider("Carbon / Energy", 0.0, 1.0, 0.15, 0.05)

            selected_profile = WeightProfile(
                w_performance=w_perf,
                w_train_time=w_train,
                w_inference_latency=w_lat,
                w_ram=w_ram,
                w_carbon=w_carb,
                name="Custom User Configuration",
            )

        # Run Recommendation Engine
        try:
            report = calculate_model_recommendations(suite_res, weight_profile=selected_profile)
        except Exception as e:
            st.error(f"❌ Error calculating recommendations: {str(e)}")
            report = None

        if report is not None and report.top_recommended_model is not None:
            top = report.top_recommended_model
            top_run = next((r for r in suite_res.models_evaluated if r.model_id == top.model_id), None)

            # Telemetry Redistribution Banner (if carbon was unavailable)
            if report.redistribution_note:
                st.info(f"ℹ️ **Dynamic Weight Redistribution:** {report.redistribution_note}")

            # Active Multi-Criteria Scoring Weights & Criteria Breakdown Panel
            st.markdown(
                f"""
                <div class="stat-card" style="margin-top: 0.6rem; margin-bottom: 1.1rem; background-color: #0F172A; border: 1px solid rgba(148, 163, 184, 0.16);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 0.6rem;">
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                            <span>⚖️</span>
                            <span>Active Multi-Criteria Weights ({report.profile_name})</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8; font-family: monospace;">
                            EcoML Score = Σ (w_j × Normalized_Score_j) × 100
                        </div>
                    </div>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 130px; background-color: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 6px; padding: 6px 10px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase;">Predictive Utility</div>
                            <div style="color: #34D399; font-size: 1.05rem; font-weight: 700;">{report.effective_weights.get('performance', 0.0)*100:.1f}%</div>
                        </div>
                        <div style="flex: 1; min-width: 130px; background-color: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 6px 10px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase;">Training Speed</div>
                            <div style="color: #38BDF8; font-size: 1.05rem; font-weight: 700;">{report.effective_weights.get('train_time', 0.0)*100:.1f}%</div>
                        </div>
                        <div style="flex: 1; min-width: 130px; background-color: rgba(251, 191, 36, 0.08); border: 1px solid rgba(251, 191, 36, 0.25); border-radius: 6px; padding: 6px 10px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase;">Inference Latency</div>
                            <div style="color: #FBBF24; font-size: 1.05rem; font-weight: 700;">{report.effective_weights.get('inference_latency', 0.0)*100:.1f}%</div>
                        </div>
                        <div style="flex: 1; min-width: 130px; background-color: rgba(167, 139, 250, 0.08); border: 1px solid rgba(167, 139, 250, 0.25); border-radius: 6px; padding: 6px 10px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase;">RAM Compactness</div>
                            <div style="color: #A78BFA; font-size: 1.05rem; font-weight: 700;">{report.effective_weights.get('ram', 0.0)*100:.1f}%</div>
                        </div>
                        <div style="flex: 1; min-width: 130px; background-color: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 6px; padding: 6px 10px;">
                            <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase;">Carbon & Energy</div>
                            <div style="color: #CBD5E1; font-size: 1.05rem; font-weight: 700;">{f"{report.effective_weights.get('carbon', 0.0)*100:.1f}%" if 'carbon' in report.effective_weights else "0.0% (Redistributed)"}</div>
                        </div>
                    </div>
                    <div style="margin-top: 0.6rem; font-size: 0.81rem; color: #94A3B8; line-height: 1.45;">
                        ℹ️ <b>EcoML Score vs. Raw Accuracy:</b> The <b>EcoML Score (0–100)</b> is an empirical multi-criteria composite fitness index balancing task accuracy against runtime and memory overhead. It is <b>not</b> raw percentage accuracy. A model with 95% accuracy that is 10× faster and uses half the memory may receive a higher EcoML Score than a 96% accurate model that takes 20 seconds to train.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top Recommendation Hero Card with Detailed Telemetry
            pareto_badge_class = "pill-green" if top.is_pareto_optimal else "pill-amber"
            pareto_badge_label = "⭐ Pareto-Optimal Frontier" if top.is_pareto_optimal else "Standard Trade-Off"

            # Determine primary task metrics based on task type
            if suite_res.task_type == "classification" and top_run is not None:
                primary_task_metric_label = "Test Accuracy"
                primary_task_metric_val = f"{top_run.accuracy * 100:.1f}%" if top_run.accuracy is not None else "N/A"
                primary_task_sub_label = "(Test Set Accuracy)"
                perf_metrics_summary = f"Accuracy: {top_run.accuracy * 100:.1f}% · F1: {top_run.f1:.4f}" if top_run.accuracy is not None and top_run.f1 is not None else "N/A"
                perf_sub_summary = f"Precision: {top_run.precision:.4f} · Recall: {top_run.recall:.4f}" if top_run.precision is not None and top_run.recall is not None else "Classification metrics"
            else:
                primary_task_metric_label = "R² Score"
                primary_task_metric_val = f"{top_run.r2:.4f}" if top_run is not None and top_run.r2 is not None else "N/A"
                primary_task_sub_label = "(Goodness-of-Fit R²)"
                perf_metrics_summary = f"R² Goodness-of-Fit: {top_run.r2:.4f}" if top_run is not None and top_run.r2 is not None else "N/A"
                perf_sub_summary = f"MAE: {top_run.mae:.2f} · RMSE: {top_run.rmse:.2f}" if top_run is not None and top_run.mae is not None else "Regression metrics"

            # Measured runtime formatting
            if top_run is not None and top_run.training_time_sec is not None:
                if top_run.training_time_sec < 0.001:
                    train_time_display = f"{top_run.training_time_sec * 1_000_000:.1f} µs"
                elif top_run.training_time_sec < 1.0:
                    train_time_display = f"{top_run.training_time_sec * 1000:.2f} ms"
                else:
                    train_time_display = f"{top_run.training_time_sec:.3f} s"
            else:
                train_time_display = "N/A"

            if top_run is not None and top_run.latency_per_sample_ms is not None:
                if top_run.latency_per_sample_ms < 0.001:
                    latency_display = f"{top_run.latency_per_sample_ms * 1000:.2f} µs / sample"
                else:
                    latency_display = f"{top_run.latency_per_sample_ms:.4f} ms / sample"
            else:
                latency_display = "N/A"

            # Measured memory & energy formatting
            if top_run is not None and top_run.ram_current_mb is not None:
                ram_display = f"{top_run.ram_current_mb:.1f} MB"
                ram_sub = f"Delta: {top_run.ram_delta_mb:+.2f} MB (Baseline: {top_run.baseline_ram_mb:.1f} MB)"
            else:
                ram_display = "N/A"
                ram_sub = "Memory telemetry"

            # Transparent energy status (never label unavailable as measured)
            if top_run is not None and top_run.energy_kwh is not None and top_run.energy_kwh > 0 and top_run.energy_status == "Measured":
                energy_short_display = f"{top_run.energy_kwh:.6f} kWh (Measured)"
            elif top_run is not None and "Disabled" in (top_run.energy_status or ""):
                energy_short_display = "Disabled (Opt-in)"
            else:
                energy_short_display = "Unavailable (Sub-second execution window)"

            st.markdown(
                f"""
                <div class="eco-hero" style="margin-top: 0.8rem; margin-bottom: 1.4rem;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;">
                        <div>
                            <div style="display: flex; gap: 8px; align-items: center; margin-bottom: 4px; flex-wrap: wrap;">
                                <span class="pill-badge pill-green">🏆 #1 TOP RECOMMENDED MODEL</span>
                                <span class="pill-badge {pareto_badge_class}">{pareto_badge_label}</span>
                                <span class="pill-badge pill-sky">Profile: {report.profile_name}</span>
                            </div>
                            <div style="color: #FFFFFF; font-size: 1.95rem; font-weight: 800; letter-spacing: -0.02em;">
                                {top.model_name}
                            </div>
                        </div>
                        <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                            <div style="text-align: right; background-color: rgba(16, 185, 129, 0.12); border: 1px solid #10B981; border-radius: 8px; padding: 0.75rem 1.25rem;">
                                <div style="color: #34D399; font-size: 2.1rem; font-weight: 800; line-height: 1;">
                                    {top.composite_score:.1f}
                                </div>
                                <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-top: 3px;">
                                    EcoML Score / 100
                                </div>
                                <div style="color: #64748B; font-size: 0.68rem; margin-top: 1px;">
                                    (Multi-Criteria Fitness)
                                </div>
                            </div>
                            <div style="text-align: right; background-color: rgba(56, 189, 248, 0.10); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 8px; padding: 0.75rem 1.25rem;">
                                <div style="color: #38BDF8; font-size: 2.1rem; font-weight: 800; line-height: 1;">
                                    {primary_task_metric_val}
                                </div>
                                <div style="color: #94A3B8; font-size: 0.72rem; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; margin-top: 3px;">
                                    {primary_task_metric_label}
                                </div>
                                <div style="color: #64748B; font-size: 0.68rem; margin-top: 1px;">
                                    {primary_task_sub_label}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 1.1rem; margin-bottom: 1.1rem;">
                        <div class="stat-card" style="background-color: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.14);">
                            <div class="stat-lbl">🎯 Predictive Performance</div>
                            <div style="color: #F8FAFC; font-size: 1.05rem; font-weight: 700; margin-top: 3px;">
                                {perf_metrics_summary}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.76rem; margin-top: 2px;">
                                {perf_sub_summary}
                            </div>
                        </div>
                        <div class="stat-card" style="background-color: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.14);">
                            <div class="stat-lbl">⏱️ Measured Training Time</div>
                            <div style="color: #38BDF8; font-size: 1.05rem; font-weight: 700; margin-top: 3px;">
                                {train_time_display}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.76rem; margin-top: 2px;">
                                Monotonic clock (time.perf_counter)
                            </div>
                        </div>
                        <div class="stat-card" style="background-color: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.14);">
                            <div class="stat-lbl">⚡ Measured Inference Latency</div>
                            <div style="color: #FBBF24; font-size: 1.05rem; font-weight: 700; margin-top: 3px;">
                                {latency_display}
                            </div>
                            <div style="color: #94A3B8; font-size: 0.76rem; margin-top: 2px;">
                                Per-sample test latency
                            </div>
                        </div>
                        <div class="stat-card" style="background-color: rgba(15, 23, 42, 0.65); border: 1px solid rgba(148, 163, 184, 0.14);">
                            <div class="stat-lbl">💾 Measured Memory & Energy</div>
                            <div style="color: #34D399; font-size: 1.05rem; font-weight: 700; margin-top: 3px;">
                                {ram_display} Peak RSS
                            </div>
                            <div style="color: #94A3B8; font-size: 0.76rem; margin-top: 2px;">
                                Energy: {energy_short_display}
                            </div>
                        </div>
                    </div>

                    <div style="border-top: 1px solid rgba(148, 163, 184, 0.15); padding-top: 0.85rem; margin-top: 0.5rem;">
                        <div style="color: #34D399; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">
                            💡 Why This Model Was Recommended
                        </div>
                        <div style="color: #E2E8F0; font-size: 0.94rem; line-height: 1.6;">
                            {report.tradeoff_synthesis}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Recommendation Ranking Matrix Table
            st.markdown("#### 📋 Multi-Criteria Decision Matrix")
            st.caption(
                "Rankings based on the Weighted Sum Model (WSM). All efficiency metrics are normalized into utility indices [0–100] "
                "where 100.0 represents the highest observed efficiency in the candidate pool."
            )

            st.markdown(
                """
                <div style="background-color: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 6px; padding: 0.65rem 1rem; margin-bottom: 0.8rem; font-size: 0.83rem; color: #94A3B8; line-height: 1.5;">
                    <span style="color: #34D399; font-weight: 700;">Column Guide & Scale:</span>
                    <b style="color: #F8FAFC;">EcoML Score:</b> Composite multi-criteria fitness [0–100] •
                    <b style="color: #F8FAFC;">Predictive Utility:</b> Task performance [0–100] (F1 / R²) •
                    <b style="color: #F8FAFC;">Efficiency Indices (0–100):</b> Inverted cost metrics (100 = fastest duration, lowest latency, or lowest RAM) •
                    <b style="color: #F8FAFC;">Pareto Optimal:</b> Identifies non-dominated models on the efficiency frontier.
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_recom = report.to_dataframe()
            st.dataframe(
                df_recom,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.TextColumn(
                        "Rank",
                        help="Overall recommendation rank based on composite EcoML score (1 = Top Pick)",
                        width="small",
                    ),
                    "Model": st.column_config.TextColumn(
                        "Algorithm / Model",
                        help="Evaluated machine learning algorithm",
                        width="medium",
                    ),
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Training and evaluation status",
                        width="small",
                    ),
                    "EcoML Score": st.column_config.NumberColumn(
                        "EcoML Score (0–100)",
                        help="Multi-criteria weighted composite score (100 = best overall fitness across performance & efficiency)",
                        format="%.1f",
                        width="medium",
                    ),
                    "Pareto Optimal": st.column_config.TextColumn(
                        "Pareto Frontier",
                        help="Indicates whether the model lies on the non-dominated Pareto efficiency frontier",
                        width="small",
                    ),
                    "Predictive Utility": st.column_config.NumberColumn(
                        "Predictive Utility (0–100)",
                        help="Normalized predictive performance (F1-score for classification, R² for regression; 100 = highest score)",
                        format="%.1f",
                        width="medium",
                    ),
                    "Train Time Eff.": st.column_config.NumberColumn(
                        "Training Speed Eff. (0–100)",
                        help="Normalized inverse training time (100 = fastest model, 0 = slowest)",
                        format="%.1f",
                        width="medium",
                    ),
                    "Latency Eff.": st.column_config.NumberColumn(
                        "Inference Latency Eff. (0–100)",
                        help="Normalized inverse per-sample prediction latency (100 = lowest latency)",
                        format="%.1f",
                        width="medium",
                    ),
                    "RAM Eff.": st.column_config.NumberColumn(
                        "RAM Compactness (0–100)",
                        help="Normalized inverse peak process resident memory (100 = lowest RAM consumption)",
                        format="%.1f",
                        width="medium",
                    ),
                    "Carbon Eff.": st.column_config.NumberColumn(
                        "Carbon Eff. (0–100)",
                        help="Normalized inverse carbon emissions (100 = lowest emissions)",
                        format="%.1f",
                        width="medium",
                    ),
                },
            )

            # Interactive Multi-Dimensional Visualizations
            st.markdown("#### 📈 Decision Frontier Visualizations")
            r_tab1, r_tab2, r_tab3 = st.tabs([
                "🕸️ Multi-Criteria Radar Comparison",
                "🎯 Dual-Axis Pareto Frontier",
                "🔍 Deep Resource Telemetry & Diagnostics",
            ])

            with r_tab1:
                if not HAS_PLOTLY:
                    st.info("Interactive radar charts require Plotly.")
                else:
                    radar_fig = go.Figure()
                    dimensions_labels = {
                        "performance": "Predictive Quality",
                        "train_time": "Training Speed",
                        "inference_latency": "Latency Eff.",
                        "ram": "RAM Compactness",
                        "carbon": "Carbon Efficiency",
                    }
                    active_dims = [d for d in report.effective_weights.keys() if d in dimensions_labels]

                    colors = [THEME_COLORS["eco_green"], "#38BDF8", "#FBBF24", "#A78BFA", "#F43F5E", "#EC4899"]
                    for idx, model_detail in enumerate(report.ranked_models):
                        if model_detail.status == "SUCCESS":
                            r_vals = [model_detail.normalized_scores.get(dim, 0.0) * 100 for dim in active_dims]
                            theta_labels = [dimensions_labels[dim] for dim in active_dims]
                            r_vals.append(r_vals[0])
                            theta_labels.append(theta_labels[0])

                            color_choice = colors[idx % len(colors)]
                            radar_fig.add_trace(go.Scatterpolar(
                                r=r_vals,
                                theta=theta_labels,
                                fill='toself',
                                mode='lines+markers',
                                marker=dict(size=6),
                                name=f"#{model_detail.rank} {model_detail.model_name} ({model_detail.composite_score:.1f})",
                                line=dict(color=color_choice, width=2.5),
                                hovertemplate=(
                                    f"<b>{model_detail.model_name}</b> (Rank #{model_detail.rank})<br>"
                                    + "<b>Dimension:</b> %{theta}<br>"
                                    + "<b>Normalized Score:</b> %{r:.1f} / 100<extra></extra>"
                                ),
                            ))

                    radar_fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 105],
                                gridcolor="rgba(148, 163, 184, 0.18)",
                                color="#94A3B8",
                                tickfont=dict(size=11, color="#94A3B8"),
                            ),
                            angularaxis=dict(
                                gridcolor="rgba(148, 163, 184, 0.18)",
                                color="#F8FAFC",
                                tickfont=dict(size=12, color="#F8FAFC", family="Inter, sans-serif"),
                            ),
                            bgcolor="#0B132B",
                        ),
                        paper_bgcolor="#0F172A",
                        font=dict(color="#E2E8F0", family="Inter, sans-serif"),
                        title=dict(
                            text=f"Multi-Criteria Trade-Off Profile ({report.profile_name})",
                            font=dict(color="#FFFFFF", size=14),
                        ),
                        legend=dict(
                            bgcolor="rgba(15, 23, 42, 0.85)",
                            bordercolor="rgba(148, 163, 184, 0.2)",
                            borderwidth=1,
                            font=dict(color="#CBD5E1", size=12),
                        ),
                        margin=dict(l=45, r=45, t=55, b=45),
                    )
                    st.plotly_chart(radar_fig, use_container_width=True)

                    st.markdown(
                        """
                        <div class="notice-panel" style="margin-top: 0.8rem; padding: 1rem 1.25rem;">
                            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 6px;">
                                <span>🕸️</span>
                                <span>How to Read the Multi-Criteria Radar Profile</span>
                            </div>
                            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.55; margin: 0;">
                                Each axis represents an evaluation criterion normalized from 0 (lowest observed in candidate pool) to 100 (optimal observed).
                                A larger, symmetrical polygon signifies an algorithm with strong all-around balance across accuracy, latency, and memory.
                                Spikes toward outer rings reveal distinct algorithmic superpowers (e.g. extreme training speed in tree models or high accuracy in ensembles).
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with r_tab2:
                if not HAS_PLOTLY:
                    st.info("Interactive scatter plot requires Plotly.")
                else:
                    pareto_df_records = []
                    for m in report.ranked_models:
                        if m.status == "SUCCESS":
                            eff_score = np.mean([
                                m.normalized_scores.get("train_time", 0.0),
                                m.normalized_scores.get("inference_latency", 0.0),
                                m.normalized_scores.get("ram", 0.0),
                            ]) * 100.0
                            pareto_df_records.append({
                                "Model": m.model_name,
                                "Predictive Utility (%)": m.normalized_scores.get("performance", 0.0) * 100.0,
                                "System Efficiency Index (%)": eff_score,
                                "EcoML Score": m.composite_score,
                                "Classification": "⭐ Pareto Optimal" if m.is_pareto_optimal else "Dominated / Sub-Optimal",
                                "Peak RAM (MB)": m.raw_metrics.get("ram_current_mb", 10.0),
                            })
                    pareto_plot_df = pd.DataFrame(pareto_df_records)

                    fig_pareto = px.scatter(
                        pareto_plot_df,
                        x="System Efficiency Index (%)",
                        y="Predictive Utility (%)",
                        color="Classification",
                        symbol="Classification",
                        size="Peak RAM (MB)",
                        hover_name="Model",
                        hover_data={
                            "Predictive Utility (%)": ":.1f",
                            "System Efficiency Index (%)": ":.1f",
                            "EcoML Score": ":.1f",
                            "Peak RAM (MB)": ":.1f",
                            "Classification": True,
                        },
                        color_discrete_map={
                            "⭐ Pareto Optimal": THEME_COLORS["eco_green"],
                            "Dominated / Sub-Optimal": "#94A3B8",
                        },
                    )
                    fig_pareto.update_traces(
                        marker=dict(
                            line=dict(width=1.5, color="#FFFFFF"),
                            opacity=0.9,
                        ),
                        hovertemplate=(
                            "<b>%{hovertext}</b><br><br>"
                            "• <b>Predictive Utility:</b> %{y:.1f}%<br>"
                            "• <b>System Efficiency Index:</b> %{x:.1f}%<br>"
                            "• <b>EcoML Score:</b> %{customdata[2]:.1f} / 100<br>"
                            "• <b>Peak Process RAM:</b> %{customdata[3]:.1f} MB<br>"
                            "• <b>Frontier Status:</b> %{customdata[4]}<extra></extra>"
                        ),
                    )
                    fig_pareto = format_plotly_theme(
                        fig_pareto,
                        "Dual-Axis Pareto Frontier: Predictive Utility vs. System Efficiency Index",
                    )
                    fig_pareto.update_xaxes(
                        range=[0, 105],
                        title=dict(text="System Efficiency Index (%) → [Faster & Lighter]", font=dict(size=12, color="#E2E8F0")),
                        tickfont=dict(size=11),
                    )
                    fig_pareto.update_yaxes(
                        range=[0, 105],
                        title=dict(text="Predictive Utility (%) → [More Accurate]", font=dict(size=12, color="#E2E8F0")),
                        tickfont=dict(size=11),
                    )
                    st.plotly_chart(fig_pareto, use_container_width=True)

                    st.markdown(
                        """
                        <div class="notice-panel" style="margin-top: 0.8rem; padding: 1rem 1.25rem;">
                            <div style="font-weight: 700; color: #FFFFFF; font-size: 0.92rem; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 6px;">
                                <span>🎯</span>
                                <span>How to Read the Dual-Axis Pareto Frontier</span>
                            </div>
                            <p style="color: #94A3B8; font-size: 0.85rem; line-height: 1.55; margin: 0;">
                                The <b>Y-axis (Predictive Utility)</b> represents task accuracy (F1-score for classification, R² for regression).
                                The <b>X-axis (System Efficiency Index)</b> represents overall resource efficiency (faster training, lower latency, and lower RAM).
                                <b>⭐ Pareto-Optimal Models</b> (emerald markers in the upper-right region) dominate the candidate pool: no competing algorithm achieves both higher accuracy and lower resource cost.
                                Any model on this frontier represents an empirically defensible deployment choice depending on your latency or accuracy budgets.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with r_tab3:
                st.markdown("##### 🔬 Deep Telemetry & System Diagnostics")
                st.caption(
                    "Observed via `psutil` background thread sampling (20ms polling interval) "
                    "measuring baseline memory, peak resident memory, memory delta, CPU utilization, and sampling overhead."
                )

                deep_diag_rows = []
                for m in suite_res.models_evaluated:
                    if m.status == "SUCCESS":
                        deep_diag_rows.append({
                            "Model": m.model_name,
                            "Baseline RAM (MB)": m.baseline_ram_mb,
                            "Peak Process RAM (MB)": m.ram_current_mb,
                            "RAM Delta (MB)": m.ram_delta_mb,
                            "Avg CPU (%)": m.cpu_percent_avg,
                            "Peak CPU (%)": m.cpu_percent_peak,
                            "Sampling Overhead (ms)": m.telemetry_overhead_ms,
                            "Energy Telemetry": m.energy_status,
                        })
                st.dataframe(pd.DataFrame(deep_diag_rows), use_container_width=True)

                st.markdown(
                    f"""
                    <div class="notice-panel">
                        <div class="notice-title">
                            <span>🛡️</span>
                            <span>Measurement Methodology & Limitations</span>
                        </div>
                        <ul style="color: {THEME_COLORS['text_secondary']}; font-size: 0.88rem; margin: 0; padding-left: 1.2rem; line-height: 1.6;">
                            <li><strong>Memory Measurement:</strong> Queried via <code>psutil.Process().memory_info().rss</code>. Reflects Resident Set Size (physical memory occupied by the process), not virtual memory allocation.</li>
                            <li><strong>Sampling Overhead:</strong> Background polling at 20ms intervals introduces minimal overhead (typically &lt; 0.5ms total measurement latency), safely separated from model fitting duration.</li>
                            <li><strong>CodeCarbon Tracking:</strong> Optional environmental telemetry relies on physical RAPL/NVML hardware counters or TDP models. Short runs (&lt; 1s) report <em>Unavailable</em> to avoid synthetic inaccuracies.</li>
                            <li><strong>Pareto Dominance:</strong> Identifies models where no other candidate is simultaneously more accurate and more efficient. Models on the Pareto frontier are all scientifically defensible choices.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------------------------
# SECTION 5: PROJECT ARCHITECTURE & METHODOLOGY
# ------------------------------------------------------------------------------
with tab_methodology:
    st.markdown("### 📐 Project Architecture & Methodology")
    st.caption("Academic context, engineering milestones, and scientific evaluation foundations.")

    st.markdown("#### 🎯 The Dual-Axis Evaluation Paradigm")
    st.markdown(
        """
        Modern machine learning algorithms are traditionally selected based purely on predictive performance
        (such as accuracy, F1-score, or R²), often overlooking the computational energy, physical carbon footprint,
        and memory demands of training and production deployment.

        **EcoML bridges this gap.** It evaluates candidate machine learning models across a dual axis:
        1. **Predictive Utility:** Ensuring task accuracy, precision, recall, and reliability across classification and regression tasks.
        2. **Environmental & System Efficiency:** Quantifying electrical energy (kWh), carbon emissions (gCO₂eq),
           resident memory overhead (RAM), and inference latency.
        """
    )

    st.markdown("---")
    st.markdown("#### 🗺️ Implementation Roadmap & Milestone Status")
    st.caption("Current progress across project engineering milestones from foundation to green model selection:")

    for step in ROADMAP_STEPS:
        is_done = step["status"] == "Completed"
        card_class = "roadmap-card-done" if is_done else "roadmap-card-planned"
        badge_class = "pill-green" if is_done else "pill-sky"
        status_label = "✅ Completed" if is_done else "🔮 Planned Future Stage"

        st.markdown(
            f"""
            <div class="roadmap-card {card_class}">
                <div style="flex: 1; min-width: 280px;">
                    <div style="color: #FFFFFF; font-weight: 700; font-size: 0.96rem; margin-bottom: 2px;">
                        {step['step']} — {step['title']}
                    </div>
                    <div style="color: {THEME_COLORS['text_secondary']}; font-size: 0.85rem; line-height: 1.4;">
                        {step['summary']}
                    </div>
                </div>
                <div class="pill-badge {badge_class}">
                    {status_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### ✅ Implemented Core Capabilities")
    st.caption("Active operational pillars currently running within the EcoML experimentation pipeline:")

    col1, col2, col3 = st.columns(3)
    columns_impl = [col1, col2, col3]

    for idx, feature in enumerate(PLANNED_FEATURES):
        target_col = columns_impl[idx % 3]
        with target_col:
            st.markdown(
                f"""
                <div class="arch-card">
                    <div>
                        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">{feature['icon']}</div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 0.3rem;">{feature['title']}</div>
                        <div style="color: {THEME_COLORS['text_secondary']}; font-size: 0.84rem; line-height: 1.45; margin-bottom: 0.8rem;">{feature['description']}</div>
                    </div>
                    <span class="pill-badge pill-green" style="align-self: flex-start;">{feature['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown("---")
    st.markdown("#### 🔮 Future Evolution Roadmap (Post-Step 4)")
    st.caption("Advanced sustainability and deployment capabilities planned for upcoming research extensions:")

    fcol1, fcol2 = st.columns(2)
    columns_future = [fcol1, fcol2]

    for idx, feature in enumerate(FUTURE_ROADMAP_FEATURES):
        target_col = columns_future[idx % 2]
        with target_col:
            st.markdown(
                f"""
                <div class="arch-card">
                    <div>
                        <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">{feature['icon']}</div>
                        <div style="font-weight: 600; color: #FFFFFF; font-size: 0.96rem; margin-bottom: 0.3rem;">{feature['title']}</div>
                        <div style="color: {THEME_COLORS['text_secondary']}; font-size: 0.84rem; line-height: 1.45; margin-bottom: 0.8rem;">{feature['description']}</div>
                    </div>
                    <span class="pill-badge pill-sky" style="align-self: flex-start;">{feature['status']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown(
        f"""
        <div class="notice-panel">
            <div class="notice-title">
                <span>🛡️</span>
                <span>Data & Experimental Integrity Notice</span>
            </div>
            <p style="color: {THEME_COLORS['text_secondary']}; margin-bottom: 0.5rem; font-size: 0.88rem; line-height: 1.55;">
                In accordance with strict scientific and academic standards, <strong>EcoML does not simulate or fabricate
                any model scores, energy measurements, or carbon figures</strong>.
            </p>
            <p style="color: {THEME_COLORS['text_secondary']}; margin: 0; font-size: 0.88rem; line-height: 1.55;">
                Empirical benchmarks, psutil telemetry, CodeCarbon tracking, and MCDM recommendations are fully operational.
                All recorded metrics reflect genuine measurements captured on your host computing hardware.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
