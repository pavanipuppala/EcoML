"""
EcoML — Project Settings and Configuration Constants.

This module centralizes all project configurations, directory paths,
theme color palettes, and evaluation parameter constants used across EcoML.
"""

from pathlib import Path

# ==============================================================================
# 1. Project Metadata
# ==============================================================================
PROJECT_NAME = "EcoML"
PROJECT_TAGLINE = "Carbon and Energy-Aware Machine Learning Model Selector"
PROJECT_VERSION = "0.4.0"
PROJECT_STATUS = "Step 4 Operational — Green Model Recommender Active"
PROJECT_DESCRIPTION = (
    "EcoML is an intelligent, sustainability-focused machine learning model "
    "selector that evaluates algorithms not only on predictive accuracy, but "
    "also on their real-world environmental footprint—measuring energy "
    "consumption, carbon emissions, memory footprint, and execution latency."
)

# ==============================================================================
# 2. Directory Paths Configuration
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# Core subdirectories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SRC_DIR = BASE_DIR / "src"
CONFIG_DIR = BASE_DIR / "config"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
TESTS_DIR = BASE_DIR / "tests"

# ==============================================================================
# 3. UI Theme & Color Palette (Green & Dark-Navy Theme)
# ==============================================================================
THEME_COLORS = {
    # Dark Navy Tones (Primary & Backgrounds)
    "navy_darkest": "#0B132B",   # Main body & deep container background
    "navy_dark": "#0F172A",      # Application background
    "navy_card": "#1E293B",      # Card & container background
    "navy_border": "#334155",    # Subtle borders & dividers
    
    # Eco Green Tones (Accents & Highlights)
    "eco_green": "#10B981",      # Emerald accent (brand primary)
    "eco_green_light": "#34D399",# Mint highlight
    "eco_green_dark": "#059669", # Deep emerald for hovers/badges
    "eco_green_bg": "#064E3B",   # Subtle green background tint
    
    # Text Tones
    "text_primary": "#F8FAFC",   # High-contrast white/slate text
    "text_secondary": "#94A3B8", # Muted slate text
    "text_accent": "#6EE7B7",    # Highlight text
    
    # Status Tones
    "status_ready": "#10B981",   # Green for "Ready"
    "status_planned": "#38BDF8", # Cyan for "Planned"
}

# ==============================================================================
# 4. Implemented Features, Future Roadmap & Implementation Steps
# ==============================================================================
PLANNED_FEATURES = [
    {
        "id": "model_accuracy",
        "title": "Model Accuracy Comparison",
        "icon": "🎯",
        "description": "Comprehensive predictive quality metrics: Accuracy, Precision, Recall, and F1-Score for classification; MAE, RMSE, and R² for regression.",
        "status": "Implemented (Step 3)",
        "stage": "Step 3 Benchmark Engine",
    },
    {
        "id": "execution_time",
        "title": "Execution Time Benchmarking",
        "icon": "⏱️",
        "description": "High-precision model training duration and per-sample inference latency benchmarking via time.perf_counter().",
        "status": "Implemented (Step 3)",
        "stage": "Step 3 Benchmark Engine",
    },
    {
        "id": "memory_usage",
        "title": "Memory Footprint Tracking",
        "icon": "💾",
        "description": "Process-level baseline RAM, peak resident set size (RSS), and memory delta observed via non-blocking background psutil sampling.",
        "status": "Implemented (Step 4)",
        "stage": "Step 4 Telemetry",
    },
    {
        "id": "energy_estimation",
        "title": "Energy Consumption Estimation",
        "icon": "⚡",
        "description": "Quantification of computational electricity consumption measured in kilowatt-hours (kWh) via CodeCarbon with hardware sensor fallback.",
        "status": "Implemented (Step 4)",
        "stage": "Step 4 Telemetry",
    },
    {
        "id": "carbon_emissions",
        "title": "Carbon Emission Estimation",
        "icon": "🌱",
        "description": "Hardware and regional grid carbon intensity-aware emissions tracking (gCO2eq) with transparent unavailable reporting on sub-second runs.",
        "status": "Implemented (Step 4)",
        "stage": "Step 4 Telemetry",
    },
    {
        "id": "model_recommendation",
        "title": "Efficient Model Recommendation",
        "icon": "🏆",
        "description": "Multi-Criteria Decision Making (MCDM) weighted scoring and dual-axis Pareto frontier analysis balancing predictive power against ecological cost.",
        "status": "Implemented (Step 4)",
        "stage": "Step 4 Recommender",
    },
]

FUTURE_ROADMAP_FEATURES = [
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
]

ROADMAP_STEPS = [
    {
        "step": "Step 1",
        "title": "Project Foundation & Architecture",
        "status": "Completed",
        "summary": "Modular directory structure, configuration constants, testing suite, and green/navy UI theme.",
    },
    {
        "step": "Step 2",
        "title": "Dataset Selection & Preprocessing",
        "status": "Completed",
        "summary": "Tabular CSV ingestion, dataset diagnostics, median/mode imputation, one-hot encoding, and leakage-free splitting.",
    },
    {
        "step": "Step 3",
        "title": "Empirical ML Benchmark Engine",
        "status": "Completed",
        "summary": "Multi-algorithm comparative evaluation for classification and regression with training duration and inference latency profiling.",
    },
    {
        "step": "Step 4",
        "title": "Deep Telemetry & Green Model Recommender",
        "status": "Completed",
        "summary": "psutil background memory tracking, CodeCarbon emissions estimation, MCDM weighting, and Pareto frontier trade-offs.",
    },
    {
        "step": "Step 5 (Future)",
        "title": "Green AutoML & Hardware Acceleration",
        "status": "Planned",
        "summary": "Energy-constrained hyperparameter tuning, GPU NVML integration, and edge quantization workflows.",
    },
]

# ==============================================================================
# 5. Default Reproducibility & System Constants
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE_DEFAULT = 0.2

# Explicit module export list
__all__ = [
    "PROJECT_NAME",
    "PROJECT_TAGLINE",
    "PROJECT_VERSION",
    "PROJECT_STATUS",
    "PROJECT_DESCRIPTION",
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "SRC_DIR",
    "CONFIG_DIR",
    "EXPERIMENTS_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "TESTS_DIR",
    "THEME_COLORS",
    "PLANNED_FEATURES",
    "FUTURE_ROADMAP_FEATURES",
    "ROADMAP_STEPS",
    "RANDOM_STATE",
    "TEST_SIZE_DEFAULT",
]


