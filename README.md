# 🌱 EcoML — Carbon and Energy-Aware Machine Learning Model Selector

> An intelligent, eco-conscious framework for evaluating machine learning algorithms across predictive accuracy, energy consumption, carbon emissions, and system efficiency.

---

## 📌 Project Overview

Traditional machine learning evaluation focuses almost exclusively on predictive performance metrics such as Accuracy, Precision, Recall, and F1-score. However, as artificial intelligence models grow in ubiquity, their cumulative computational demand, electricity consumption, and environmental carbon footprint have become critical engineering concerns.

**EcoML** is designed to bridge the gap between model accuracy and environmental sustainability ("Green AI"). Developed as a B.Tech final-year Computer Science and Engineering (Artificial Intelligence & Machine Learning) project, EcoML provides an empirical benchmarking platform that balances task performance against real-world computational costs.

---

## 🎯 Key Objectives

1. **Dual-Axis Evaluation:** Assess machine learning models across both predictive utility (accuracy, F1-score) and ecological cost (energy, emissions).
2. **Transparent System Telemetry:** Measure hardware resource consumption—including peak memory (RAM) and execution duration—using operating system sensors.
3. **Hardware-Aware Carbon Tracking:** Estimate real-world electrical consumption (kWh) and equivalent carbon emissions ($g\text{CO}_2\text{eq}$) using CodeCarbon and regional grid emission factors.
4. **Multi-Criteria Model Selection:** Formulate an objective scoring mechanism to recommend models that achieve the highest accuracy per unit of energy expended.
5. **Interactive Visualization:** Provide an accessible, user-friendly Streamlit web interface for model comparison and reporting.

---

## 🚀 Implemented Capabilities (Steps 2–4)

- **🎯 Predictive Performance Comparison (Step 3):** Standard evaluation metrics across classification (Accuracy, Precision, Recall, F1-Score) and regression (MAE, RMSE, R²).
- **⏱️ High-Resolution Execution Benchmarking (Step 3):** Granular profiling of model training duration and per-sample inference latency via `time.perf_counter()`.
- **💾 Process Memory Telemetry (Step 4):** Non-blocking background thread sampling baseline RAM, peak resident set size (RSS), and memory delta via `psutil`.
- **⚡ Hardware Energy Consumption Tracking (Step 4):** Quantification of computational electrical energy in kilowatt-hours (kWh) via `CodeCarbon`.
- **🌱 Carbon Emission Estimation (Step 4):** Real-world regional grid emissions tracking ($g\text{CO}_2\text{eq}$) with transparent hardware fallback when sensors are unavailable.
- **🏆 Multi-Criteria Green Model Recommender (Step 4):** Weighted Sum Model (WSM) scoring with customizable profiles (Balanced, Green-First, Accuracy-First, Edge) and dual-axis Pareto frontier trade-off analysis.

---

## 🔮 Future Evolution Roadmap (Post-Step 4)

- **🎛️ Green Hyperparameter Eco-Tuning:** Energy-constrained Bayesian optimization that halts search when accuracy gains plateau relative to electrical power consumed.
- **⚡ Deep Learning & GPU Telemetry:** NVML power telemetry and FLOPs-per-watt profiling for PyTorch and TensorFlow deep neural architectures.
- **🌐 Carbon-Aware Cloud Scheduler:** Automated training dispatch targeting cloud regions and time windows with peak renewable grid generation.
- **📦 Model Quantization & Export:** Post-training 8-bit integer quantization and ONNX export for low-power edge deployment.

---

## 📁 Project Structure

```text
ECO-ML/
├── app.py                  # Main Streamlit web application entry point
├── requirements.txt        # Project dependencies and libraries
├── README.md               # Comprehensive documentation and setup guide
├── config/
│   └── settings.py         # Centralized configuration, paths, and constants
├── data/
│   ├── raw/                # Unprocessed source datasets
│   └── processed/          # Cleaned, encoded, and preprocessed datasets
├── src/
│   ├── __init__.py         # Source package initialization
│   ├── data_preprocessing.py # Dataset ingestion, validation, and preprocessing
│   ├── benchmark_engine.py # Model training, latency, and evaluation engine
│   ├── telemetry.py        # psutil memory/CPU profiling and CodeCarbon wrapper
│   └── recommender.py      # MCDM green scoring and Pareto frontier analyzer
├── experiments/            # Benchmark logs, intermediate runs, and tracking outputs
├── models/                 # Serialized, trained model artifacts
├── reports/                # Exportable summaries, figures, and comparison reports
└── tests/
    ├── __init__.py         # Test package initialization
    ├── test_foundation.py  # Unit tests verifying project setup and configuration
    ├── test_preprocessing.py # Unit tests for data preprocessing pipeline
    ├── test_benchmark_engine.py # Tests for model benchmarking engine
    ├── test_telemetry.py   # Tests for system memory and carbon telemetry
    ├── test_recommender.py # Tests for multi-criteria recommendation scoring
    └── test_state_flow.py  # Tests for cross-tab session state and invalidation
```

---

## 🛠️ Technology Stack

| Component | Library / Tool | Purpose |
| :--- | :--- | :--- |
| **Interactive UI** | `streamlit` | Modern, responsive web dashboard |
| **Data Processing** | `pandas`, `numpy` | Tabular data manipulation and numerical operations |
| **Machine Learning** | `scikit-learn` | Model architectures, cross-validation, and metrics |
| **System Telemetry** | `psutil` | Process-level RAM and CPU utilization profiling |
| **Carbon Tracking** | `codecarbon` | Real-time computation energy & emissions estimation |
| **Interactive Charts** | `plotly` | Dynamic multi-dimensional comparison charts |
| **Unit Testing** | `pytest` | Automated testing and verification |

---

## ⚙️ Getting Started (Local Setup)

Follow these step-by-step instructions to set up and run the project locally.

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.13)
- **pip** (Python package installer)
- Git (optional, for version control)

### 2. Open Terminal & Navigate to Project Directory
```bash
cd P:\ECO-ML
```

### 3. Create a Virtual Environment (Recommended)
Creating an isolated virtual environment ensures clean dependency management:

- **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
  *(If execution policies restrict script activation, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Verify Setup with Unit Tests
```bash
pytest
```

### 6. Launch the EcoML Web Application
```bash
streamlit run app.py
```
Once launched, open your web browser at the URL shown in the terminal (typically `http://localhost:8501`).

---

## 🛡️ Scientific Integrity Policy

EcoML is committed to empirical validity and reproducible science:
- **No Fabricated Data:** Model performance scores, resource numbers, energy consumption, and carbon estimates are never hardcoded or simulated.
- **Empirical Measurement:** All metrics will be captured dynamically during training and inference runs on your physical host machine.

---

## 🗺️ Development Roadmap

- [x] **Step 1: Project Foundation** — Directory layout, configuration module, dependency manifest, test suite, and landing dashboard.
- [x] **Step 2: Dataset Acquisition & Preprocessing** — Robust CSV ingestion, structural validation, exploratory summaries, median & most-frequent imputation, one-hot encoding, and leakage-free train/test partitioning.
- [x] **Step 3: Baseline ML Benchmark Engine** — Multi-model empirical benchmark suite for classification and regression, measuring predictive performance, high-resolution training/inference timing, process memory footprint, and isolated carbon telemetry.
- [x] **Step 4: Deep Resource Telemetry & Multi-Criteria Green Model Recommender** — High-precision psutil background polling (baseline, peak, delta RAM & CPU), transparent CodeCarbon integration, dynamic weight redistribution for missing metrics, dual-axis Pareto efficiency frontier, and multi-criteria decision making (WSM) across user profiles.
- [ ] **Step 5: Comparative Analysis & Interactive Dashboards** — Plotly trade-off curves (Accuracy vs. Energy / Carbon).
- [ ] **Step 6: Green Model Recommendation Engine** — Multi-objective optimization score calculation.

---

## 🎓 Academic Information
- **Degree:** B.Tech Computer Science and Engineering
- **Specialization:** Artificial Intelligence & Machine Learning
- **Project Domain:** Sustainable Computing / Green AI

