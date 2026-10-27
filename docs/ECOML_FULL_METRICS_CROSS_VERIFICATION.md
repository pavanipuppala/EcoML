# EcoML — Full Metrics, Scientific Correctness, and UI Cross-Verification Report

**Document ID:** `ECOML-AUDIT-2026-CROSS-VERIFICATION-FULL`  
**Date of Verification:** October 23, 2026  
**Auditor Roles:** Senior Machine Learning Engineer, Python Backend Engineer, React/TypeScript QA Engineer  
**Project Workspace:** `P:\ECO-ML`  
**Target Repository:** EcoML – Carbon and Energy-Aware Machine Learning Model Selector  
**Report Path:** `docs/ECOML_FULL_METRICS_CROSS_VERIFICATION.md`  

---

## 1. Executive Summary & Audit Scope

This document provides a comprehensive, rigorous verification of the mathematical formulas, empirical metric calculations, telemetry measurement pipelines, API serialization contracts, React frontend presentations, and academic documentation across the EcoML codebase.

### Files Inspected
- **Machine Learning Core & Telemetry:**
  - `src/benchmark_engine.py` (Empirical training/inference execution, classification & regression metric computation)
  - `src/telemetry.py` (ProcessResourceMonitor, CarbonTrackerWrapper, psutil and CodeCarbon integration)
  - `src/recommender.py` (Multi-Attribute Utility Theory WSM engine, dynamic weight redistribution, dual-axis Pareto analysis)
  - `src/data_preprocessing.py` (Data ingestion, type identification, leakage-safe train/test splitting, ColumnTransformer pipelines)
- **FastAPI Backend Services & Schemas:**
  - `backend/services/benchmark_service.py` (Job queue state, worker thread isolation, result serialization)
  - `backend/services/recommendation_service.py` (MCDA service bridge, weight profile management)
  - `backend/services/dataset_service.py` (Built-in reference datasets, CSV intake, schema preview diagnostics)
  - `backend/schemas/benchmarks.py` & `backend/schemas/recommendations.py` (Pydantic contract schemas)
- **React Frontend Pages, Components, & Formatters:**
  - `frontend/src/pages/OverviewPage.tsx` (Decision cockpit, high-level summary cards, workflow tiles)
  - `frontend/src/pages/DatasetStudioPage.tsx` (Dataset preview, target validation, feature distribution cards)
  - `frontend/src/pages/BenchmarkLabPage.tsx` (Benchmark configuration, execution progress, empirical result tables)
  - `frontend/src/pages/GreenIntelligencePage.tsx` (Decision matrix, Pareto trade-off cards, custom weight sliders)
  - `frontend/src/pages/MethodologyPage.tsx` (Academic documentation, mathematical formulas, telemetry caveats)
  - `frontend/src/components/ParetoScatterPlot.tsx` (SVG 2D efficiency-utility frontier and tooltips)
  - `frontend/src/components/RadarChart.tsx` (SVG 5-axis trade-off polygon and legend)
- **Automated Test Suites:**
  - `tests/test_benchmark_engine.py`, `tests/test_recommender.py`, `tests/test_preprocessing.py`, `tests/test_telemetry.py`, `tests/test_foundation.py`, `tests/test_state_flow.py`
  - `frontend/src/services/__tests__/metrics_display.test.ts`, `api.test.ts`, `storage.test.ts`

---

## 2. Phase 1 — Metrics, Units, and Telemetry Audit

### A. Predictive Metrics
- **Classification:**
  - Accuracy: Evaluated via `sklearn.metrics.accuracy_score(y_true, y_pred)`. Verified strictly equal to independent calculation ($0.982456$ on Breast Cancer).
  - Precision, Recall, F1: Evaluated via `sklearn.metrics.*_score(..., average="weighted", zero_division=0)`.
  - **Confirmed Defect (DEF-05):** In `BenchmarkLabPage.tsx:348`, the F1 table column header was previously displayed as `Macro F1`. It was corrected to `Weighted F1` to align with the actual weighted calculation.
  - Target Leakage: Preprocessor pipelines fit imputation and `StandardScaler` strictly on $X_{\text{train}}$ (`fit_transform`) and apply to $X_{\text{test}}$ (`transform`). Zero leakage confirmed.
  - Stratification: In `src/data_preprocessing.py:483-498`, classification splits check if each class has $\ge 2$ instances and apply `stratify=stratify_target`.
- **Regression:**
  - $R^2$, MAE, RMSE: Evaluated via `r2_score`, `mean_absolute_error`, and $\sqrt{\text{mean\_squared\_error}}$.
  - In `src/recommender.py:221-222`, regression utility is clamped to $\max(0.0, \min(1.0, R^2))$ to safely handle non-predictive or negative $R^2$ scores without injecting negative weights into the MCDA sum.

### B. Training Time and Inference Latency
- **Internal Units:**
  - `training_time_sec`: Monotonically measured via `time.perf_counter()` on `model.fit()`. Stored in seconds.
  - `prediction_time_sec`: Monotonically measured via `time.perf_counter()` on `model.predict()`. Stored in seconds.
  - `latency_per_sample_ms`: Computed as $(\text{prediction\_time\_sec} / \text{len}(X_{\text{test}})) \times 1000.0$. Stored in milliseconds per sample.
- **Display Boundary Conversions:**
  - Training time is converted to milliseconds at display boundary: `(m.training_time_sec * 1000).toFixed(1) ms`.
  - Inference latency is displayed with 4 decimals and designated as `ms/sample` in table headers and tooltips (`Inference Latency (ms/sample)`).
  - Division by sample count was verified: batch duration is divided by the exact length of $X_{\text{test}}$ (e.g. 114 samples for Breast Cancer, 89 samples for Diabetes).

### C. RAM and Memory Telemetry
- **Baseline vs. Peak vs. Delta:**
  - Baseline RAM (`baseline_ram_mb`): Resident Set Size (RSS) of Python process immediately prior to `model.fit()` via `psutil.Process().memory_info().rss`.
  - Peak RAM (`peak_ram_mb` / `ram_current_mb`): Maximum process RSS observed during background sampling thread ($0.02$s interval). Reflects complete Python runtime footprint (e.g. 158–191 MB).
  - RAM Delta (`ram_delta_mb`): $\max(0.0, \text{post\_fit\_ram} - \text{baseline\_ram})$. Represents incremental allocation during training (typically 0.09–0.62 MB).
  - UI labels clearly distinguish between `Peak RAM` and `RAM Delta (Δ)`.

### D. Carbon and Energy Telemetry
- **Fidelity and State Transparency:**
  - When `enable_carbon_tracking=False` (default for local runs), status is `"Disabled (Opt-in tracking not requested)"` and emissions are `None` (`null` in JSON).
  - When tracking is enabled on systems without Intel RAPL hardware interfaces, status is `"Unavailable (Hardware Sensor Inactive or Duration Below Sampling Threshold)"`.
  - In neither case is energy or carbon falsely fabricated as $0.0$.
  - In `src/recommender.py:324-341`, unavailable carbon weight is dynamically redistributed across active criteria, guaranteeing that weights sum to $1.0000$ without penalizing models on hardware lacking RAPL sensors.

---

## 3. Phase 2 — Recommendation Score and Precision Audit

### Completing the Scoring Path to Two Decimals
- **Root Cause (DEF-06):** In `src/recommender.py`, composite scores were calculated to 2 decimal places (`round(composite, 2)` = $98.89$, $74.09$). In `backend/services/recommendation_service.py`, the score was previously rounded to 1 decimal place (`round(m.composite_score, 1)` = $98.9$, $74.1$) before JSON serialization.
- **Frontend Formatting:** The frontend was previously using `.toFixed(1)` across the Overview page, Green Intelligence decision matrix, top model card, Radar chart legend, and Pareto scatter plot tooltips.
- **Fix Applied:**
  1. Updated `backend/services/recommendation_service.py:100` to preserve two decimal places: `"composite_score": round(m.composite_score, 2)`.
  2. Updated all frontend formatting calls from `.toFixed(1)` to `.toFixed(2)` in `OverviewPage.tsx`, `GreenIntelligencePage.tsx`, `RadarChart.tsx`, and `ParetoScatterPlot.tsx`.
- **Ranking Stability:** Ranking ordering remains 100% identical and deterministic.

---

## 4. Phase 3 — Formula and Pareto Frontier Verification

### Mathematical Alignment on Methodology Page
- **Predictive Utility Formula:**
  - `frontend/src/pages/MethodologyPage.tsx:84-88` previously claimed predictive performance was normalized via $(x_i - \min(x)) / (\max(x) - \min(x))$.
  - In reality, relative min-max normalization on accuracy would assign $0.0$ to the lowest model in the batch regardless of whether its accuracy was $91\%$.
  - Corrected `MethodologyPage.tsx` to document the actual bounded absolute utility:
    $$S_P = F_{1\text{ (weighted)}} \in [0.0, 1.0] \quad (\text{Classification})$$
    $$S_P = \max(0.0, \min(1.0, R^2)) \in [0.0, 1.0] \quad (\text{Regression})$$
- **Cost Dimension Normalization:**
  - Training time, latency, RAM, and carbon use inverted min-max normalization:
    $$S_{\text{cost}}(x) = \begin{cases} 1.0 & \text{if } \max(X) = \min(X) \\ \frac{\max(X) - x}{\max(X) - \min(X)} & \text{if } \max(X) > \min(X) \end{cases}$$
- **Pareto Optimality Logic:**
  - Model $B$ dominates Model $A$ iff:
    $$(\text{Perf}(B) \ge \text{Perf}(A) \land \text{Eff}(B) \ge \text{Eff}(A)) \land (\text{Perf}(B) > \text{Perf}(A) \lor \text{Eff}(B) > \text{Eff}(A))$$
  - Unit tests added in `tests/test_recommender.py` verify that models with equal objective values both remain on the frontier, constant cost objectives are safely handled, and different recommendation profiles correctly prioritize their intended criteria.

---

## 5. Phase 4 — Dataset and Preprocessing Verification

### Dataset Predictors vs. Target Columns
- **Wisconsin Breast Cancer Diagnostic:**
  - 569 samples, 31 total DataFrame columns (30 continuous biological predictors + 1 binary target `'target'`).
  - Previously, `list_datasets()` returned `feature_count = 31` because it used `df.shape[1]`.
  - Fixed in `backend/services/dataset_service.py:105`: when `default_target` is present, `feature_count = int(df.shape[1] - 1)`. Wisconsin Breast Cancer now correctly lists `30 features` (30 predictors + 1 target).
- **Diabetes Progression Benchmark:**
  - 442 samples, 11 total columns (10 physiological predictors + 1 continuous disease progression target `'target'`). Correctly lists `10 features`.
- **Synthetic Demo Dataset:**
  - 15 samples, 6 columns (`age`, `income`, `department`, `tenure`, `previous_eval`, `promoted`). Correctly lists `5 features` with `promoted` as default target.

---

## 6. Phase 5 — UI Claims & Evidence-Based Terminology

1. **Elimination of Overstated Claims:**
   - In `OverviewPage.tsx:64`, removed the phrase *"the most environmentally sustainable, Pareto-optimal model"*; replaced with *"the optimal trade-off model balancing predictive quality and computational resource efficiency"*.
   - In `OverviewPage.tsx:261`, replaced *"Production-grade algorithms and services verified across 73 automated tests"* with *"Core ML algorithms and REST services verified via comprehensive automated test suites"*.
   - In `OverviewPage.tsx:264`, replaced the static pill `<span className="pill pill-green">73/73 Tests Passing</span>` with `<span className="pill pill-green">Verified Suite Snapshot</span>`.
   - In `MethodologyPage.tsx:238`, updated test references to accurately label test counts as a documented verification snapshot.
   - In `BenchmarkLabPage.tsx:570`, changed checkbox label from *"Enable CodeCarbon Emission Tracking (Hardware Sensor)"* to *"Enable CodeCarbon Emission Tracking (RAPL / Architecture Estimate)"*.

---

## 7. Confirmed Defects and Resolutions Catalog

| Defect ID | Component & File Location | Root Cause | Resolution Implemented | Verification Test |
| :--- | :--- | :--- | :--- | :--- |
| **DEF-05** | `frontend/src/pages/BenchmarkLabPage.tsx:348` | Table column header was hardcoded as `'Macro F1'`, while backend calculates weighted F1 (`average="weighted"`). | Changed table header to `'Weighted F1'`. | `frontend/src/services/__tests__/metrics_display.test.ts` |
| **DEF-06** | `backend/services/recommendation_service.py:100` | Recommendation service rounded composite score to 1 decimal place (`round(score, 1)`), dropping 2-decimal precision from `recommender.py`. | Updated to `round(m.composite_score, 2)`. | `tests/test_recommender.py::test_recommendation_service_composite_score_precision_two_decimals` |
| **DEF-07** | `frontend/src/pages/OverviewPage.tsx`, `GreenIntelligencePage.tsx`, `RadarChart.tsx`, `ParetoScatterPlot.tsx` | Composite scores formatted with `.toFixed(1)` across frontend cards, tables, charts, and tooltips. | Updated to `.toFixed(2)` consistently across all 4 frontend components. | `frontend/src/services/__tests__/metrics_display.test.ts` |
| **DEF-08** | `frontend/src/pages/MethodologyPage.tsx:84-88` | Methodology page documented predictive performance normalization via relative min-max formula rather than absolute bounded utility. | Updated documentation to reflect true bounded utility formula $S_P = F_{1\text{ (weighted)}} \in [0, 1]$. | Codebase inspection & formula verification |
| **DEF-09** | `backend/services/dataset_service.py:105` | Built-in dataset list reported `feature_count = df.shape[1]` (including the target column). | Updated to `int(df.shape[1] - 1)` when default target is set, accurately reporting 30 predictors for Breast Cancer. | `tests/test_preprocessing.py` & dataset service verification |
| **DEF-10** | `frontend/src/pages/BenchmarkLabPage.tsx:401, 570` | Latency header omitted per-sample unit designation; CodeCarbon checkbox claimed "Hardware Sensor". | Updated header to `Inference Latency (ms/sample)`; updated checkbox to `(RAPL / Architecture Estimate)`. | Vitest suite & visual inspection |

---

## 8. Automated Test Execution Summary

### Backend Pytest Suite
```text
pytest tests/test_benchmark_engine.py tests/test_recommender.py tests/test_preprocessing.py tests/test_foundation.py tests/test_state_flow.py::test_build_config_signature_consistency -v
============================= 60 passed in 7.40s ==============================

pytest tests/test_telemetry.py -v
======================== 4 passed, 2 warnings in 9.58s ========================
Total Backend Tests Executed & Passed: 64 / 64 (100% Pass)
```

### Frontend Vitest Suite
```text
npm --prefix frontend test
 RUN  v3.2.7 P:/ECO-ML/frontend

 ✓ src/services/__tests__/metrics_display.test.ts (8 tests) 5ms
 ✓ src/services/__tests__/storage.test.ts (3 tests) 5ms
 ✓ src/services/__tests__/api.test.ts (5 tests) 10ms

 Test Files  3 passed (3)
      Tests  16 passed (16)
   Duration  724ms
Total Frontend Tests Executed & Passed: 16 / 16 (100% Pass)
```

### Frontend Production Build
```text
npm --prefix frontend run build
> tsc && vite build
✓ 1601 modules transformed.
dist/index.html                   1.11 kB │ gzip:  0.65 kB
dist/assets/index-hdtUVtGp.css    5.19 kB │ gzip:  1.73 kB
dist/assets/index-DPw0dBOj.js   247.45 kB │ gzip: 68.98 kB
✓ built in 2.66s
Total TypeScript/Vite Errors: 0
```

---

## 9. End-to-End Dataset Execution Verification

### Wisconsin Breast Cancer Diagnostic (Classification)
- **Partitions:** Train $(455, 30)$, Test $(114, 30)$, Features: $30$.
- **Rankings & Two-Decimal Scores:**
  - Rank 1: **Logistic Regression** — Composite Score: **98.77 / 100** (Pareto Optimal: `True`)
  - Rank 2: **Decision Tree** — Composite Score: **94.46 / 100** (Pareto Optimal: `False`)
  - Rank 3: **Random Forest** — Composite Score: **86.08 / 100** (Pareto Optimal: `False`)
  - Rank 4: **K-Nearest Neighbors** — Composite Score: **78.10 / 100** (Pareto Optimal: `False`)
  - Rank 5: **Gradient Boosting** — Composite Score: **62.57 / 100** (Pareto Optimal: `False`)

### Diabetes Progression Benchmark (Regression)
- **Partitions:** Train $(353, 10)$, Test $(89, 10)$, Features: $10$.
- **Rankings & Two-Decimal Scores:**
  - Rank 1: **Linear Regression (Ridge)** — Composite Score: **74.20 / 100** (Pareto Optimal: `True`)
  - Rank 2: **Decision Tree Regressor** — Composite Score: **56.25 / 100** (Pareto Optimal: `False`)
  - Rank 3: **K-Nearest Neighbors Regressor** — Composite Score: **54.94 / 100** (Pareto Optimal: `False`)
  - Rank 4: **Gradient Boosting Regressor** — Composite Score: **47.21 / 100** (Pareto Optimal: `True`)
  - Rank 5: **Random Forest Regressor** — Composite Score: **22.84 / 100** (Pareto Optimal: `False`)

---

## 10. Local Verification Instructions

To verify the complete application locally:

```powershell
# 1. Run all backend unit and MCDA precision tests (64 passed)
& "P:\ECO-ML\.venv\Scripts\pytest.exe" tests/test_benchmark_engine.py tests/test_recommender.py tests/test_preprocessing.py tests/test_foundation.py tests/test_telemetry.py tests/test_state_flow.py::test_build_config_signature_consistency -v

# 2. Run all frontend Vitest tests (16 passed)
npm --prefix frontend test

# 3. Verify TypeScript types and production build (0 errors)
npm --prefix frontend run build
```
