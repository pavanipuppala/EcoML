"""
EcoML — Machine Learning Benchmark Engine.

This module executes empirical, leakage-free benchmarking across multiple machine learning
models on preprocessed dataset partitions. It evaluates predictive quality alongside
computational efficiency metrics (training time, inference latency, RAM footprint,
and optional environmental emission telemetry).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data_preprocessing import PreprocessingResult
from src.telemetry import ProcessResourceMonitor, CarbonTrackerWrapper, TelemetryCategory


@dataclass
class SingleModelBenchmark:
    """Stores benchmarking observations for an individual model run.

    Attributes:
        model_id: Programmatic identifier (e.g. 'random_forest').
        model_name: Human-readable display label.
        task_type: 'classification' or 'regression'.
        status: Execution status ('SUCCESS' or 'FAILED').
        error_message: Exception message if execution failed.
        accuracy: Classification accuracy score.
        precision: Classification precision (weighted average).
        recall: Classification recall (weighted average).
        f1: Classification F1-score (weighted average).
        confusion_matrix: Confusion matrix list of lists.
        mae: Regression Mean Absolute Error.
        rmse: Regression Root Mean Squared Error.
        r2: Regression R-squared coefficient of determination.
        training_time_sec: Total duration of model.fit() in seconds.
        prediction_time_sec: Total duration of model.predict() in seconds.
        latency_per_sample_ms: Average inference latency per sample in milliseconds.
        ram_current_mb: Resident process RAM observed post-training in megabytes.
        ram_delta_mb: Observed increment in resident RAM during training in megabytes.
        energy_kwh: Estimated electrical energy consumption in kilowatt-hours.
        carbon_emissions_g: Estimated carbon emissions in grams CO2 equivalent.
        energy_status: Status description of energy telemetry ('Measured', 'Estimated', 'Unavailable', 'Disabled').
        model_instance: The fitted scikit-learn model object.
    """
    model_id: str
    model_name: str
    task_type: str
    status: str = "SUCCESS"
    error_message: Optional[str] = None

    # Classification Metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Regression Metrics
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None

    # Computational Efficiency (Measured)
    training_time_sec: Optional[float] = None
    prediction_time_sec: Optional[float] = None
    latency_per_sample_ms: Optional[float] = None
    ram_current_mb: Optional[float] = None
    ram_delta_mb: Optional[float] = None
    baseline_ram_mb: Optional[float] = None
    cpu_percent_avg: Optional[float] = None
    cpu_percent_peak: Optional[float] = None
    telemetry_overhead_ms: Optional[float] = None

    # Energy & Carbon Telemetry (Estimated or Unavailable)
    energy_kwh: Optional[float] = None
    carbon_emissions_g: Optional[float] = None
    energy_status: str = "Disabled"
    model_instance: Optional[Any] = None


@dataclass
class BenchmarkSuiteResult:
    """Container aggregating multi-model benchmarking results for an experiment."""
    task_type: str
    dataset_name: str
    target_column: str
    sample_count_train: int
    sample_count_test: int
    feature_count: int
    models_evaluated: List[SingleModelBenchmark] = field(default_factory=list)
    benchmark_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    random_state: int = 42

    def to_dataframe(self) -> pd.DataFrame:
        """Convert benchmark records into a structured pandas DataFrame."""
        records = []
        for run in self.models_evaluated:
            record: Dict[str, Any] = {
                "Model": run.model_name,
                "Status": run.status,
            }
            if run.status == "SUCCESS":
                if self.task_type == "classification":
                    record["Accuracy"] = round(run.accuracy, 4) if run.accuracy is not None else None
                    record["Precision"] = round(run.precision, 4) if run.precision is not None else None
                    record["Recall"] = round(run.recall, 4) if run.recall is not None else None
                    record["F1-Score"] = round(run.f1, 4) if run.f1 is not None else None
                else:
                    record["MAE"] = round(run.mae, 4) if run.mae is not None else None
                    record["RMSE"] = round(run.rmse, 4) if run.rmse is not None else None
                    record["R² Score"] = round(run.r2, 4) if run.r2 is not None else None

                record["Training Time (s)"] = round(run.training_time_sec, 4) if run.training_time_sec is not None else None
                record["Inference Latency (ms)"] = round(run.latency_per_sample_ms, 4) if run.latency_per_sample_ms is not None else None
                record["Baseline RAM (MB)"] = run.baseline_ram_mb
                record["Peak Process RAM (MB)"] = run.ram_current_mb
                record["RAM Delta (MB)"] = run.ram_delta_mb
                record["Avg CPU (%)"] = run.cpu_percent_avg
                record["Peak CPU (%)"] = run.cpu_percent_peak
                record["Energy (kWh)"] = run.energy_kwh if run.energy_kwh is not None else "Unavailable"
                record["Carbon (gCO2eq)"] = run.carbon_emissions_g if run.carbon_emissions_g is not None else "Unavailable"
                record["Energy Telemetry Status"] = run.energy_status
                record["Telemetry Overhead (ms)"] = run.telemetry_overhead_ms
            else:
                record["Error Details"] = run.error_message or "Execution failed"
            records.append(record)
        return pd.DataFrame(records)


def get_supported_models(task_type: str) -> Dict[str, str]:
    """Retrieve supported baseline models for a specific machine learning task.

    Args:
        task_type: 'classification' or 'regression'.

    Returns:
        Dict[str, str]: Mapping from model identifier to display name.

    Raises:
        ValueError: If task_type is unsupported.
    """
    clean_task = task_type.strip().lower()
    if clean_task == "classification":
        return {
            "logistic_regression": "Logistic Regression",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "knn": "K-Nearest Neighbors",
            "gradient_boosting": "Gradient Boosting",
        }
    elif clean_task == "regression":
        return {
            "linear_regression": "Linear Regression (Ridge)",
            "decision_tree": "Decision Tree Regressor",
            "random_forest": "Random Forest Regressor",
            "knn": "K-Nearest Neighbors Regressor",
            "gradient_boosting": "Gradient Boosting Regressor",
        }
    else:
        raise ValueError(f"Unsupported task_type '{task_type}'. Must be 'classification' or 'regression'.")


def build_model_instance(model_id: str, task_type: str, random_state: int = 42) -> Any:
    """Instantiate a baseline model using consistent random seeds for fair comparison.

    Args:
        model_id: Identifier of the requested model.
        task_type: 'classification' or 'regression'.
        random_state: Reproducibility seed for stochastic algorithms.

    Returns:
        Fitted or un-fitted scikit-learn estimator instance.

    Raises:
        ValueError: If model_id is unrecognized for the given task.
    """
    clean_task = task_type.strip().lower()

    if clean_task == "classification":
        if model_id == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_id == "decision_tree":
            return DecisionTreeClassifier(random_state=random_state)
        elif model_id == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_id == "knn":
            return KNeighborsClassifier(n_neighbors=5)
        elif model_id == "gradient_boosting":
            return GradientBoostingClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unknown classification model ID '{model_id}'.")

    elif clean_task == "regression":
        if model_id == "linear_regression":
            return Ridge(alpha=1.0, random_state=random_state)
        elif model_id == "decision_tree":
            return DecisionTreeRegressor(random_state=random_state)
        elif model_id == "random_forest":
            return RandomForestRegressor(n_estimators=100, random_state=random_state)
        elif model_id == "knn":
            return KNeighborsRegressor(n_neighbors=5)
        elif model_id == "gradient_boosting":
            return GradientBoostingRegressor(random_state=random_state)
        else:
            raise ValueError(f"Unknown regression model ID '{model_id}'.")

    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
) -> Dict[str, Any]:
    """Compute task-appropriate predictive performance metrics with safe edge-case handling.

    Args:
        y_true: Ground truth target vector.
        y_pred: Predicted target vector from candidate model.
        task_type: 'classification' or 'regression'.

    Returns:
        Dict[str, Any]: Dictionary containing task-specific metric scores.

    Raises:
        ValueError: If array lengths mismatch or task_type is unrecognized.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatched lengths: y_true ({len(y_true)}) vs y_pred ({len(y_pred)}).")

    clean_task = task_type.strip().lower()

    if clean_task == "classification":
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred).tolist()
        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
        }

    elif clean_task == "regression":
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_true, y_pred))
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    else:
        raise ValueError(f"Unsupported task_type '{task_type}'.")


def benchmark_single_model(
    model_id: str,
    model_name: str,
    model_instance: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    track_emissions: bool = False,
    n_warmup: int = 1,
    n_repeats: int = 1,
) -> SingleModelBenchmark:
    """Execute training, timing, prediction, and resource measurement for a single model.

    Args:
        model_id: Programmatic model identifier.
        model_name: Human-readable model label.
        model_instance: Instantiated scikit-learn estimator.
        X_train: Preprocessed training features.
        y_train: Training target labels.
        X_test: Preprocessed testing features.
        y_test: Testing target labels.
        task_type: 'classification' or 'regression'.
        track_emissions: Whether to attempt hardware-level emission tracking via CodeCarbon.
        n_warmup: Number of initial inference warm-up iterations (default: 1).
        n_repeats: Number of timed inference passes to average (default: 1).

    Returns:
        SingleModelBenchmark: Populated benchmark record with predictive & efficiency metrics.
    """
    clean_task = task_type.strip().lower()

    # Start deep resource monitor and carbon tracker
    resource_monitor = ProcessResourceMonitor(sampling_interval_sec=0.02)
    carbon_tracker = CarbonTrackerWrapper(enabled=track_emissions)

    carbon_tracker.start()
    resource_monitor.start()

    # Step 1: Model Training with high-resolution timer
    try:
        t_start_train = time.perf_counter()
        model_instance.fit(X_train, y_train)
        training_time_sec = time.perf_counter() - t_start_train
    except Exception as exc:
        resource_monitor.stop()
        carbon_tracker.stop()
        return SingleModelBenchmark(
            model_id=model_id,
            model_name=model_name,
            task_type=clean_task,
            status="FAILED",
            error_message=f"Training error in {model_name}: {str(exc)}",
        )

    # Stop resource monitor and carbon tracker
    res_snap = resource_monitor.stop()
    carbon_snap = carbon_tracker.stop()

    ram_current_mb = round(res_snap.peak_ram_mb, 2)
    ram_delta_mb = round(res_snap.ram_delta_mb, 2)
    baseline_ram_mb = round(res_snap.baseline_ram_mb, 2)
    cpu_percent_avg = round(res_snap.avg_cpu_percent, 2)
    cpu_percent_peak = round(res_snap.peak_cpu_percent, 2)
    telemetry_overhead_ms = round(res_snap.overhead_ms, 3)

    energy_kwh = carbon_snap.energy_kwh
    carbon_emissions_g = carbon_snap.carbon_emissions_g
    energy_status = carbon_snap.status_message

    # Step 2: Prediction Latency Measurement with optional warm-up & repeat aggregation
    try:
        sample_count = max(1, len(X_test))
        # Warm-up pass to eliminate initial memory page fault & CPU clock ramp overhead
        if n_warmup > 0 and sample_count > 0:
            _ = model_instance.predict(X_test[:min(10, sample_count)])

        durations: List[float] = []
        y_pred = None
        for _ in range(max(1, n_repeats)):
            t_start_pred = time.perf_counter()
            y_pred = model_instance.predict(X_test)
            durations.append(time.perf_counter() - t_start_pred)

        pred_time_sec = float(np.mean(durations))
        latency_per_sample_ms = (pred_time_sec / sample_count) * 1000.0
    except Exception as exc:
        return SingleModelBenchmark(
            model_id=model_id,
            model_name=model_name,
            task_type=clean_task,
            status="FAILED",
            error_message=f"Inference error in {model_name}: {str(exc)}",
            training_time_sec=training_time_sec,
            ram_current_mb=ram_current_mb,
            ram_delta_mb=ram_delta_mb,
            energy_kwh=energy_kwh,
            carbon_emissions_g=carbon_emissions_g,
            energy_status=energy_status,
        )

    # Step 3: Compute Task-Specific Performance Metrics
    try:
        metrics = evaluate_predictions(y_test, y_pred, clean_task)
    except Exception as exc:
        return SingleModelBenchmark(
            model_id=model_id,
            model_name=model_name,
            task_type=clean_task,
            status="FAILED",
            error_message=f"Evaluation metric error: {str(exc)}",
            training_time_sec=training_time_sec,
            prediction_time_sec=pred_time_sec,
            latency_per_sample_ms=latency_per_sample_ms,
            ram_current_mb=ram_current_mb,
            ram_delta_mb=ram_delta_mb,
            energy_kwh=energy_kwh,
            carbon_emissions_g=carbon_emissions_g,
            energy_status=energy_status,
        )

    # Build successfully populated benchmark object
    benchmark = SingleModelBenchmark(
        model_id=model_id,
        model_name=model_name,
        task_type=clean_task,
        status="SUCCESS",
        training_time_sec=training_time_sec,
        prediction_time_sec=pred_time_sec,
        latency_per_sample_ms=latency_per_sample_ms,
        ram_current_mb=ram_current_mb,
        ram_delta_mb=ram_delta_mb,
        baseline_ram_mb=baseline_ram_mb,
        cpu_percent_avg=cpu_percent_avg,
        cpu_percent_peak=cpu_percent_peak,
        telemetry_overhead_ms=telemetry_overhead_ms,
        energy_kwh=energy_kwh,
        carbon_emissions_g=carbon_emissions_g,
        energy_status=energy_status,
        model_instance=model_instance,
    )

    if clean_task == "classification":
        benchmark.accuracy = metrics.get("accuracy")
        benchmark.precision = metrics.get("precision")
        benchmark.recall = metrics.get("recall")
        benchmark.f1 = metrics.get("f1")
        benchmark.confusion_matrix = metrics.get("confusion_matrix")
    else:
        benchmark.mae = metrics.get("mae")
        benchmark.rmse = metrics.get("rmse")
        benchmark.r2 = metrics.get("r2")

    return benchmark


def run_benchmark_suite(
    prep_result: PreprocessingResult,
    selected_model_ids: Optional[List[str]] = None,
    random_state: int = 42,
    track_emissions: bool = False,
    dataset_name: str = "Dataset",
) -> BenchmarkSuiteResult:
    """Execute benchmarking across all selected models on preprocessed partitions.

    Args:
        prep_result: Preprocessed dataset dataclass containing training/testing arrays.
        selected_model_ids: Optional list of model IDs to benchmark. If None, benchmarks all supported.
        random_state: Random state for deterministic model fitting.
        track_emissions: Whether to attempt hardware-level carbon emission tracking.
        dataset_name: Label describing the evaluated dataset.

    Returns:
        BenchmarkSuiteResult: Complete suite containing all individual model evaluations.

    Raises:
        ValueError: If preprocessed arrays are empty or incompatible.
    """
    if prep_result.X_train_processed.shape[0] == 0:
        raise ValueError("X_train_processed contains 0 training samples.")

    supported_models = get_supported_models(prep_result.task_type)

    if selected_model_ids is None or len(selected_model_ids) == 0:
        model_ids_to_run = list(supported_models.keys())
    else:
        model_ids_to_run = [m_id for m_id in selected_model_ids if m_id in supported_models]
        if not model_ids_to_run:
            raise ValueError(f"None of the requested model IDs {selected_model_ids} are supported.")

    results: List[SingleModelBenchmark] = []

    for m_id in model_ids_to_run:
        m_name = supported_models[m_id]
        try:
            model_obj = build_model_instance(m_id, prep_result.task_type, random_state=random_state)
            bench = benchmark_single_model(
                model_id=m_id,
                model_name=m_name,
                model_instance=model_obj,
                X_train=prep_result.X_train_processed.copy(),
                y_train=prep_result.y_train.copy(),
                X_test=prep_result.X_test_processed.copy(),
                y_test=prep_result.y_test.copy(),
                task_type=prep_result.task_type,
                track_emissions=track_emissions,
            )
        except Exception as exc:
            bench = SingleModelBenchmark(
                model_id=m_id,
                model_name=m_name,
                task_type=prep_result.task_type,
                status="FAILED",
                error_message=str(exc),
            )
        results.append(bench)

    return BenchmarkSuiteResult(
        task_type=prep_result.task_type,
        dataset_name=dataset_name,
        target_column=prep_result.target_column,
        sample_count_train=prep_result.train_shape[0],
        sample_count_test=prep_result.test_shape[0],
        feature_count=prep_result.train_shape[1],
        models_evaluated=results,
        random_state=random_state,
    )


def generate_benchmark_interpretation(suite_result: BenchmarkSuiteResult) -> Dict[str, Any]:
    """Generate an empirical, evidence-based comparative summary of benchmark findings.

    Args:
        suite_result: Completed benchmark suite.

    Returns:
        Dict[str, Any]: Empirical observations without fabricated assertions.
    """
    successful_runs = [m for m in suite_result.models_evaluated if m.status == "SUCCESS"]

    if not successful_runs:
        return {
            "summary_text": "No models completed successfully. Please check the error logs above.",
            "best_predictive_model": None,
            "fastest_training_model": None,
            "lowest_latency_model": None,
            "tradeoff_observation": "No empirical data available.",
        }

    # Best predictive model
    if suite_result.task_type == "classification":
        best_pred = max(successful_runs, key=lambda m: (m.f1 if m.f1 is not None else -1.0))
        pred_metric_name = "Weighted F1"
        pred_metric_val = f"{best_pred.f1:.4f}" if best_pred.f1 is not None else "N/A"
    else:
        best_pred = max(successful_runs, key=lambda m: (m.r2 if m.r2 is not None else -1e9))
        pred_metric_name = "R² Score"
        pred_metric_val = f"{best_pred.r2:.4f}" if best_pred.r2 is not None else "N/A"

    # Fastest training model
    fastest_train = min(
        successful_runs,
        key=lambda m: (m.training_time_sec if m.training_time_sec is not None else 1e9),
    )

    # Lowest inference latency
    lowest_latency = min(
        successful_runs,
        key=lambda m: (m.latency_per_sample_ms if m.latency_per_sample_ms is not None else 1e9),
    )

    # Formulate objective comparison with consistent ms and second designations
    tradeoffs = []
    if fastest_train.model_id != best_pred.model_id:
        if fastest_train.training_time_sec and best_pred.training_time_sec and fastest_train.training_time_sec > 0:
            speedup = best_pred.training_time_sec / fastest_train.training_time_sec
            best_ms = best_pred.training_time_sec * 1000.0
            fastest_ms = fastest_train.training_time_sec * 1000.0
            tradeoffs.append(
                f"**{best_pred.model_name}** achieved the top {pred_metric_name} ({pred_metric_val}), "
                f"but required **{speedup:.1f}× longer training time** than **{fastest_train.model_name}** "
                f"({best_ms:.1f} ms [{best_pred.training_time_sec:.4f}s] vs {fastest_ms:.1f} ms [{fastest_train.training_time_sec:.4f}s])."
            )
    else:
        best_ms = (best_pred.training_time_sec or 0.0) * 1000.0
        tradeoffs.append(
            f"**{best_pred.model_name}** simultaneously achieved the highest {pred_metric_name} ({pred_metric_val}) "
            f"and the fastest training time ({best_ms:.1f} ms [{best_pred.training_time_sec:.4f}s])."
        )

    if lowest_latency.model_id != best_pred.model_id and lowest_latency.latency_per_sample_ms is not None:
        tradeoffs.append(
            f"For inference-sensitive deployments, **{lowest_latency.model_name}** yielded the lowest latency "
            f"at **{lowest_latency.latency_per_sample_ms:.4f} ms/sample**."
        )

    fastest_train_ms = (fastest_train.training_time_sec or 0.0) * 1000.0
    return {
        "summary_text": (
            f"Evaluated {len(successful_runs)} models on {suite_result.dataset_name}. "
            f"Top predictive performer: {best_pred.model_name} ({pred_metric_name}: {pred_metric_val})."
        ),
        "best_predictive_model": best_pred.model_name,
        "best_predictive_metric": f"{pred_metric_name}: {pred_metric_val}",
        "fastest_training_model": f"{fastest_train.model_name} ({fastest_train_ms:.1f} ms / {fastest_train.training_time_sec:.4f}s)",
        "lowest_latency_model": f"{lowest_latency.model_name} ({lowest_latency.latency_per_sample_ms:.4f} ms/sample)",
        "tradeoff_observation": " ".join(tradeoffs),
    }

