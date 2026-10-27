"""
EcoML — Benchmark Execution and Job State Service.

Manages asynchronous model benchmarking jobs, sequential telemetry locks,
progress tracking, and empirical resource measurements without event loop blocking.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from src.benchmark_engine import (
    BenchmarkSuiteResult,
    SingleModelBenchmark,
    get_supported_models,
    build_model_instance,
    benchmark_single_model,
    generate_benchmark_interpretation,
)
from src.data_preprocessing import PreprocessingResult
from backend.services.dataset_service import dataset_service


class BenchmarkService:
    """Thread-safe benchmark manager with sequential execution locking for telemetry isolation."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._thread_lock = threading.Lock()

    def get_supported_model_list(self, task_type: str) -> List[Dict[str, str]]:
        """Return list of supported models for task type."""
        model_dict = get_supported_models(task_type)
        return [
            {"model_id": m_id, "model_name": m_name, "task_type": task_type.lower()}
            for m_id, m_name in model_dict.items()
        ]

    def submit_benchmark_job(
        self,
        dataset_id: str,
        selected_models: Optional[List[str]] = None,
        enable_carbon_tracking: bool = False,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Validate preprocessed dataset, register job, and dispatch asynchronous worker thread."""
        # Ensure dataset has been preprocessed
        prep_data: PreprocessingResult = dataset_service.get_preprocessed(dataset_id)
        dataset_name, _ = dataset_service.get_dataset(dataset_id)

        model_dict = get_supported_models(prep_data.task_type)
        all_models = list(model_dict.keys())
        if selected_models:
            invalid = [m for m in selected_models if m not in all_models]
            if invalid:
                raise ValueError(f"Invalid model IDs for {prep_data.task_type}: {invalid}. Supported: {all_models}")
            models_to_run = [m for m in selected_models if m in all_models]
        else:
            models_to_run = all_models

        if not models_to_run:
            raise ValueError("No models selected for evaluation.")

        benchmark_id = f"bench_{uuid.uuid4().hex[:10]}"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        job_record = {
            "benchmark_id": benchmark_id,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "task_type": prep_data.task_type,
            "target_column": prep_data.target_column,
            "sample_count_train": int(prep_data.train_shape[0]),
            "sample_count_test": int(prep_data.test_shape[0]),
            "feature_count": int(prep_data.train_shape[1]),
            "status": "QUEUED",
            "progress_percent": 0.0,
            "completed_models": 0,
            "total_models": len(models_to_run),
            "models_to_run": models_to_run,
            "enable_carbon_tracking": enable_carbon_tracking,
            "random_state": random_state,
            "created_at": now_str,
            "completed_at": None,
            "error_message": None,
            "suite_result": None,
            "interpretation": None,
        }
        self._jobs[benchmark_id] = job_record

        # Launch background worker in dedicated OS thread
        worker_thread = threading.Thread(
            target=self._run_benchmark_worker_sync,
            args=(benchmark_id, prep_data),
            daemon=True,
            name=f"EcoML-Bench-{benchmark_id}",
        )
        worker_thread.start()

        return {
            "benchmark_id": benchmark_id,
            "dataset_id": dataset_id,
            "status": "QUEUED",
            "total_models": len(models_to_run),
        }

    def _run_benchmark_worker_sync(self, benchmark_id: str, prep_data: PreprocessingResult) -> None:
        """Sequential background worker with concurrency locking for telemetry isolation."""
        job = self._jobs[benchmark_id]

        with self._thread_lock:
            job["status"] = "RUNNING"
            models_to_run = job["models_to_run"]
            completed_runs: List[SingleModelBenchmark] = []

            try:
                model_dict = get_supported_models(prep_data.task_type)
                for idx, model_id in enumerate(models_to_run):
                    model_name = model_dict.get(model_id, model_id)
                    model_instance = build_model_instance(
                        model_id=model_id,
                        task_type=prep_data.task_type,
                        random_state=job["random_state"],
                    )
                    run_result = benchmark_single_model(
                        model_id=model_id,
                        model_name=model_name,
                        model_instance=model_instance,
                        X_train=prep_data.X_train_processed,
                        y_train=prep_data.y_train,
                        X_test=prep_data.X_test_processed,
                        y_test=prep_data.y_test,
                        task_type=prep_data.task_type,
                        track_emissions=job["enable_carbon_tracking"],
                    )
                    completed_runs.append(run_result)
                    job["completed_models"] = idx + 1
                    job["progress_percent"] = round(((idx + 1) / len(models_to_run)) * 100.0, 1)

                # Assemble final BenchmarkSuiteResult
                suite = BenchmarkSuiteResult(
                    task_type=prep_data.task_type,
                    dataset_name=job["dataset_name"],
                    target_column=prep_data.target_column,
                    sample_count_train=int(prep_data.train_shape[0]),
                    sample_count_test=int(prep_data.test_shape[0]),
                    feature_count=int(prep_data.train_shape[1]),
                    models_evaluated=completed_runs,
                    benchmark_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    random_state=job["random_state"],
                )
                interp = generate_benchmark_interpretation(suite)

                job["suite_result"] = suite
                job["interpretation"] = interp
                job["status"] = "COMPLETED"
                job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            except Exception as exc:
                job["status"] = "FAILED"
                job["error_message"] = str(exc)
                job["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_job_status(self, benchmark_id: str) -> Dict[str, Any]:
        """Fetch current benchmark execution status."""
        if benchmark_id not in self._jobs:
            raise ValueError(f"Benchmark job '{benchmark_id}' not found.")
        job = self._jobs[benchmark_id]
        return {
            "benchmark_id": job["benchmark_id"],
            "dataset_id": job["dataset_id"],
            "status": job["status"],
            "progress_percent": job["progress_percent"],
            "completed_models": job["completed_models"],
            "total_models": job["total_models"],
            "error_message": job["error_message"],
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
        }

    def get_job_results(self, benchmark_id: str) -> Dict[str, Any]:
        """Fetch completed benchmark results, or raise ValueError if pending/failed."""
        if benchmark_id not in self._jobs:
            raise ValueError(f"Benchmark job '{benchmark_id}' not found.")
        job = self._jobs[benchmark_id]

        if job["status"] == "FAILED":
            raise ValueError(f"Benchmark job '{benchmark_id}' failed: {job['error_message']}")
        if job["status"] in ("QUEUED", "RUNNING"):
            raise ValueError(f"Benchmark job '{benchmark_id}' is still in progress (status: {job['status']}).")

        suite: BenchmarkSuiteResult = job["suite_result"]

        # Format evaluated models for JSON API
        formatted_models = []
        for m in suite.models_evaluated:
            formatted_models.append({
                "model_id": m.model_id,
                "model_name": m.model_name,
                "task_type": m.task_type,
                "status": m.status,
                "error_message": m.error_message,
                "accuracy": round(m.accuracy, 4) if m.accuracy is not None else None,
                "precision": round(m.precision, 4) if m.precision is not None else None,
                "recall": round(m.recall, 4) if m.recall is not None else None,
                "f1": round(m.f1, 4) if m.f1 is not None else None,
                "confusion_matrix": m.confusion_matrix,
                "mae": round(m.mae, 4) if m.mae is not None else None,
                "rmse": round(m.rmse, 4) if m.rmse is not None else None,
                "r2": round(m.r2, 4) if m.r2 is not None else None,
                "training_time_sec": round(m.training_time_sec, 4) if m.training_time_sec is not None else None,
                "prediction_time_sec": round(m.prediction_time_sec, 4) if m.prediction_time_sec is not None else None,
                "latency_per_sample_ms": round(m.latency_per_sample_ms, 4) if m.latency_per_sample_ms is not None else None,
                "telemetry": {
                    "baseline_ram_mb": round(m.baseline_ram_mb, 2) if m.baseline_ram_mb is not None else None,
                    "peak_ram_mb": round(m.ram_current_mb, 2) if m.ram_current_mb is not None else None,
                    "ram_delta_mb": round(m.ram_delta_mb, 2) if m.ram_delta_mb is not None else None,
                    "avg_cpu_percent": round(m.cpu_percent_avg, 2) if m.cpu_percent_avg is not None else None,
                    "peak_cpu_percent": round(m.cpu_percent_peak, 2) if m.cpu_percent_peak is not None else None,
                    "energy_kwh": m.energy_kwh,
                    "carbon_emissions_g": m.carbon_emissions_g,
                    "energy_status": m.energy_status,
                    "telemetry_overhead_ms": round(m.telemetry_overhead_ms, 3) if m.telemetry_overhead_ms is not None else None,
                },
            })

        return {
            "benchmark_id": benchmark_id,
            "dataset_id": job["dataset_id"],
            "dataset_name": suite.dataset_name,
            "task_type": suite.task_type,
            "target_column": suite.target_column,
            "sample_count_train": suite.sample_count_train,
            "sample_count_test": suite.sample_count_test,
            "feature_count": suite.feature_count,
            "benchmark_timestamp": suite.benchmark_timestamp,
            "models_evaluated": formatted_models,
            "tradeoff_interpretation": job["interpretation"],
        }

    def get_suite_result_object(self, benchmark_id: str) -> BenchmarkSuiteResult:
        """Return the internal BenchmarkSuiteResult instance for recommender service."""
        if benchmark_id not in self._jobs:
            raise ValueError(f"Benchmark job '{benchmark_id}' not found.")
        job = self._jobs[benchmark_id]
        if job["status"] != "COMPLETED" or job["suite_result"] is None:
            raise ValueError(f"Benchmark job '{benchmark_id}' is not in COMPLETED state (status: {job['status']}).")
        return job["suite_result"]


# Global singleton instance for benchmark service
benchmark_service = BenchmarkService()
