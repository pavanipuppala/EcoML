"""
EcoML — Machine Learning Benchmark Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ModelInfo(BaseModel):
    """Supported model metadata."""
    model_id: str
    model_name: str
    task_type: str


class SupportedModelsResponse(BaseModel):
    """List of available models for the given task."""
    task_type: str
    models: List[ModelInfo]


class BenchmarkSubmissionRequest(BaseModel):
    """Model benchmark execution request."""
    dataset_id: str = Field(..., description="ID of preprocessed dataset.")
    selected_models: Optional[List[str]] = Field(None, description="Subset of model IDs to benchmark.")
    enable_carbon_tracking: bool = Field(False, description="Enable CodeCarbon hardware emission tracking.")
    random_state: int = Field(42, ge=0, le=2147483647, description="Reproducibility random seed.")


class TelemetryMetrics(BaseModel):
    """Process resource and environmental telemetry observations."""
    baseline_ram_mb: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    ram_delta_mb: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    peak_cpu_percent: Optional[float] = None
    energy_kwh: Optional[float] = None
    carbon_emissions_g: Optional[float] = None
    energy_status: str
    telemetry_overhead_ms: Optional[float] = None


class SingleModelResult(BaseModel):
    """Empirical observations for a single evaluated model."""
    model_id: str
    model_name: str
    task_type: str
    status: str
    error_message: Optional[str] = None

    # Classification
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None

    # Regression
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2: Optional[float] = None

    # Performance & Timing
    training_time_sec: Optional[float] = None
    prediction_time_sec: Optional[float] = None
    latency_per_sample_ms: Optional[float] = None

    # Telemetry
    telemetry: TelemetryMetrics


class BenchmarkJobStatusResponse(BaseModel):
    """Asynchronous benchmark execution job status."""
    benchmark_id: str
    dataset_id: str
    status: str = Field(..., description="'QUEUED', 'RUNNING', 'COMPLETED', or 'FAILED'.")
    progress_percent: float = Field(0.0, ge=0.0, le=100.0)
    completed_models: int = 0
    total_models: int = 0
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class BenchmarkResultsResponse(BaseModel):
    """Complete benchmark suite results."""
    benchmark_id: str
    dataset_id: str
    dataset_name: str
    task_type: str
    target_column: str
    sample_count_train: int
    sample_count_test: int
    feature_count: int
    benchmark_timestamp: str
    models_evaluated: List[SingleModelResult]
    tradeoff_interpretation: Optional[Dict[str, Any]] = None

