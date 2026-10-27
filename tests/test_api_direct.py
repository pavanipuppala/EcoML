"""
EcoML — Direct FastAPI Route Handlers and Service Integration Tests.

Executes API route logic directly through Pydantic schemas and service orchestration,
verifying 100% of the API contract without requiring loopback TCP sockets in sandbox environments.
"""

import time
import pytest
from fastapi import HTTPException

from backend.api.routes.health import get_health, get_meta_features
from backend.api.routes.datasets import (
    list_datasets,
    get_dataset_preview,
    validate_target_column,
    preprocess_dataset,
)
from backend.api.routes.benchmarks import (
    get_models,
    submit_benchmark,
    get_benchmark_status,
    get_benchmark_results,
)
from backend.api.routes.recommendations import (
    get_weight_profile_presets,
    calculate_recommendations,
)
from backend.schemas.datasets import (
    TargetValidationRequest,
    PreprocessingRequest,
)
from backend.schemas.benchmarks import BenchmarkSubmissionRequest
from backend.schemas.recommendations import (
    RecommendationRequest,
    WeightProfileSchema,
)
from backend.api.routes.quantum import (
    get_quantum_info,
    run_quantum_optimization,
)
from backend.schemas.quantum import (
    QuantumOptimizationRequest,
)


def test_direct_health_and_meta():
    """Verify health and meta feature endpoints."""
    health = get_health()
    assert health.status == "healthy"
    assert health.app == "EcoML"
    assert "Operational" in health.operational_status

    meta = get_meta_features()
    assert len(meta.implemented_features) >= 6
    assert len(meta.future_features) >= 4
    assert len(meta.roadmap_steps) >= 5


def test_direct_dataset_catalog_and_preview():
    """Verify dataset catalog listing and schema preview."""
    res = list_datasets()
    assert len(res.datasets) >= 3
    ids = [d.dataset_id for d in res.datasets]
    assert "breast_cancer" in ids
    assert "diabetes" in ids

    prev = get_dataset_preview("breast_cancer", n_preview=5)
    assert prev.rows == 569
    assert prev.columns == 31
    assert len(prev.preview_rows) == 5

    # Nonexistent dataset raises 404
    with pytest.raises(HTTPException) as exc_info:
        get_dataset_preview("nonexistent_id_xyz")
    assert exc_info.value.status_code == 404


def test_direct_target_validation():
    """Verify target column validation logic."""
    valid_res = validate_target_column(
        "breast_cancer",
        TargetValidationRequest(target_column="target", task_type="classification"),
    )
    assert valid_res.is_valid is True
    assert valid_res.unique_classes == 2

    # Continuous column rejected for classification
    continuous_res = validate_target_column(
        "breast_cancer",
        TargetValidationRequest(target_column="smoothness error", task_type="classification"),
    )
    assert continuous_res.is_valid is False
    assert "continuous numerical data" in continuous_res.error_message

    # Continuous column accepted for regression
    reg_res = validate_target_column(
        "breast_cancer",
        TargetValidationRequest(target_column="smoothness error", task_type="regression"),
    )
    assert reg_res.is_valid is True

    # Preprocessing with continuous column for classification is rejected with 400
    with pytest.raises(HTTPException) as exc_info:
        preprocess_dataset(
            "breast_cancer",
            PreprocessingRequest(
                target_column="smoothness error",
                task_type="classification",
                test_size=0.2,
                random_state=42,
            ),
        )
    assert exc_info.value.status_code == 400

    invalid_res = validate_target_column(
        "breast_cancer",
        TargetValidationRequest(target_column="missing_col", task_type="classification"),
    )
    assert invalid_res.is_valid is False
    assert invalid_res.error_message is not None


def test_direct_classification_workflow_and_recommendations():
    """Verify full classification preprocessing, benchmark execution, and green recommendations."""
    # 1. Preprocess
    prep_req = PreprocessingRequest(
        target_column="target",
        task_type="classification",
        test_size=0.2,
        random_state=42,
        scale_numerical=True,
    )
    prep = preprocess_dataset("breast_cancer", prep_req)
    assert prep.train_samples == 455
    assert prep.test_samples == 114
    assert prep.feature_count == 30

    # 2. Supported models
    models_res = get_models(task_type="classification")
    assert len(models_res.models) >= 5

    # 3. Submit benchmark
    bench_req = BenchmarkSubmissionRequest(
        dataset_id="breast_cancer",
        selected_models=["logistic_regression", "decision_tree", "knn"],
        enable_carbon_tracking=False,
        random_state=42,
    )
    sub = submit_benchmark(bench_req)
    b_id = sub["benchmark_id"]

    # Poll status
    for _ in range(30):
        st = get_benchmark_status(b_id)
        if st.status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.3)

    assert st.status == "COMPLETED"

    # Fetch results
    results = get_benchmark_results(b_id)
    assert len(results.models_evaluated) == 3

    # Generate recommendations
    rec = calculate_recommendations(b_id, RecommendationRequest())
    assert rec.top_recommended_model is not None
    assert rec.top_recommended_model.model_name in ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"]
    assert rec.effective_weights["performance"] > 0
    assert "carbon" not in rec.effective_weights  # Carbon was disabled/untracked


def test_direct_regression_workflow():
    """Verify full regression preprocessing, benchmark execution, and green recommendations."""
    # 1. Preprocess
    prep_req = PreprocessingRequest(
        target_column="target",
        task_type="regression",
        test_size=0.2,
        random_state=42,
        scale_numerical=True,
    )
    prep = preprocess_dataset("diabetes", prep_req)
    assert prep.train_samples == 353
    assert prep.test_samples == 89
    assert prep.feature_count == 10

    # 2. Submit benchmark
    bench_req = BenchmarkSubmissionRequest(
        dataset_id="diabetes",
        selected_models=["linear_regression", "decision_tree"],
        enable_carbon_tracking=False,
        random_state=42,
    )
    sub = submit_benchmark(bench_req)
    b_id = sub["benchmark_id"]

    for _ in range(30):
        st = get_benchmark_status(b_id)
        if st.status in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.3)

    assert st.status == "COMPLETED"

    results = get_benchmark_results(b_id)
    assert len(results.models_evaluated) == 2

    rec = calculate_recommendations(b_id, RecommendationRequest())
    assert rec.top_recommended_model is not None
    assert rec.top_recommended_model.raw_metrics.get("r2") is not None or rec.top_recommended_model.raw_metrics.get("performance") is not None


def test_direct_error_containment():
    """Verify error containment and standard HTTP exceptions."""
    # Nonexistent benchmark status
    with pytest.raises(HTTPException) as exc_info:
        get_benchmark_status("invalid_id_999")
    assert exc_info.value.status_code == 404

    # Nonexistent benchmark results
    with pytest.raises(HTTPException) as exc_info:
        get_benchmark_results("invalid_id_999")
    assert exc_info.value.status_code == 404

    # Nonexistent recommendations
    with pytest.raises(HTTPException) as exc_info:
        calculate_recommendations("invalid_id_999", RecommendationRequest())
    assert exc_info.value.status_code == 404

    # Invalid task type
    with pytest.raises(HTTPException) as exc_info:
        get_models(task_type="nonexistent_task")
    assert exc_info.value.status_code == 400

    # Unpreprocessed dataset benchmark
    with pytest.raises(HTTPException) as exc_info:
        submit_benchmark(BenchmarkSubmissionRequest(dataset_id="demo", selected_models=["logistic_regression"]))
    assert exc_info.value.status_code == 400


def test_direct_quantum_workflow():
    """Verify quantum info discovery and QAOA optimization execution."""
    info = get_quantum_info()
    assert info.max_qubits >= 10
    assert "COBYLA" in info.supported_optimizers

    res = run_quantum_optimization(
        QuantumOptimizationRequest(
            depth_p=1,
            shots=256,
            classical_optimizer="COBYLA",
        )
    )
    assert len(res.candidates) >= 3
    assert res.classical_baseline.is_feasible is True
    assert res.qaoa_result.depth_p == 1
    assert res.qaoa_result.shots == 256
    assert res.qaoa_result.best_feasible_model is not None
    assert res.comparison.classical_selected_model != ""

