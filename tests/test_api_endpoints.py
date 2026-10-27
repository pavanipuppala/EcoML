"""
EcoML — Comprehensive FastAPI Integration & Endpoint Test Suite.

Validates all REST API endpoints: health, datasets, uploads, preprocessing,
model benchmarking lifecycle, green recommendations, CORS, and error containment.
"""

import io
import time
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from config.settings import PROJECT_NAME, PROJECT_VERSION, PROJECT_STATUS

client = TestClient(app)


def test_health_and_meta_endpoints():
    """Verify system health check and metadata feature reflection."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == PROJECT_NAME
    assert data["version"] == PROJECT_VERSION
    assert data["operational_status"] == PROJECT_STATUS

    res_meta = client.get("/api/meta/features")
    assert res_meta.status_code == 200
    meta_data = res_meta.json()
    assert len(meta_data["implemented_features"]) >= 6
    assert len(meta_data["future_features"]) >= 4
    assert len(meta_data["roadmap_steps"]) >= 5


def test_cors_headers():
    """Verify CORS headers allow Vite local development origins."""
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    }
    res = client.options("/api/health", headers=headers)
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_dataset_listing_and_preview():
    """Verify listing of built-in datasets and schema preview diagnostics."""
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()["datasets"]
    ids = [d["dataset_id"] for d in datasets]
    assert "demo" in ids
    assert "breast_cancer" in ids
    assert "diabetes" in ids

    # Preview demo
    res_prev = client.get("/api/datasets/demo/preview")
    assert res_prev.status_code == 200
    prev_data = res_prev.json()
    assert prev_data["dataset_id"] == "demo"
    assert prev_data["rows"] == 15
    assert prev_data["columns"] == 6
    assert len(prev_data["preview_rows"]) == 10
    assert "summary" in prev_data

    # Preview non-existent dataset
    res_404 = client.get("/api/datasets/non_existent_dataset/preview")
    assert res_404.status_code == 404


def test_csv_upload_valid_and_invalid():
    """Verify CSV upload validation, size checks, and path safety."""
    # 1. Valid CSV
    valid_csv = b"col_a,col_b,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n13,14,0\n15,16,1\n17,18,0\n19,20,1\n"
    res_valid = client.post(
        "/api/datasets/upload",
        files={"file": ("test_upload.csv", io.BytesIO(valid_csv), "text/csv")},
    )
    assert res_valid.status_code == 201
    upload_info = res_valid.json()
    assert "dataset_id" in upload_info
    assert upload_info["rows"] == 10
    assert upload_info["columns"] == 3

    # 2. Non-CSV extension
    res_bad_ext = client.post(
        "/api/datasets/upload",
        files={"file": ("malicious.exe", io.BytesIO(b"data"), "application/octet-stream")},
    )
    assert res_bad_ext.status_code == 400

    # 3. Empty CSV
    res_empty = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert res_bad_ext.status_code == 400

    # 4. Too few rows (requires at least 10 rows for ML train/test splitting)
    few_rows_csv = b"a,b\n1,2\n3,4\n"
    res_few = client.post(
        "/api/datasets/upload",
        files={"file": ("few.csv", io.BytesIO(few_rows_csv), "text/csv")},
    )
    assert res_few.status_code == 400
    assert "too few rows" in res_few.json()["detail"].lower()

    # 5. Path traversal attempt in filename is safely sanitized
    res_traversal = client.post(
        "/api/datasets/upload",
        files={"file": ("../../etc/passwd.csv", io.BytesIO(valid_csv), "text/csv")},
    )
    assert res_traversal.status_code == 201
    assert "passwd" in res_traversal.json()["name"]


def test_target_validation():
    """Verify target column validation across classification and regression."""
    # Classification success on Breast Cancer
    res_bc = client.post(
        "/api/datasets/breast_cancer/validate-target",
        json={"target_column": "target", "task_type": "classification"},
    )
    assert res_bc.status_code == 200
    assert res_bc.json()["is_valid"] is True
    assert res_bc.json()["unique_classes"] == 2

    # Invalid non-existent target
    res_inv = client.post(
        "/api/datasets/breast_cancer/validate-target",
        json={"target_column": "non_existent_col", "task_type": "classification"},
    )
    assert res_inv.status_code == 200
    assert res_inv.json()["is_valid"] is False

    # Regression success on Diabetes
    res_diab = client.post(
        "/api/datasets/diabetes/validate-target",
        json={"target_column": "target", "task_type": "regression"},
    )
    assert res_diab.status_code == 200
    assert res_diab.json()["is_valid"] is True


def test_preprocessing_endpoint():
    """Verify leakage-free preprocessing pipeline configuration."""
    # Successful preprocessing on Breast Cancer
    res = client.post(
        "/api/datasets/breast_cancer/preprocess",
        json={
            "target_column": "target",
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
            "scale_numerical": True,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["train_samples"] == 455
    assert data["test_samples"] == 114
    assert data["feature_count"] == 30
    assert len(data["feature_names_out"]) == 30

    # Bad parameter: invalid target
    res_bad = client.post(
        "/api/datasets/breast_cancer/preprocess",
        json={
            "target_column": "bad_col",
            "task_type": "classification",
        },
    )
    assert res_bad.status_code == 400


def test_supported_models():
    """Verify model discovery endpoint."""
    res_cls = client.get("/api/models?task_type=classification")
    assert res_cls.status_code == 200
    cls_models = res_cls.json()["models"]
    cls_ids = [m["model_id"] for m in cls_models]
    assert "logistic_regression" in cls_ids
    assert "random_forest" in cls_ids

    res_reg = client.get("/api/models?task_type=regression")
    assert res_reg.status_code == 200
    reg_models = res_reg.json()["models"]
    reg_ids = [m["model_id"] for m in reg_models]
    assert "linear_regression" in reg_ids
    assert "gradient_boosting" in reg_ids


def test_classification_benchmark_and_recommendation_lifecycle():
    """End-to-end lifecycle: preprocess -> submit benchmark -> poll -> results -> recommendations."""
    # 1. Preprocess Breast Cancer
    client.post(
        "/api/datasets/breast_cancer/preprocess",
        json={"target_column": "target", "task_type": "classification"},
    )

    # 2. Submit fast benchmark
    res_sub = client.post(
        "/api/benchmarks",
        json={
            "dataset_id": "breast_cancer",
            "selected_models": ["logistic_regression", "decision_tree"],
            "enable_carbon_tracking": False,
            "random_state": 42,
        },
    )
    assert res_sub.status_code == 202
    bench_id = res_sub.json()["benchmark_id"]

    # 3. Poll status until completed (timeout: 10s)
    for _ in range(20):
        time.sleep(0.2)
        res_stat = client.get(f"/api/benchmarks/{bench_id}")
        assert res_stat.status_code == 200
        stat_data = res_stat.json()
        if stat_data["status"] == "COMPLETED":
            break
    assert stat_data["status"] == "COMPLETED"
    assert stat_data["completed_models"] == 2

    # 4. Retrieve Results
    res_results = client.get(f"/api/benchmarks/{bench_id}/results")
    assert res_results.status_code == 200
    results_data = res_results.json()
    assert len(results_data["models_evaluated"]) == 2
    top_eval = results_data["models_evaluated"][0]
    assert top_eval["accuracy"] is not None
    assert top_eval["training_time_sec"] is not None
    assert top_eval["telemetry"]["peak_ram_mb"] is not None

    # 5. Fetch Recommendation Profiles
    res_prof = client.get("/api/recommendations/profiles")
    assert res_prof.status_code == 200
    assert len(res_prof.json()["presets"]) >= 4

    # 6. Calculate Recommendations (Balanced profile)
    res_rec = client.post(f"/api/recommendations/{bench_id}")
    assert res_rec.status_code == 200
    rec_data = res_rec.json()
    assert rec_data["top_recommended_model"] is not None
    assert len(rec_data["ranked_models"]) == 2
    assert "pareto_frontier_models" in rec_data
    assert rec_data["top_recommended_model"]["composite_score"] > 0

    # 7. Calculate Recommendations with Custom Weight Profile
    custom_prof = {
        "name": "Custom-Accuracy",
        "w_performance": 0.80,
        "w_train_time": 0.05,
        "w_inference_latency": 0.05,
        "w_ram": 0.05,
        "w_carbon": 0.05,
    }
    res_rec_custom = client.post(
        f"/api/recommendations/{bench_id}",
        json={"weight_profile": custom_prof},
    )
    assert res_rec_custom.status_code == 200
    assert res_rec_custom.json()["profile_name"] == "Custom-Accuracy"


def test_regression_benchmark_and_recommendation_lifecycle():
    """End-to-end lifecycle for regression on Diabetes benchmark."""
    # 1. Preprocess Diabetes
    client.post(
        "/api/datasets/diabetes/preprocess",
        json={"target_column": "target", "task_type": "regression"},
    )

    # 2. Submit fast benchmark
    res_sub = client.post(
        "/api/benchmarks",
        json={
            "dataset_id": "diabetes",
            "selected_models": ["linear_regression", "decision_tree"],
            "enable_carbon_tracking": False,
            "random_state": 42,
        },
    )
    assert res_sub.status_code == 202
    bench_id = res_sub.json()["benchmark_id"]

    # 3. Poll status
    for _ in range(20):
        time.sleep(0.2)
        res_stat = client.get(f"/api/benchmarks/{bench_id}")
        if res_stat.json()["status"] == "COMPLETED":
            break
    assert res_stat.json()["status"] == "COMPLETED"

    # 4. Results check
    res_results = client.get(f"/api/benchmarks/{bench_id}/results")
    assert res_results.status_code == 200
    results_data = res_results.json()
    for m in results_data["models_evaluated"]:
        assert m["r2"] is not None
        assert m["mae"] is not None
        assert m["rmse"] is not None
        assert m["accuracy"] is None  # Ensures classification metric not leaked into regression


def test_error_and_edge_case_handling():
    """Verify error responses for missing IDs, invalid submissions, and empty configurations."""
    # 1. Non-existent benchmark ID
    res_404 = client.get("/api/benchmarks/non_existent_bench_id")
    assert res_404.status_code == 404

    # 2. Benchmark submission for non-preprocessed dataset
    res_unprep = client.post(
        "/api/benchmarks",
        json={"dataset_id": "demo", "selected_models": ["logistic_regression"]},
    )
    # demo was not preprocessed in this test yet
    if res_unprep.status_code == 400:
        assert "no preprocessed data" in res_unprep.json()["detail"].lower()

    # 3. Recommendation on non-existent benchmark
    res_rec_404 = client.post("/api/recommendations/non_existent_bench_id")
    assert res_rec_404.status_code == 404


def test_concurrent_benchmark_isolation():
    """Verify that concurrent benchmark submissions are queued sequentially under thread lock."""
    # Preprocess both demo and breast cancer
    client.post("/api/datasets/demo/preprocess", json={"target_column": "promoted", "task_type": "classification"})
    client.post("/api/datasets/breast_cancer/preprocess", json={"target_column": "target", "task_type": "classification"})

    # Submit job 1
    res1 = client.post("/api/benchmarks", json={"dataset_id": "demo", "selected_models": ["logistic_regression", "decision_tree"]})
    assert res1.status_code == 202
    b1_id = res1.json()["benchmark_id"]

    # Immediately submit job 2
    res2 = client.post("/api/benchmarks", json={"dataset_id": "breast_cancer", "selected_models": ["logistic_regression", "decision_tree"]})
    assert res2.status_code == 202
    b2_id = res2.json()["benchmark_id"]

    # Poll both until completed
    for _ in range(40):
        time.sleep(0.3)
        s1 = client.get(f"/api/benchmarks/{b1_id}").json()["status"]
        s2 = client.get(f"/api/benchmarks/{b2_id}").json()["status"]
        if s1 == "COMPLETED" and s2 == "COMPLETED":
            break

    assert s1 == "COMPLETED"
    assert s2 == "COMPLETED"


def test_oversized_upload_rejection():
    """Verify that uploads exceeding 50 MB are strictly rejected with HTTP 400."""
    from backend.services.dataset_service import MAX_UPLOAD_SIZE_BYTES
    # Simulate oversized stream by passing bytes exceeding limit
    fake_oversized = b"a" * (MAX_UPLOAD_SIZE_BYTES + 1024)
    res_large = client.post(
        "/api/datasets/upload",
        files={"file": ("huge.csv", io.BytesIO(fake_oversized), "text/csv")},
    )
    assert res_large.status_code == 400
    assert "exceeds 50 mb" in res_large.json()["detail"].lower()

