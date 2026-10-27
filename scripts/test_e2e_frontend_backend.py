"""
End-to-end verification script for EcoML Phase 3.
Simulates the exact HTTP requests that the React frontend makes to the FastAPI backend.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app

def test_full_workflow():
    client = TestClient(app)

    print("1. Testing Health Endpoint (/api/health)...")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    health = res.json()
    assert health["status"] in ("healthy", "ok")
    assert "Operational" in health["operational_status"]
    print(f"   -> Health OK: {health['app']} v{health['version']} ({health['operational_status']})")

    print("\n2. Testing Meta Features (/api/meta/features)...")
    res = client.get("/api/meta/features")
    assert res.status_code == 200
    meta = res.json()
    assert len(meta["implemented_features"]) > 0
    print(f"   -> Implemented features: {len(meta['implemented_features'])}")
    print(f"   -> Roadmap steps: {len(meta['roadmap_steps'])}")

    print("\n3. Testing Datasets Listing (/api/datasets)...")
    res = client.get("/api/datasets")
    assert res.status_code == 200
    datasets = res.json()["datasets"]
    assert len(datasets) >= 3
    bc_dataset = next(d for d in datasets if d["dataset_id"] == "breast_cancer")
    print(f"   -> Found {len(datasets)} datasets. Selected: {bc_dataset['name']}")

    print("\n4. Testing Dataset Preview (/api/datasets/{id}/preview)...")
    res = client.get(f"/api/datasets/{bc_dataset['dataset_id']}/preview?n_preview=5")
    assert res.status_code == 200
    preview = res.json()
    assert preview["rows"] == 569
    assert preview["columns"] == 31
    assert len(preview["preview_rows"]) == 5
    print(f"   -> Dataset shape: {preview['rows']} rows x {preview['columns']} columns")

    print("\n5. Testing Target Validation (/api/datasets/{id}/validate-target)...")
    res = client.post(
        f"/api/datasets/{bc_dataset['dataset_id']}/validate-target",
        json={"target_column": "target", "task_type": "classification"},
    )
    assert res.status_code == 200
    val = res.json()
    assert val["is_valid"] is True
    assert val["unique_classes"] == 2
    print(f"   -> Target 'target' is valid: {val['unique_classes']} classes, {val['null_count']} nulls")

    print("\n6. Testing Preprocessing (/api/datasets/{id}/preprocess)...")
    res = client.post(
        f"/api/datasets/{bc_dataset['dataset_id']}/preprocess",
        json={
            "target_column": "target",
            "task_type": "classification",
            "test_size": 0.2,
            "random_state": 42,
            "scale_numerical": True,
        },
    )
    assert res.status_code == 200
    prep = res.json()
    assert prep["train_samples"] == 455
    assert prep["test_samples"] == 114
    assert prep["feature_count"] == 30
    print(f"   -> Partitioned: {prep['train_samples']} train, {prep['test_samples']} test. Hash: {prep['config_signature']}")

    print("\n7. Testing Supported Models (/api/models)...")
    res = client.get("/api/models?task_type=classification")
    assert res.status_code == 200
    models = res.json()["models"]
    model_ids = [m["model_id"] for m in models]
    print(f"   -> Supported models: {model_ids}")

    print("\n8. Submitting Benchmark Job (/api/benchmarks)...")
    res = client.post(
        "/api/benchmarks",
        json={
            "dataset_id": bc_dataset["dataset_id"],
            "selected_models": ["logistic_regression", "decision_tree", "random_forest", "knn"],
            "enable_carbon_tracking": False,
            "random_state": 42,
        },
    )
    assert res.status_code in (200, 202)
    bench_job = res.json()
    benchmark_id = bench_job["benchmark_id"]
    print(f"   -> Benchmark Job created: {benchmark_id} (Status: {bench_job['status']})")

    print("\n9. Polling Benchmark Progress until COMPLETED...")
    for _ in range(40):
        res = client.get(f"/api/benchmarks/{benchmark_id}")
        assert res.status_code == 200
        status = res.json()
        print(f"   -> Progress: {status['progress_percent']}% ({status['completed_models']}/{status['total_models']} models) Status: {status['status']}")
        if status["status"] == "COMPLETED":
            break
        elif status["status"] == "FAILED":
            raise RuntimeError(f"Benchmark failed: {status['error_message']}")
        time.sleep(0.5)

    assert status["status"] == "COMPLETED"

    print("\n10. Fetching Benchmark Results (/api/benchmarks/{id}/results)...")
    res = client.get(f"/api/benchmarks/{benchmark_id}/results")
    assert res.status_code == 200
    results = res.json()
    assert len(results["models_evaluated"]) == 4
    for m in results["models_evaluated"]:
        print(f"   * {m['model_name']}: Acc={m['accuracy']:.4f}, FitTime={(m['training_time_sec']*1000):.1f}ms, PeakRAM={m['telemetry']['peak_ram_mb']:.1f}MB")

    print("\n11. Testing Recommendation Profiles (/api/recommendations/profiles)...")
    res = client.get("/api/recommendations/profiles")
    assert res.status_code == 200
    presets = res.json()["presets"]
    print(f"   -> Available presets: {[p['name'] for p in presets]}")

    print("\n12. Calculating Green Intelligence Recommendations (/api/recommendations/{id})...")
    res = client.post(f"/api/recommendations/{benchmark_id}", json={})
    assert res.status_code == 200
    rec = res.json()
    top = rec["top_recommended_model"]
    print(f"   -> Profile: {rec['profile_name']}")
    print(f"   -> Rank #1 Model: {top['model_name']} with EcoML Score: {top['composite_score']:.1f}/100")
    print(f"   -> Pareto Optimal Models: {rec['pareto_frontier_models']}")
    print(f"   -> Effective Weights: {rec['effective_weights']}")
    print(f"   -> Tradeoff Synthesis: {rec['tradeoff_synthesis']}")

    # 13. Regression Workflow Verification
    print("\n13. Testing Regression Workflow (Diabetes)...")
    res = client.post(
        "/api/datasets/diabetes/preprocess",
        json={
            "target_column": "target",
            "task_type": "regression",
            "test_size": 0.2,
            "random_state": 42,
            "scale_numerical": True,
        },
    )
    assert res.status_code == 200
    prep_reg = res.json()
    print(f"   -> Partitioned Diabetes: {prep_reg['train_samples']} train, {prep_reg['test_samples']} test")

    res = client.post(
        "/api/benchmarks",
        json={
            "dataset_id": "diabetes",
            "selected_models": ["linear_regression", "random_forest", "knn"],
            "enable_carbon_tracking": False,
            "random_state": 42,
        },
    )
    assert res.status_code in (200, 202)
    reg_bid = res.json()["benchmark_id"]

    for _ in range(40):
        res = client.get(f"/api/benchmarks/{reg_bid}")
        status = res.json()
        if status["status"] == "COMPLETED":
            break
        elif status["status"] == "FAILED":
            raise RuntimeError(f"Regression benchmark failed: {status['error_message']}")
        time.sleep(0.5)

    assert status["status"] == "COMPLETED"

    res = client.get(f"/api/benchmarks/{reg_bid}/results")
    assert res.status_code == 200
    reg_results = res.json()
    assert len(reg_results["models_evaluated"]) == 3
    for m in reg_results["models_evaluated"]:
        print(f"   * {m['model_name']}: R²={m['r2']:.4f}, RMSE={m['rmse']:.4f}, FitTime={(m['training_time_sec']*1000):.1f}ms")

    res = client.post(f"/api/recommendations/{reg_bid}", json={})
    assert res.status_code == 200
    reg_rec = res.json()
    reg_top = reg_rec["top_recommended_model"]
    r2_val = reg_top["raw_metrics"].get("r2") or reg_top["raw_metrics"].get("performance")
    print(f"   -> Regression Rank #1: {reg_top['model_name']} (EcoML Score: {reg_top['composite_score']:.1f}/100, R² Utility: {r2_val:.4f})")

    print("\n=======================================================")
    print("SUCCESS: BOTH CLASSIFICATION & REGRESSION WORKFLOWS PASS 100%!")
    print("=======================================================")

if __name__ == "__main__":
    test_full_workflow()
