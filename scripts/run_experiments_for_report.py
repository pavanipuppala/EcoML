import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from sklearn.datasets import load_breast_cancer, load_diabetes
from src.data_preprocessing import separate_features_and_target, split_and_preprocess_data
from src.benchmark_engine import run_benchmark_suite
from src.recommender import calculate_model_recommendations, WeightProfile

def run_all():
    results = {}

    # 1. Classification: Wisconsin Breast Cancer
    bc = load_breast_cancer(as_frame=True).frame
    X_bc, y_bc = separate_features_and_target(bc, "target")
    prep_bc = split_and_preprocess_data(X_bc, y_bc, "classification", test_size=0.2, random_state=42)
    models_cls = ["logistic_regression", "decision_tree", "random_forest", "knn", "gradient_boosting"]
    suite_bc = run_benchmark_suite(
        prep_result=prep_bc,
        selected_model_ids=models_cls,
        random_state=42,
        track_emissions=False,
        dataset_name="Wisconsin Breast Cancer Diagnostic",
    )

    profiles = {
        "Balanced": WeightProfile.balanced(),
        "Green-First": WeightProfile.green_first(),
        "Accuracy-First": WeightProfile.accuracy_first(),
        "Edge-Deployment": WeightProfile.edge_deployment(),
    }

    results["classification"] = {
        "dataset": "Wisconsin Breast Cancer Diagnostic",
        "n_samples": int(bc.shape[0]),
        "n_features": int(bc.shape[1] - 1),
        "train_samples": int(prep_bc.train_shape[0]),
        "test_samples": int(prep_bc.test_shape[0]),
        "models": [],
        "profiles": {},
    }

    for m in suite_bc.models_evaluated:
        results["classification"]["models"].append({
            "model_id": m.model_id,
            "model_name": m.model_name,
            "status": m.status,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "training_time_sec": m.training_time_sec,
            "prediction_time_sec": m.prediction_time_sec,
            "latency_per_sample_ms": m.latency_per_sample_ms,
            "baseline_ram_mb": m.baseline_ram_mb,
            "ram_current_mb": m.ram_current_mb,
            "ram_delta_mb": m.ram_delta_mb,
            "carbon_emissions_g": m.carbon_emissions_g,
            "energy_status": m.energy_status,
        })

    for pname, prof in profiles.items():
        rec = calculate_model_recommendations(suite_bc, prof)
        results["classification"]["profiles"][pname] = {
            "top_model": rec.top_recommended_model.model_name if rec.top_recommended_model else None,
            "top_score": rec.top_recommended_model.composite_score if rec.top_recommended_model else None,
            "pareto_models": rec.pareto_frontier_models,
            "effective_weights": rec.effective_weights,
            "redistribution_note": rec.redistribution_note,
            "ranked": [
                {
                    "rank": rm.rank,
                    "model_id": rm.model_id,
                    "model_name": rm.model_name,
                    "composite_score": rm.composite_score,
                    "is_pareto": rm.is_pareto_optimal,
                    "norm_scores": rm.normalized_scores,
                }
                for rm in rec.ranked_models
            ]
        }

    # 2. Regression: Diabetes Progression
    diab = load_diabetes(as_frame=True).frame
    X_diab, y_diab = separate_features_and_target(diab, "target")
    prep_diab = split_and_preprocess_data(X_diab, y_diab, "regression", test_size=0.2, random_state=42)
    models_reg = ["linear_regression", "decision_tree", "random_forest", "knn", "gradient_boosting"]
    suite_diab = run_benchmark_suite(
        prep_result=prep_diab,
        selected_model_ids=models_reg,
        random_state=42,
        track_emissions=False,
        dataset_name="Diabetes Progression Benchmark",
    )

    results["regression"] = {
        "dataset": "Diabetes Progression Benchmark",
        "n_samples": int(diab.shape[0]),
        "n_features": int(diab.shape[1] - 1),
        "train_samples": int(prep_diab.train_shape[0]),
        "test_samples": int(prep_diab.test_shape[0]),
        "models": [],
        "profiles": {},
    }

    for m in suite_diab.models_evaluated:
        results["regression"]["models"].append({
            "model_id": m.model_id,
            "model_name": m.model_name,
            "status": m.status,
            "r2": m.r2,
            "mae": m.mae,
            "rmse": m.rmse,
            "training_time_sec": m.training_time_sec,
            "prediction_time_sec": m.prediction_time_sec,
            "latency_per_sample_ms": m.latency_per_sample_ms,
            "baseline_ram_mb": m.baseline_ram_mb,
            "ram_current_mb": m.ram_current_mb,
            "ram_delta_mb": m.ram_delta_mb,
            "carbon_emissions_g": m.carbon_emissions_g,
            "energy_status": m.energy_status,
        })

    for pname, prof in profiles.items():
        rec = calculate_model_recommendations(suite_diab, prof)
        results["regression"]["profiles"][pname] = {
            "top_model": rec.top_recommended_model.model_name if rec.top_recommended_model else None,
            "top_score": rec.top_recommended_model.composite_score if rec.top_recommended_model else None,
            "pareto_models": rec.pareto_frontier_models,
            "effective_weights": rec.effective_weights,
            "redistribution_note": rec.redistribution_note,
            "ranked": [
                {
                    "rank": rm.rank,
                    "model_id": rm.model_id,
                    "model_name": rm.model_name,
                    "composite_score": rm.composite_score,
                    "is_pareto": rm.is_pareto_optimal,
                    "norm_scores": rm.normalized_scores,
                }
                for rm in rec.ranked_models
            ]
        }

    with open("experiments/actual_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Successfully ran all experiments and saved to experiments/actual_benchmark_results.json")

if __name__ == "__main__":
    run_all()
