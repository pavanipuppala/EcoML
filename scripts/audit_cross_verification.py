"""
EcoML — Controlled Numerical Cross-Verification and Mathematical Audit Script.

This script independently re-executes data preprocessing, model fitting, metric computation,
MCDA multi-criteria score aggregation, and Pareto frontier determination.
It compares independently derived results with:
1. `src/benchmark_engine.py` (Empirical engine)
2. `src/recommender.py` (MCDA engine)
3. `experiments/actual_benchmark_results.json` (Empirically recorded baseline)

All outputs are saved to `experiments/audit_recalculation_results.json`.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
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

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_preprocessing import (
    separate_features_and_target,
    split_and_preprocess_data,
    PreprocessingResult,
)
from src.benchmark_engine import (
    BenchmarkSuiteResult,
    SingleModelBenchmark,
    build_model_instance,
    benchmark_single_model,
    get_supported_models,
)
from src.recommender import (
    WeightProfile,
    calculate_model_recommendations,
    RecommendationReport,
    compute_pareto_frontier,
    _normalize_cost_series,
    _extract_performance_metric,
)


def independent_mcda_recalculation(
    models: List[Dict[str, Any]],
    task_type: str,
    profile: WeightProfile,
    has_carbon: bool = False,
) -> Dict[str, Any]:
    """Manually recalculates MCDA composite scores and Pareto dominance from pure mathematical principles."""
    weights = profile.normalized_dict()
    if not has_carbon:
        # Dynamic redistribution
        rem = sum(w for k, w in weights.items() if k != "carbon")
        eff_weights = {k: w / rem for k, w in weights.items() if k != "carbon"}
    else:
        eff_weights = dict(weights)

    # 1. Normalized predictive utility
    if task_type == "classification":
        # In recommender.py: val = run.f1 if run.f1 is not None else run.accuracy
        perfs = [m["f1"] if m.get("f1") is not None else m["accuracy"] for m in models]
    else:
        perfs = [max(0.0, min(1.0, m["r2"])) for m in models]

    # 2. Inverted normalized costs
    def norm_cost(vals):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [1.0 for _ in vals]
        return [(mx - v) / (mx - mn) for v in vals]

    tt = [m["training_time_sec"] for m in models]
    lat = [m["latency_per_sample_ms"] for m in models]
    ram = [m["ram_current_mb"] for m in models]

    norm_tt = norm_cost(tt)
    norm_lat = norm_cost(lat)
    norm_ram = norm_cost(ram)

    norm_carb = None
    if has_carbon:
        carb = [m["carbon_emissions_g"] for m in models]
        norm_carb = norm_cost(carb)

    scores = []
    for i, m in enumerate(models):
        score_parts = {
            "performance": perfs[i] * eff_weights["performance"],
            "train_time": norm_tt[i] * eff_weights["train_time"],
            "inference_latency": norm_lat[i] * eff_weights["inference_latency"],
            "ram": norm_ram[i] * eff_weights["ram"],
        }
        if has_carbon and norm_carb is not None:
            score_parts["carbon"] = norm_carb[i] * eff_weights["carbon"]
        composite = sum(score_parts.values()) * 100.0
        scores.append({
            "model_id": m["model_id"],
            "composite_score_raw": composite,
            "composite_score_rounded": round(composite, 2),
            "normalized_scores": {
                "performance": perfs[i],
                "train_time": norm_tt[i],
                "inference_latency": norm_lat[i],
                "ram": norm_ram[i],
                **({"carbon": norm_carb[i]} if norm_carb else {}),
            },
        })

    # Pareto optimality: Predictive Utility vs (norm_tt + norm_lat + norm_ram) / 3.0
    eff_indices = [(norm_tt[i] + norm_lat[i] + norm_ram[i]) / 3.0 for i in range(len(models))]
    pareto_set = []
    for i in range(len(models)):
        perf_a = perfs[i]
        eff_a = eff_indices[i]
        dominated = False
        for j in range(len(models)):
            if i == j:
                continue
            perf_b = perfs[j]
            eff_b = eff_indices[j]
            if (perf_b >= perf_a and eff_b >= eff_a) and (perf_b > perf_a or eff_b > eff_a):
                dominated = True
                break
        if not dominated:
            pareto_set.append(models[i]["model_id"])

    return {
        "effective_weights": eff_weights,
        "scores": scores,
        "pareto_frontier": pareto_set,
        "efficiency_indices": eff_indices,
    }


def audit_classification() -> Dict[str, Any]:
    print("=" * 60)
    print("RUNNING CLASSIFICATION AUDIT: Wisconsin Breast Cancer")
    print("=" * 60)

    # 1. Dataset verification
    raw_data = load_breast_cancer(as_frame=True)
    df = raw_data.frame
    raw_shape = list(df.shape)
    features_raw = list(raw_data.feature_names)
    target_name = "target"
    class_dist = {str(k): int(v) for k, v in df[target_name].value_counts().to_dict().items()}

    print(f"Raw shape: {raw_shape}, Features: {len(features_raw)}, Target: '{target_name}'")
    print(f"Class distribution: {class_dist}")

    # 2. Preprocessing via separate_features_and_target + split_and_preprocess_data
    X, y = separate_features_and_target(df, target_column=target_name)
    prep_res: PreprocessingResult = split_and_preprocess_data(
        X,
        y,
        task_type="classification",
        test_size=0.2,
        random_state=42,
        scale_numerical=True,
    )

    feature_count = int(prep_res.train_shape[1])
    print(f"Preprocessed train shape: {prep_res.train_shape}, test shape: {prep_res.test_shape}")
    print(f"Preprocessed features count: {feature_count}")

    # 3. Model benchmarking and independent evaluation
    models_to_test = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "knn",
        "gradient_boosting",
    ]

    model_evaluations = []
    benchmark_runs: List[SingleModelBenchmark] = []

    for model_id in models_to_test:
        print(f"--> Benchmarking {model_id}...")
        model_inst = build_model_instance(model_id, "classification", random_state=42)

        bench_run = benchmark_single_model(
            model_id=model_id,
            model_name=get_supported_models("classification")[model_id],
            model_instance=model_inst,
            X_train=prep_res.X_train_processed,
            y_train=prep_res.y_train,
            X_test=prep_res.X_test_processed,
            y_test=prep_res.y_test,
            task_type="classification",
            track_emissions=False,
        )
        benchmark_runs.append(bench_run)

        # Independent calculation on test predictions
        y_pred = bench_run.model_instance.predict(prep_res.X_test_processed)
        y_true = prep_res.y_test

        indep_acc = float(accuracy_score(y_true, y_pred))
        indep_prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        indep_prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        indep_rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        indep_rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        indep_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        indep_f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        indep_cm = confusion_matrix(y_true, y_pred).tolist()

        eval_record = {
            "model_id": model_id,
            "model_name": bench_run.model_name,
            "independent": {
                "accuracy": indep_acc,
                "precision_weighted": indep_prec_weighted,
                "precision_macro": indep_prec_macro,
                "recall_weighted": indep_rec_weighted,
                "recall_macro": indep_rec_macro,
                "f1_weighted": indep_f1_weighted,
                "f1_macro": indep_f1_macro,
                "confusion_matrix": indep_cm,
            },
            "benchmark_engine": {
                "accuracy": bench_run.accuracy,
                "precision": bench_run.precision,
                "recall": bench_run.recall,
                "f1": bench_run.f1,
                "confusion_matrix": bench_run.confusion_matrix,
                "training_time_sec": bench_run.training_time_sec,
                "latency_per_sample_ms": bench_run.latency_per_sample_ms,
                "ram_current_mb": bench_run.ram_current_mb,
                "ram_delta_mb": bench_run.ram_delta_mb,
                "baseline_ram_mb": bench_run.baseline_ram_mb,
                "energy_status": bench_run.energy_status,
                "carbon_emissions_g": bench_run.carbon_emissions_g,
            },
            "deltas": {
                "accuracy_diff": abs(indep_acc - (bench_run.accuracy or 0.0)),
                "precision_weighted_diff": abs(indep_prec_weighted - (bench_run.precision or 0.0)),
                "f1_weighted_diff": abs(indep_f1_weighted - (bench_run.f1 or 0.0)),
                "f1_macro_vs_engine_f1": abs(indep_f1_macro - (bench_run.f1 or 0.0)),
            },
        }
        model_evaluations.append(eval_record)
        print(f"    Acc: {indep_acc:.6f}, F1(weighted): {indep_f1_weighted:.6f}, F1(macro): {indep_f1_macro:.6f}")

    suite_result = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Wisconsin Breast Cancer",
        target_column=target_name,
        sample_count_train=prep_res.train_shape[0],
        sample_count_test=prep_res.test_shape[0],
        feature_count=feature_count,
        models_evaluated=benchmark_runs,
    )

    # 4. Multi-Criteria Recommendations across Profiles
    profiles = [
        WeightProfile.balanced(),
        WeightProfile.green_first(),
        WeightProfile.accuracy_first(),
        WeightProfile.edge_deployment(),
    ]

    profile_reports = {}
    for prof in profiles:
        print(f"--> Computing recommendations for profile: {prof.name}")
        report: RecommendationReport = calculate_model_recommendations(suite_result, weight_profile=prof)

        raw_model_inputs = [
            {
                "model_id": b.model_id,
                "accuracy": b.accuracy,
                "f1": b.f1,
                "training_time_sec": b.training_time_sec,
                "latency_per_sample_ms": b.latency_per_sample_ms,
                "ram_current_mb": b.ram_current_mb,
                "carbon_emissions_g": b.carbon_emissions_g,
            }
            for b in benchmark_runs
        ]
        indep_mcda = independent_mcda_recalculation(
            raw_model_inputs,
            task_type="classification",
            profile=prof,
            has_carbon=False,
        )

        ranked_comparison = []
        for ranked in report.ranked_models:
            indep_match = next((s for s in indep_mcda["scores"] if s["model_id"] == ranked.model_id), None)
            ranked_comparison.append({
                "model_id": ranked.model_id,
                "rank": ranked.rank,
                "engine_composite_score": ranked.composite_score,
                "indep_composite_score_raw": indep_match["composite_score_raw"] if indep_match else None,
                "indep_composite_score_rounded": indep_match["composite_score_rounded"] if indep_match else None,
                "score_diff": abs(ranked.composite_score - indep_match["composite_score_rounded"]) if indep_match else None,
                "engine_is_pareto": ranked.is_pareto_optimal,
                "indep_is_pareto": (ranked.model_id in indep_mcda["pareto_frontier"]),
                "effective_weights": ranked.effective_weights,
            })

        profile_reports[prof.name] = {
            "effective_weights": report.effective_weights,
            "pareto_frontier_models": report.pareto_frontier_models,
            "top_recommended_model": report.top_recommended_model.model_id if report.top_recommended_model else None,
            "rankings": ranked_comparison,
        }

    return {
        "dataset": {
            "name": "Wisconsin Breast Cancer",
            "task_type": "classification",
            "raw_shape": raw_shape,
            "train_shape": list(prep_res.train_shape),
            "test_shape": list(prep_res.test_shape),
            "feature_count": feature_count,
            "class_distribution": class_dist,
        },
        "model_evaluations": model_evaluations,
        "profile_recommendations": profile_reports,
    }


def audit_regression() -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("RUNNING REGRESSION AUDIT: Diabetes Progression")
    print("=" * 60)

    # 1. Dataset verification
    raw_data = load_diabetes(as_frame=True)
    df = raw_data.frame
    raw_shape = list(df.shape)
    features_raw = list(raw_data.feature_names)
    target_name = "target"

    print(f"Raw shape: {raw_shape}, Features: {len(features_raw)}, Target: '{target_name}'")
    print(f"Target statistics: min={df[target_name].min()}, max={df[target_name].max()}, mean={df[target_name].mean():.2f}")

    # 2. Preprocessing via separate_features_and_target + split_and_preprocess_data
    X, y = separate_features_and_target(df, target_column=target_name)
    prep_res: PreprocessingResult = split_and_preprocess_data(
        X,
        y,
        task_type="regression",
        test_size=0.2,
        random_state=42,
        scale_numerical=True,
    )

    feature_count = int(prep_res.train_shape[1])
    print(f"Preprocessed train shape: {prep_res.train_shape}, test shape: {prep_res.test_shape}")
    print(f"Preprocessed features count: {feature_count}")

    # 3. Model benchmarking and independent evaluation
    models_to_test = [
        "linear_regression",
        "decision_tree",
        "random_forest",
        "knn",
        "gradient_boosting",
    ]

    model_evaluations = []
    benchmark_runs: List[SingleModelBenchmark] = []

    for model_id in models_to_test:
        print(f"--> Benchmarking {model_id}...")
        model_inst = build_model_instance(model_id, "regression", random_state=42)

        bench_run = benchmark_single_model(
            model_id=model_id,
            model_name=get_supported_models("regression")[model_id],
            model_instance=model_inst,
            X_train=prep_res.X_train_processed,
            y_train=prep_res.y_train,
            X_test=prep_res.X_test_processed,
            y_test=prep_res.y_test,
            task_type="regression",
            track_emissions=False,
        )
        benchmark_runs.append(bench_run)

        # Independent calculation on test predictions
        y_pred = bench_run.model_instance.predict(prep_res.X_test_processed)
        y_true = prep_res.y_test

        indep_r2 = float(r2_score(y_true, y_pred))
        indep_mae = float(mean_absolute_error(y_true, y_pred))
        indep_mse = float(mean_squared_error(y_true, y_pred))
        indep_rmse = float(np.sqrt(indep_mse))

        eval_record = {
            "model_id": model_id,
            "model_name": bench_run.model_name,
            "independent": {
                "r2": indep_r2,
                "mae": indep_mae,
                "rmse": indep_rmse,
                "mse": indep_mse,
            },
            "benchmark_engine": {
                "r2": bench_run.r2,
                "mae": bench_run.mae,
                "rmse": bench_run.rmse,
                "training_time_sec": bench_run.training_time_sec,
                "latency_per_sample_ms": bench_run.latency_per_sample_ms,
                "ram_current_mb": bench_run.ram_current_mb,
                "ram_delta_mb": bench_run.ram_delta_mb,
                "baseline_ram_mb": bench_run.baseline_ram_mb,
                "energy_status": bench_run.energy_status,
                "carbon_emissions_g": bench_run.carbon_emissions_g,
            },
            "deltas": {
                "r2_diff": abs(indep_r2 - (bench_run.r2 or 0.0)),
                "mae_diff": abs(indep_mae - (bench_run.mae or 0.0)),
                "rmse_diff": abs(indep_rmse - (bench_run.rmse or 0.0)),
            },
        }
        model_evaluations.append(eval_record)
        print(f"    R2: {indep_r2:.6f}, MAE: {indep_mae:.4f}, RMSE: {indep_rmse:.4f}")

    suite_result = BenchmarkSuiteResult(
        task_type="regression",
        dataset_name="Diabetes Progression",
        target_column=target_name,
        sample_count_train=prep_res.train_shape[0],
        sample_count_test=prep_res.test_shape[0],
        feature_count=feature_count,
        models_evaluated=benchmark_runs,
    )

    # 4. Multi-Criteria Recommendations across Profiles
    profiles = [
        WeightProfile.balanced(),
        WeightProfile.green_first(),
        WeightProfile.accuracy_first(),
        WeightProfile.edge_deployment(),
    ]

    profile_reports = {}
    for prof in profiles:
        print(f"--> Computing recommendations for profile: {prof.name}")
        report: RecommendationReport = calculate_model_recommendations(suite_result, weight_profile=prof)

        raw_model_inputs = [
            {
                "model_id": b.model_id,
                "r2": b.r2,
                "training_time_sec": b.training_time_sec,
                "latency_per_sample_ms": b.latency_per_sample_ms,
                "ram_current_mb": b.ram_current_mb,
                "carbon_emissions_g": b.carbon_emissions_g,
            }
            for b in benchmark_runs
        ]
        indep_mcda = independent_mcda_recalculation(
            raw_model_inputs,
            task_type="regression",
            profile=prof,
            has_carbon=False,
        )

        ranked_comparison = []
        for ranked in report.ranked_models:
            indep_match = next((s for s in indep_mcda["scores"] if s["model_id"] == ranked.model_id), None)
            ranked_comparison.append({
                "model_id": ranked.model_id,
                "rank": ranked.rank,
                "engine_composite_score": ranked.composite_score,
                "indep_composite_score_raw": indep_match["composite_score_raw"] if indep_match else None,
                "indep_composite_score_rounded": indep_match["composite_score_rounded"] if indep_match else None,
                "score_diff": abs(ranked.composite_score - indep_match["composite_score_rounded"]) if indep_match else None,
                "engine_is_pareto": ranked.is_pareto_optimal,
                "indep_is_pareto": (ranked.model_id in indep_mcda["pareto_frontier"]),
                "effective_weights": ranked.effective_weights,
            })

        profile_reports[prof.name] = {
            "effective_weights": report.effective_weights,
            "pareto_frontier_models": report.pareto_frontier_models,
            "top_recommended_model": report.top_recommended_model.model_id if report.top_recommended_model else None,
            "rankings": ranked_comparison,
        }

    return {
        "dataset": {
            "name": "Diabetes Progression",
            "task_type": "regression",
            "raw_shape": raw_shape,
            "train_shape": list(prep_res.train_shape),
            "test_shape": list(prep_res.test_shape),
            "feature_count": feature_count,
        },
        "model_evaluations": model_evaluations,
        "profile_recommendations": profile_reports,
    }


def compare_with_saved_baseline(audit_data: Dict[str, Any]) -> Dict[str, Any]:
    baseline_path = os.path.join(PROJECT_ROOT, "experiments", "actual_benchmark_results.json")
    if not os.path.exists(baseline_path):
        print(f"Warning: Baseline file {baseline_path} not found.")
        return {"status": "BASELINE_NOT_FOUND"}

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    comparison = {}
    # Compare classification
    cls_baseline = baseline_data.get("classification", {}).get("models", [])
    cls_audit = audit_data.get("classification", {}).get("model_evaluations", [])

    cls_comp = []
    for aud in cls_audit:
        m_id = aud["model_id"]
        base_match = next((b for b in cls_baseline if b.get("model_id") == m_id), None)
        if base_match:
            cls_comp.append({
                "model_id": m_id,
                "accuracy_audit": aud["independent"]["accuracy"],
                "accuracy_baseline": base_match.get("accuracy"),
                "accuracy_diff": abs(aud["independent"]["accuracy"] - base_match.get("accuracy", 0.0)),
                "f1_weighted_audit": aud["independent"]["f1_weighted"],
                "f1_baseline": base_match.get("f1"),
                "f1_diff": abs(aud["independent"]["f1_weighted"] - base_match.get("f1", 0.0)),
                "training_time_audit": aud["benchmark_engine"]["training_time_sec"],
                "training_time_baseline": base_match.get("training_time_sec"),
            })
    comparison["classification_vs_saved_baseline"] = cls_comp

    # Compare regression
    reg_baseline = baseline_data.get("regression", {}).get("models", [])
    reg_audit = audit_data.get("regression", {}).get("model_evaluations", [])

    reg_comp = []
    for aud in reg_audit:
        m_id = aud["model_id"]
        base_match = next((b for b in reg_baseline if b.get("model_id") == m_id), None)
        if base_match:
            reg_comp.append({
                "model_id": m_id,
                "r2_audit": aud["independent"]["r2"],
                "r2_baseline": base_match.get("r2"),
                "r2_diff": abs(aud["independent"]["r2"] - base_match.get("r2", 0.0)),
                "mae_audit": aud["independent"]["mae"],
                "mae_baseline": base_match.get("mae"),
                "mae_diff": abs(aud["independent"]["mae"] - base_match.get("mae", 0.0)),
                "rmse_audit": aud["independent"]["rmse"],
                "rmse_baseline": base_match.get("rmse"),
                "rmse_diff": abs(aud["independent"]["rmse"] - base_match.get("rmse", 0.0)),
            })
    comparison["regression_vs_saved_baseline"] = reg_comp

    return comparison


def main():
    print("Beginning EcoML Complete Metrics Cross-Verification Audit Execution...")
    start_time = time.time()

    cls_audit = audit_classification()
    reg_audit = audit_regression()

    full_audit = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_duration_sec": round(time.time() - start_time, 2),
        "classification": cls_audit,
        "regression": reg_audit,
    }

    baseline_comparison = compare_with_saved_baseline(full_audit)
    full_audit["baseline_comparison"] = baseline_comparison

    output_path = os.path.join(PROJECT_ROOT, "experiments", "audit_recalculation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)

    print(f"\nAudit complete in {full_audit['audit_duration_sec']}s. Results saved to:\n  {output_path}")


if __name__ == "__main__":
    main()
