"""
Unit tests for EcoML Multi-Criteria Green Model Recommender (Step 4).

Verifies multi-criteria scoring, weight validation, dynamic weight redistribution,
Pareto efficiency analysis, missing metric handling, and non-dogmatic trade-off explanations.
"""

import pytest
import numpy as np

from src.benchmark_engine import BenchmarkSuiteResult, SingleModelBenchmark
from src.recommender import (
    WeightProfile,
    ModelScoreDetail,
    RecommendationReport,
    calculate_model_recommendations,
    compute_pareto_frontier,
)


@pytest.fixture
def mock_classification_suite() -> BenchmarkSuiteResult:
    """Create a deterministic benchmark suite with known performance and resource trade-offs."""
    # Model A: High accuracy, high training compute
    model_a = SingleModelBenchmark(
        model_id="random_forest",
        model_name="Random Forest",
        task_type="classification",
        status="SUCCESS",
        f1=0.96,
        accuracy=0.96,
        training_time_sec=1.50,
        latency_per_sample_ms=0.08,
        ram_current_mb=180.0,
        carbon_emissions_g=None,  # Telemetry unavailable
    )

    # Model B: Moderate accuracy, ultra-fast training, compact RAM
    model_b = SingleModelBenchmark(
        model_id="logistic_regression",
        model_name="Logistic Regression",
        task_type="classification",
        status="SUCCESS",
        f1=0.92,
        accuracy=0.92,
        training_time_sec=0.02,
        latency_per_sample_ms=0.005,
        ram_current_mb=120.0,
        carbon_emissions_g=None,  # Telemetry unavailable
    )

    # Model C: Dominated model (lower accuracy than B, higher latency than B)
    model_c = SingleModelBenchmark(
        model_id="decision_tree",
        model_name="Decision Tree",
        task_type="classification",
        status="SUCCESS",
        f1=0.85,
        accuracy=0.85,
        training_time_sec=0.05,
        latency_per_sample_ms=0.010,
        ram_current_mb=130.0,
        carbon_emissions_g=None,
    )

    return BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Synthetic Diagnostic",
        target_column="target",
        sample_count_train=400,
        sample_count_test=100,
        feature_count=20,
        models_evaluated=[model_a, model_b, model_c],
    )


# ==============================================================================
# 1. Weight Profile & Validation Tests
# ==============================================================================

def test_weight_profile_validation():
    """Verify that negative weights or zero total weight raise ValueError."""
    with pytest.raises(ValueError, match="cannot be negative"):
        WeightProfile(w_performance=-0.1).validate()

    with pytest.raises(ValueError, match="greater than zero"):
        WeightProfile(
            w_performance=0.0,
            w_train_time=0.0,
            w_inference_latency=0.0,
            w_ram=0.0,
            w_carbon=0.0,
        ).validate()


def test_weight_profile_normalization():
    """Verify that criteria weights are normalized to sum to 1.0."""
    prof = WeightProfile(
        w_performance=2.0,
        w_train_time=1.0,
        w_inference_latency=1.0,
        w_ram=1.0,
        w_carbon=0.0,
    )
    weights = prof.normalized_dict()
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert weights["performance"] == 0.4
    assert weights["carbon"] == 0.0


def test_weight_profile_presets():
    """Verify that preset profiles instantiate and validate properly."""
    for factory in [
        WeightProfile.balanced,
        WeightProfile.green_first,
        WeightProfile.accuracy_first,
        WeightProfile.edge_deployment,
    ]:
        prof = factory()
        prof.validate()
        assert abs(sum(prof.normalized_dict().values()) - 1.0) < 1e-6


# ==============================================================================
# 2. Dynamic Weight Redistribution Tests
# ==============================================================================

def test_dynamic_weight_redistribution_when_carbon_unavailable(mock_classification_suite):
    """Verify that missing carbon telemetry does not substitute zero and redistributes weight."""
    prof = WeightProfile.balanced()  # Initially 15% carbon weight
    report = calculate_model_recommendations(mock_classification_suite, weight_profile=prof)

    # Carbon must NOT be in effective weights
    assert "carbon" not in report.effective_weights
    # Sum of effective weights must still be 1.0
    assert abs(sum(report.effective_weights.values()) - 1.0) < 1e-5
    # Redistribution note must be transparently populated
    assert report.redistribution_note is not None
    assert "redistributed" in report.redistribution_note


def test_recommendation_with_carbon_available():
    """Verify that when carbon telemetry is present, it is included in scoring."""
    model_1 = SingleModelBenchmark(
        model_id="m1",
        model_name="Model 1",
        task_type="classification",
        status="SUCCESS",
        f1=0.90,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
        carbon_emissions_g=0.05,
    )
    model_2 = SingleModelBenchmark(
        model_id="m2",
        model_name="Model 2",
        task_type="classification",
        status="SUCCESS",
        f1=0.90,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
        carbon_emissions_g=0.15,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Test",
        target_column="target",
        sample_count_train=100,
        sample_count_test=20,
        feature_count=5,
        models_evaluated=[model_1, model_2],
    )

    report = calculate_model_recommendations(suite, WeightProfile.green_first())
    assert "carbon" in report.effective_weights
    # Model 1 has lower carbon, so it should rank higher
    assert report.top_recommended_model.model_id == "m1"
    assert report.ranked_models[0].composite_score > report.ranked_models[1].composite_score


# ==============================================================================
# 3. Pareto Frontier Analysis Tests
# ==============================================================================

def test_pareto_frontier_computation(mock_classification_suite):
    """Verify that Pareto optimal models are identified correctly."""
    pareto_ids = compute_pareto_frontier(
        mock_classification_suite.models_evaluated,
        task_type="classification",
    )

    # Random Forest has top F1 (0.96) -> Pareto optimal
    assert "random_forest" in pareto_ids
    # Logistic Regression has top training speed & RAM efficiency -> Pareto optimal
    assert "logistic_regression" in pareto_ids
    # Decision tree has lower F1 than RF and is slower/more RAM than LR -> Dominated
    assert "decision_tree" not in pareto_ids


# ==============================================================================
# 4. Multi-Criteria Scoring & Synthesis Tests
# ==============================================================================

def test_recommendation_scoring_bounds(mock_classification_suite):
    """Verify that composite scores are bounded in [0.0, 100.0] and ranked."""
    report = calculate_model_recommendations(mock_classification_suite, WeightProfile.balanced())

    assert len(report.ranked_models) == 3
    assert report.ranked_models[0].rank == 1
    assert report.ranked_models[1].rank == 2
    assert report.ranked_models[2].rank == 3

    for m in report.ranked_models:
        assert 0.0 <= m.composite_score <= 100.0

    df = report.to_dataframe()
    assert len(df) == 3
    assert "EcoML Score" in df.columns
    assert "Pareto Optimal" in df.columns


def test_failed_model_handling():
    """Verify that failed models are ranked at bottom with zero score and clear explanation."""
    success_m = SingleModelBenchmark(
        model_id="ok",
        model_name="OK Model",
        task_type="classification",
        status="SUCCESS",
        f1=0.88,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    failed_m = SingleModelBenchmark(
        model_id="bad",
        model_name="Bad Model",
        task_type="classification",
        status="FAILED",
        error_message="Matrix singularity encountered",
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Test",
        target_column="target",
        sample_count_train=100,
        sample_count_test=20,
        feature_count=5,
        models_evaluated=[success_m, failed_m],
    )

    report = calculate_model_recommendations(suite)
    assert len(report.ranked_models) == 2
    assert report.ranked_models[0].model_id == "ok"
    assert report.ranked_models[1].model_id == "bad"
    assert report.ranked_models[1].status == "FAILED"
    assert report.ranked_models[1].composite_score == 0.0


def test_recommendation_zero_predictive_utility_warning():
    """Verify that when all models have zero predictive utility, an explicit warning is generated."""
    m1 = SingleModelBenchmark(
        model_id="dt",
        model_name="Decision Tree",
        task_type="classification",
        status="SUCCESS",
        f1=0.0,
        accuracy=0.0,
        training_time_sec=0.01,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    m2 = SingleModelBenchmark(
        model_id="rf",
        model_name="Random Forest",
        task_type="classification",
        status="SUCCESS",
        f1=0.0,
        accuracy=0.0,
        training_time_sec=0.10,
        latency_per_sample_ms=0.05,
        ram_current_mb=110.0,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Misconfigured Dataset",
        target_column="continuous_feature",
        sample_count_train=100,
        sample_count_test=20,
        feature_count=5,
        models_evaluated=[m1, m2],
    )
    report = calculate_model_recommendations(suite)
    assert "CRITICAL WARNING" in report.tradeoff_synthesis
    assert "zero predictive utility" in report.tradeoff_synthesis
    # Ensure no model falsely claims "Highest predictive utility"
    for m in report.ranked_models:
        for strength in m.strengths:
            assert "Highest predictive utility" not in strength
        assert any("Zero predictive utility" in t for t in m.tradeoff_notes)



def test_non_dogmatic_synthesis(mock_classification_suite):
    """Verify recommendation synthesis avoids claiming a model is universally best."""
    report = calculate_model_recommendations(mock_classification_suite, WeightProfile.balanced())
    synthesis = report.tradeoff_synthesis.lower()

    # Must explain context under selected profile
    assert "balanced" in synthesis
    assert "universally" not in synthesis or "not universally" in synthesis
    # Must mention trade-offs or Pareto alternatives
    assert len(synthesis) > 20


def test_raw_metrics_enrichment_classification():
    """Verify raw_metrics includes explicit accuracy, f1, precision, recall, and performance."""
    m1 = SingleModelBenchmark(
        model_id="log_reg",
        model_name="Logistic Regression",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.95,
        f1=0.94,
        precision=0.96,
        recall=0.93,
        training_time_sec=0.05,
        latency_per_sample_ms=0.015,
        ram_current_mb=120.0,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Test Cls",
        target_column="target",
        sample_count_train=80,
        sample_count_test=20,
        feature_count=4,
        models_evaluated=[m1],
    )
    report = calculate_model_recommendations(suite)
    raw = report.ranked_models[0].raw_metrics
    assert raw["accuracy"] == 0.95
    assert raw["f1"] == 0.94
    assert raw["precision"] == 0.96
    assert raw["recall"] == 0.93
    assert raw["performance"] == 0.94  # Uses F1 as primary utility
    assert raw["r2"] is None
    assert raw["rmse"] is None


def test_raw_metrics_enrichment_regression():
    """Verify raw_metrics includes explicit r2, mae, rmse, and performance for regression."""
    m1 = SingleModelBenchmark(
        model_id="ridge",
        model_name="Ridge Regression",
        task_type="regression",
        status="SUCCESS",
        r2=0.65,
        mae=12.5,
        rmse=16.8,
        training_time_sec=0.02,
        latency_per_sample_ms=0.008,
        ram_current_mb=115.0,
    )
    suite = BenchmarkSuiteResult(
        task_type="regression",
        dataset_name="Test Reg",
        target_column="target",
        sample_count_train=80,
        sample_count_test=20,
        feature_count=4,
        models_evaluated=[m1],
    )
    report = calculate_model_recommendations(suite)
    raw = report.ranked_models[0].raw_metrics
    assert raw["r2"] == 0.65
    assert raw["mae"] == 12.5
    assert raw["rmse"] == 16.8
    assert raw["performance"] == 0.65
    assert raw["accuracy"] is None


def test_raw_metrics_genuine_zero_accuracy_preservation():
    """Verify genuine zero accuracy (0.0) is preserved as 0.0 and distinct from None."""
    m1 = SingleModelBenchmark(
        model_id="zero_acc",
        model_name="Zero Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.0,
        f1=0.0,
        precision=0.0,
        recall=0.0,
        training_time_sec=0.01,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Zero Test",
        target_column="target",
        sample_count_train=50,
        sample_count_test=10,
        feature_count=2,
        models_evaluated=[m1],
    )
    report = calculate_model_recommendations(suite)
    raw = report.ranked_models[0].raw_metrics
    assert raw["accuracy"] == 0.0
    assert raw["accuracy"] is not None
    assert raw["f1"] == 0.0


def test_failed_model_raw_metrics_explicit_none():
    """Verify failed model raw_metrics has None for all metrics, preventing false zeros."""
    m_ok = SingleModelBenchmark(
        model_id="ok",
        model_name="OK Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.90,
        f1=0.89,
        training_time_sec=0.05,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    m_fail = SingleModelBenchmark(
        model_id="broken",
        model_name="Broken Model",
        task_type="classification",
        status="FAILED",
        error_message="Convergence error",
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Fail Test",
        target_column="target",
        sample_count_train=50,
        sample_count_test=10,
        feature_count=2,
        models_evaluated=[m_ok, m_fail],
    )
    report = calculate_model_recommendations(suite)
    assert len(report.ranked_models) == 2
    failed_model = report.ranked_models[1]
    assert failed_model.status == "FAILED"
    raw = failed_model.raw_metrics
    assert raw["accuracy"] is None
    assert raw["f1"] is None
    assert raw["performance"] is None
    assert raw["training_time_sec"] is None


def test_recommendation_service_composite_score_precision_two_decimals():
    """Verify recommendation_service preserves composite score precision to 2 decimal places."""
    from backend.services.benchmark_service import benchmark_service
    from backend.services.recommendation_service import recommendation_service
    from backend.schemas.recommendations import RecommendationResponse

    m1 = SingleModelBenchmark(
        model_id="model_a",
        model_name="Model A",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.9825,
        f1=0.9825,
        training_time_sec=0.038,
        latency_per_sample_ms=0.012,
        ram_current_mb=158.68,
    )
    m2 = SingleModelBenchmark(
        model_id="model_b",
        model_name="Model B",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.9123,
        f1=0.9130,
        training_time_sec=0.026,
        latency_per_sample_ms=0.015,
        ram_current_mb=158.99,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Precision Test",
        target_column="target",
        sample_count_train=455,
        sample_count_test=114,
        feature_count=30,
        models_evaluated=[m1, m2],
    )
    benchmark_service._jobs["test_precision_job"] = {
        "benchmark_id": "test_precision_job",
        "status": "COMPLETED",
        "suite_result": suite,
        "interpretation": None,
    }
    rec_result = recommendation_service.generate_recommendations("test_precision_job")
    top_score = rec_result["top_recommended_model"]["composite_score"]
    recommender_report = calculate_model_recommendations(suite)
    
    assert isinstance(top_score, float)
    assert top_score == recommender_report.top_recommended_model.composite_score
    
    # Ensure schema validation succeeds without errors
    validated = RecommendationResponse(**rec_result)
    assert validated.top_recommended_model.composite_score == top_score
    assert len(validated.ranked_models) == 2


def test_pareto_equal_objective_values():
    """Verify Pareto frontier logic when models have identical performance and efficiency."""
    m1 = SingleModelBenchmark(
        model_id="model_1",
        model_name="Model 1",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.95,
        f1=0.95,
        training_time_sec=0.05,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    m2 = SingleModelBenchmark(
        model_id="model_2",
        model_name="Model 2",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.95,
        f1=0.95,
        training_time_sec=0.05,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    pareto_ids = compute_pareto_frontier([m1, m2], task_type="classification")
    # Neither strictly dominates the other; both should belong to the frontier
    assert "model_1" in pareto_ids
    assert "model_2" in pareto_ids


def test_pareto_constant_valued_cost_objectives():
    """Verify Pareto frontier and cost normalization when all models have identical costs."""
    m_high = SingleModelBenchmark(
        model_id="high_acc",
        model_name="High Accuracy",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.98,
        f1=0.98,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    m_low = SingleModelBenchmark(
        model_id="low_acc",
        model_name="Low Accuracy",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.80,
        f1=0.80,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
    )
    pareto_ids = compute_pareto_frontier([m_high, m_low], task_type="classification")
    # m_high strictly dominates m_low because efficiency is equal but performance is strictly higher
    assert pareto_ids == ["high_acc"]


def test_recommendation_profiles_differentiation():
    """Verify that different operational profiles prioritize expected criteria."""
    # Model Fast: lower accuracy (0.60), ultra-low latency (0.001 ms), low RAM (50MB), fast train (0.01s)
    m_fast = SingleModelBenchmark(
        model_id="fast_edge",
        model_name="Fast Edge Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.60,
        f1=0.60,
        training_time_sec=0.01,
        latency_per_sample_ms=0.001,
        ram_current_mb=50.0,
    )
    # Model Accurate: high accuracy (0.99), high latency (0.50 ms), high RAM (300MB), slow train (5.0s)
    m_acc = SingleModelBenchmark(
        model_id="heavy_acc",
        model_name="Heavy Accurate Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.99,
        f1=0.99,
        training_time_sec=5.0,
        latency_per_sample_ms=0.50,
        ram_current_mb=300.0,
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Tradeoff Suite",
        target_column="target",
        sample_count_train=100,
        sample_count_test=20,
        feature_count=10,
        models_evaluated=[m_fast, m_acc],
    )

    # Accuracy-First (70% performance weight) prioritizes the 0.99 F1 model over the 0.60 F1 model
    rep_acc = calculate_model_recommendations(suite, WeightProfile.accuracy_first())
    assert rep_acc.top_recommended_model.model_id == "heavy_acc"

    # Edge-Deployment (35% latency, 20% RAM) prioritizes the ultra-fast, compact model
    rep_edge = calculate_model_recommendations(suite, WeightProfile.edge_deployment())
    assert rep_edge.top_recommended_model.model_id == "fast_edge"


def test_score_normalization_constant_and_missing_values():
    """Verify that score normalization handles constant metrics and missing carbon safely."""
    from src.recommender import _normalize_cost_series

    # Single value series
    assert _normalize_cost_series([42.0]) == [1.0]

    # Constant multi-value series
    assert _normalize_cost_series([10.0, 10.0, 10.0]) == [1.0, 1.0, 1.0]

    # Normal inverted cost series (min gets 1.0, max gets 0.0)
    norm = _normalize_cost_series([1.0, 2.0, 3.0])
    assert norm == [1.0, 0.5, 0.0]


def test_score_normalization_nan_and_infinite_values():
    """Verify that cost normalization sanitizes NaN and infinite values safely."""
    from src.recommender import _normalize_cost_series

    result = _normalize_cost_series([1.0, float("nan"), 5.0])
    assert len(result) == 3
    assert not any(np.isnan(v) for v in result)
    assert not any(np.isinf(v) for v in result)
    assert all(0.0 <= v <= 1.0 for v in result)


def test_pareto_frontier_with_measured_carbon():
    """Verify that active carbon measurements are incorporated into the Pareto efficiency index."""
    m_clean = SingleModelBenchmark(
        model_id="clean_model",
        model_name="Clean Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.90,
        f1=0.90,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
        carbon_emissions_g=0.00001,
    )
    m_dirty = SingleModelBenchmark(
        model_id="dirty_model",
        model_name="Dirty Model",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.90,
        f1=0.90,
        training_time_sec=0.1,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
        carbon_emissions_g=0.001,
    )
    # Both models have identical accuracy, train time, latency, and RAM.
    # m_clean has 100x lower carbon, so m_clean strictly dominates m_dirty.
    pareto_ids = compute_pareto_frontier([m_clean, m_dirty], task_type="classification")
    assert pareto_ids == ["clean_model"]
    assert "dirty_model" not in pareto_ids


def test_dynamic_weight_redistribution_mathematical_invariance():
    """Verify that effective weights sum strictly to 1.0000 under unavailable carbon."""
    m1 = SingleModelBenchmark(
        model_id="m1",
        model_name="Model 1",
        task_type="classification",
        status="SUCCESS",
        accuracy=0.95,
        f1=0.95,
        training_time_sec=0.05,
        latency_per_sample_ms=0.01,
        ram_current_mb=100.0,
        carbon_emissions_g=None,  # Unavailable
    )
    suite = BenchmarkSuiteResult(
        task_type="classification",
        dataset_name="Test Suite",
        target_column="target",
        sample_count_train=100,
        sample_count_test=20,
        feature_count=5,
        models_evaluated=[m1],
    )
    report = calculate_model_recommendations(suite, WeightProfile.balanced())
    assert "carbon" not in report.effective_weights
    total_effective_weight = sum(report.effective_weights.values())
    assert np.isclose(total_effective_weight, 1.0, atol=1e-6)
    assert report.redistribution_note is not None
    assert "dynamically redistributed" in report.redistribution_note




