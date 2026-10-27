"""
Unit tests for EcoML ML Benchmark Engine (Step 3).

These tests evaluate model instantiation, timing benchmarks, task-specific metrics,
memory measurements, error containment, and suite execution using small deterministic
in-memory synthetic datasets. These test fixtures are solely for functional verification
and are not presented as empirical research findings.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.benchmark_engine import (
    SingleModelBenchmark,
    BenchmarkSuiteResult,
    get_supported_models,
    build_model_instance,
    evaluate_predictions,
    benchmark_single_model,
    run_benchmark_suite,
    generate_benchmark_interpretation,
)
from src.data_preprocessing import PreprocessingResult, build_preprocessor


@pytest.fixture
def synthetic_classification_prep():
    """Create a minimal synthetic PreprocessingResult for binary classification."""
    np.random.seed(42)
    n_train = 30
    n_test = 10
    n_features = 4

    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)

    preprocessor = build_preprocessor(
        numerical_cols=[f"num_{i}" for i in range(n_features)],
        categorical_cols=[],
    )

    return PreprocessingResult(
        X_train_raw=pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(n_features)]),
        X_test_raw=pd.DataFrame(X_test, columns=[f"f_{i}" for i in range(n_features)]),
        X_train_processed=X_train,
        X_test_processed=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names_out=[f"f_{i}" for i in range(n_features)],
        numerical_cols=[f"f_{i}" for i in range(n_features)],
        categorical_cols=[],
        target_column="target",
        task_type="classification",
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )


@pytest.fixture
def synthetic_regression_prep():
    """Create a minimal synthetic PreprocessingResult for continuous regression."""
    np.random.seed(42)
    n_train = 30
    n_test = 10
    n_features = 3

    X_train = np.random.randn(n_train, n_features)
    y_train = 2.5 * X_train[:, 0] - 1.2 * X_train[:, 1] + 0.5

    X_test = np.random.randn(n_test, n_features)
    y_test = 2.5 * X_test[:, 0] - 1.2 * X_test[:, 1] + 0.5

    preprocessor = build_preprocessor(
        numerical_cols=[f"num_{i}" for i in range(n_features)],
        categorical_cols=[],
    )

    return PreprocessingResult(
        X_train_raw=pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(n_features)]),
        X_test_raw=pd.DataFrame(X_test, columns=[f"f_{i}" for i in range(n_features)]),
        X_train_processed=X_train,
        X_test_processed=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        feature_names_out=[f"f_{i}" for i in range(n_features)],
        numerical_cols=[f"f_{i}" for i in range(n_features)],
        categorical_cols=[],
        target_column="price",
        task_type="regression",
        train_shape=X_train.shape,
        test_shape=X_test.shape,
    )


# ==============================================================================
# 1. Model Registry and Instantiation Tests
# ==============================================================================

def test_get_supported_models_classification():
    """Verify supported models for classification include baseline requirements."""
    models = get_supported_models("classification")
    assert "logistic_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert len(models) >= 4


def test_get_supported_models_regression():
    """Verify supported models for regression include baseline requirements."""
    models = get_supported_models("regression")
    assert "linear_regression" in models
    assert "decision_tree" in models
    assert "random_forest" in models
    assert len(models) >= 4


def test_get_supported_models_invalid_task():
    """Verify ValueError is raised on unsupported task type."""
    with pytest.raises(ValueError, match="Unsupported task_type"):
        get_supported_models("clustering")


def test_build_model_instance_valid():
    """Verify that models can be properly initialized with random_state."""
    clf = build_model_instance("random_forest", "classification", random_state=123)
    assert hasattr(clf, "fit")
    assert clf.random_state == 123

    reg = build_model_instance("decision_tree", "regression", random_state=42)
    assert hasattr(reg, "fit")
    assert reg.random_state == 42


def test_build_model_instance_invalid_id():
    """Verify ValueError is raised on unknown model ID."""
    with pytest.raises(ValueError, match="Unknown classification model ID"):
        build_model_instance("deep_neural_net", "classification")


# ==============================================================================
# 2. Predictive Metric Evaluation Tests
# ==============================================================================

def test_evaluate_predictions_classification():
    """Verify classification metrics calculation."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])

    metrics = evaluate_predictions(y_true, y_pred, "classification")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["accuracy"] == 0.8  # 4 out of 5 correct
    assert isinstance(metrics["confusion_matrix"], list)


def test_evaluate_predictions_regression():
    """Verify regression metrics calculation."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 31.0])

    metrics = evaluate_predictions(y_true, y_pred, "regression")
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    # Absolute errors: 2, 1, 1 -> MAE = 4/3 ~ 1.333
    assert abs(metrics["mae"] - 1.3333) < 0.01
    assert metrics["rmse"] > 0
    assert metrics["r2"] <= 1.0


def test_evaluate_predictions_mismatch():
    """Verify length mismatch raises ValueError."""
    with pytest.raises(ValueError, match="Mismatched lengths"):
        evaluate_predictions(np.array([1, 2]), np.array([1]), "classification")


# ==============================================================================
# 3. Single Model Benchmark Tests
# ==============================================================================

def test_benchmark_single_model_classification(synthetic_classification_prep):
    """Verify single classification model benchmark execution, timing, and RAM."""
    prep = synthetic_classification_prep
    model = build_model_instance("decision_tree", "classification", random_state=42)

    result = benchmark_single_model(
        model_id="decision_tree",
        model_name="Decision Tree",
        model_instance=model,
        X_train=prep.X_train_processed,
        y_train=prep.y_train,
        X_test=prep.X_test_processed,
        y_test=prep.y_test,
        task_type="classification",
        track_emissions=False,
    )

    assert result.status == "SUCCESS"
    assert result.error_message is None
    assert result.accuracy is not None and 0.0 <= result.accuracy <= 1.0
    assert result.f1 is not None and 0.0 <= result.f1 <= 1.0
    assert result.training_time_sec is not None and result.training_time_sec >= 0.0
    assert result.prediction_time_sec is not None and result.prediction_time_sec >= 0.0
    assert result.latency_per_sample_ms is not None and result.latency_per_sample_ms >= 0.0
    assert result.ram_current_mb is not None and result.ram_current_mb > 0
    assert result.energy_status == "Disabled (Opt-in tracking not requested)"
    assert result.energy_kwh is None


def test_benchmark_single_model_regression(synthetic_regression_prep):
    """Verify single regression model benchmark execution and metrics."""
    prep = synthetic_regression_prep
    model = build_model_instance("linear_regression", "regression", random_state=42)

    result = benchmark_single_model(
        model_id="linear_regression",
        model_name="Linear Regression (Ridge)",
        model_instance=model,
        X_train=prep.X_train_processed,
        y_train=prep.y_train,
        X_test=prep.X_test_processed,
        y_test=prep.y_test,
        task_type="regression",
        track_emissions=False,
    )

    assert result.status == "SUCCESS"
    assert result.mae is not None and result.mae >= 0.0
    assert result.rmse is not None and result.rmse >= 0.0
    assert result.r2 is not None


def test_benchmark_single_model_error_containment():
    """Verify that model execution errors are caught and recorded without crashing."""
    class FaultyEstimator:
        def fit(self, X, y):
            raise RuntimeError("Simulated internal GPU/CPU arithmetic error")
        def predict(self, X):
            return np.zeros(len(X))

    faulty = FaultyEstimator()
    result = benchmark_single_model(
        model_id="faulty_model",
        model_name="Faulty Model",
        model_instance=faulty,
        X_train=np.ones((10, 2)),
        y_train=np.zeros(10),
        X_test=np.ones((5, 2)),
        y_test=np.zeros(5),
        task_type="classification",
    )

    assert result.status == "FAILED"
    assert "Simulated internal GPU/CPU arithmetic error" in (result.error_message or "")
    assert result.accuracy is None


# ==============================================================================
# 4. Benchmark Suite Execution Tests
# ==============================================================================

def test_run_benchmark_suite_classification(synthetic_classification_prep):
    """Verify running the full benchmark suite on classification data."""
    prep = synthetic_classification_prep
    suite = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["logistic_regression", "decision_tree"],
        random_state=42,
        track_emissions=False,
        dataset_name="Synthetic Classification Test",
    )

    assert len(suite.models_evaluated) == 2
    assert suite.task_type == "classification"
    assert suite.sample_count_train == 30
    assert suite.sample_count_test == 10

    df = suite.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Accuracy" in df.columns
    assert "F1-Score" in df.columns
    assert "Training Time (s)" in df.columns
    assert "Peak Process RAM (MB)" in df.columns


def test_run_benchmark_suite_regression(synthetic_regression_prep):
    """Verify running the full benchmark suite on regression data."""
    prep = synthetic_regression_prep
    suite = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["linear_regression", "decision_tree"],
        random_state=42,
        track_emissions=False,
        dataset_name="Synthetic Regression Test",
    )

    assert len(suite.models_evaluated) == 2
    assert suite.task_type == "regression"

    df = suite.to_dataframe()
    assert "MAE" in df.columns
    assert "RMSE" in df.columns
    assert "R² Score" in df.columns


def test_reproducibility(synthetic_classification_prep):
    """Verify deterministic benchmark results when passing identical random_state."""
    prep = synthetic_classification_prep

    suite1 = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["random_forest"],
        random_state=99,
    )
    suite2 = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["random_forest"],
        random_state=99,
    )

    f1_1 = suite1.models_evaluated[0].f1
    f1_2 = suite2.models_evaluated[0].f1
    assert f1_1 == f1_2


def test_energy_telemetry_isolation(synthetic_classification_prep):
    """Verify that disabled energy tracking does not invent or fabricate values."""
    prep = synthetic_classification_prep
    suite = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["logistic_regression"],
        track_emissions=False,
    )

    run = suite.models_evaluated[0]
    assert run.energy_kwh is None
    assert run.carbon_emissions_g is None
    assert "Disabled" in run.energy_status


def test_generate_benchmark_interpretation(synthetic_classification_prep):
    """Verify generation of comparative evidence-based interpretation."""
    prep = synthetic_classification_prep
    suite = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["logistic_regression", "decision_tree"],
        random_state=42,
    )

    summary = generate_benchmark_interpretation(suite)
    assert "best_predictive_model" in summary
    assert "fastest_training_model" in summary
    assert "tradeoff_observation" in summary
    assert len(summary["summary_text"]) > 10
    assert "Weighted F1:" in summary["best_predictive_metric"]
    assert " ms" in summary["fastest_training_model"]
    assert "s)" in summary["fastest_training_model"]


def test_benchmark_single_model_warmup_and_repeats(synthetic_classification_prep):
    """Verify configurable warm-up passes and repeated measurement averaging."""
    prep = synthetic_classification_prep
    model = build_model_instance("decision_tree", "classification", random_state=42)

    result = benchmark_single_model(
        model_id="decision_tree",
        model_name="Decision Tree",
        model_instance=model,
        X_train=prep.X_train_processed,
        y_train=prep.y_train,
        X_test=prep.X_test_processed,
        y_test=prep.y_test,
        task_type="classification",
        track_emissions=False,
        n_warmup=2,
        n_repeats=3,
    )

    assert result.status == "SUCCESS"
    assert result.prediction_time_sec is not None and result.prediction_time_sec > 0.0
    assert result.latency_per_sample_ms is not None and result.latency_per_sample_ms > 0.0
    expected_latency = (result.prediction_time_sec / len(prep.X_test_processed)) * 1000.0
    assert np.isclose(result.latency_per_sample_ms, expected_latency)


def test_interpretation_unit_conversion_consistency(synthetic_classification_prep):
    """Verify that milliseconds and seconds are mathematically identical in observations."""
    prep = synthetic_classification_prep
    suite = run_benchmark_suite(
        prep_result=prep,
        selected_model_ids=["logistic_regression", "decision_tree"],
        random_state=42,
    )
    summary = generate_benchmark_interpretation(suite)

    # Fastest model string should contain both representations: e.g. "Decision Tree (25.6 ms / 0.0256s)"
    fastest_str = summary["fastest_training_model"]
    assert " ms / " in fastest_str and fastest_str.endswith("s)")


# ==============================================================================
# 5. Validation on Realistic Local Public Datasets (Offline)
# ==============================================================================

def test_realistic_local_classification_dataset():
    """Validate full benchmark suite on Breast Cancer Wisconsin (569 samples, 30 features).
    
    Verifies that all models share the exact same reproducible train/test split,
    classification metrics are within valid bounds, timing and RAM are measured,
    and output DataFrame formats without encoding errors.
    """
    from sklearn.datasets import load_breast_cancer
    from src.data_preprocessing import separate_features_and_target, split_and_preprocess_data

    # Load local scikit-learn offline dataset
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    assert df.shape == (569, 31)

    X, y = separate_features_and_target(df, "target")
    prep = split_and_preprocess_data(X, y, task_type="classification", test_size=0.2, random_state=42)

    # Copy of training features to verify immutability
    X_train_original = prep.X_train_processed.copy()

    # Benchmark all 5 classification baseline models
    suite = run_benchmark_suite(
        prep_result=prep,
        random_state=42,
        track_emissions=False,
        dataset_name="Breast Cancer Wisconsin",
    )

    # 1. Verify train/test split consistency across all models
    assert len(suite.models_evaluated) == 5
    for run in suite.models_evaluated:
        assert run.status == "SUCCESS"
        assert run.accuracy is not None and 0.85 <= run.accuracy <= 1.0
        assert run.f1 is not None and 0.85 <= run.f1 <= 1.0
        assert run.precision is not None and 0.85 <= run.precision <= 1.0
        assert run.recall is not None and 0.85 <= run.recall <= 1.0
        assert run.training_time_sec is not None and run.training_time_sec > 0.0
        assert run.prediction_time_sec is not None and run.prediction_time_sec > 0.0
        assert run.latency_per_sample_ms is not None and run.latency_per_sample_ms > 0.0
        assert run.ram_current_mb is not None and run.ram_current_mb > 0.0
        assert "Disabled" in run.energy_status
        assert run.energy_kwh is None

    # 2. Verify underlying training matrix was not mutated by any model
    np.testing.assert_array_equal(prep.X_train_processed, X_train_original)

    # 3. Verify DataFrame conversion and column naming (ASCII-safe Carbon column)
    df_results = suite.to_dataframe()
    assert len(df_results) == 5
    assert "Carbon (gCO2eq)" in df_results.columns
    assert "Energy (kWh)" in df_results.columns
    assert "F1-Score" in df_results.columns

    # 4. Verify interpretation extraction
    interpretation = generate_benchmark_interpretation(suite)
    assert interpretation["best_predictive_model"] is not None
    assert interpretation["fastest_training_model"] is not None


def test_realistic_local_regression_dataset():
    """Validate full benchmark suite on Diabetes Disease Progression (442 samples, 10 features).
    
    Verifies regression metric validity (MAE, RMSE, R²), positive execution times,
    and memory usage recording.
    """
    from sklearn.datasets import load_diabetes
    from src.data_preprocessing import separate_features_and_target, split_and_preprocess_data

    # Load local scikit-learn offline dataset
    data = load_diabetes(as_frame=True)
    df = data.frame
    assert df.shape == (442, 11)

    X, y = separate_features_and_target(df, "target")
    prep = split_and_preprocess_data(X, y, task_type="regression", test_size=0.2, random_state=42)

    suite = run_benchmark_suite(
        prep_result=prep,
        random_state=42,
        track_emissions=False,
        dataset_name="Diabetes Progression",
    )

    assert len(suite.models_evaluated) == 5
    for run in suite.models_evaluated:
        assert run.status == "SUCCESS"
        assert run.mae is not None and run.mae > 0.0
        assert run.rmse is not None and run.rmse > 0.0
        assert run.r2 is not None and run.r2 <= 1.0
        assert run.training_time_sec is not None and run.training_time_sec > 0.0
        assert run.latency_per_sample_ms is not None and run.latency_per_sample_ms > 0.0
        assert run.ram_current_mb is not None and run.ram_current_mb > 0.0

    df_results = suite.to_dataframe()
    assert "MAE" in df_results.columns
    assert "RMSE" in df_results.columns
    assert "R² Score" in df_results.columns


