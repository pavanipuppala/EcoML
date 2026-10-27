"""
EcoML Source Package.

This package contains core modules for dataset processing, model training,
deep resource and emission tracking, and multi-objective model selection.
"""

from src.data_preprocessing import (
    PreprocessingResult,
    load_dataset,
    validate_dataset,
    get_dataset_summary,
    identify_column_types,
    validate_target_column,
    separate_features_and_target,
    build_preprocessor,
    split_and_preprocess_data,
)

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

from src.telemetry import (
    TelemetryCategory,
    ProcessTelemetrySnapshot,
    ProcessResourceMonitor,
    CarbonTelemetryResult,
    CarbonTrackerWrapper,
)

from src.recommender import (
    WeightProfile,
    ModelScoreDetail,
    RecommendationReport,
    calculate_model_recommendations,
    compute_pareto_frontier,
)

__all__ = [
    # Step 2: Preprocessing
    "PreprocessingResult",
    "load_dataset",
    "validate_dataset",
    "get_dataset_summary",
    "identify_column_types",
    "validate_target_column",
    "separate_features_and_target",
    "build_preprocessor",
    "split_and_preprocess_data",
    # Step 3: Benchmarking
    "SingleModelBenchmark",
    "BenchmarkSuiteResult",
    "get_supported_models",
    "build_model_instance",
    "evaluate_predictions",
    "benchmark_single_model",
    "run_benchmark_suite",
    "generate_benchmark_interpretation",
    # Step 4: Telemetry & Recommendation
    "TelemetryCategory",
    "ProcessTelemetrySnapshot",
    "ProcessResourceMonitor",
    "CarbonTelemetryResult",
    "CarbonTrackerWrapper",
    "WeightProfile",
    "ModelScoreDetail",
    "RecommendationReport",
    "calculate_model_recommendations",
    "compute_pareto_frontier",
]

__version__ = "0.4.0"
