"""
Basic foundation tests for EcoML.

Verifies that project directories and configuration constants are defined
and point to the expected workspace structure.
"""

from pathlib import Path
from config.settings import (
    PROJECT_NAME,
    PROJECT_STATUS,
    BASE_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SRC_DIR,
    CONFIG_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TESTS_DIR,
    PLANNED_FEATURES,
    FUTURE_ROADMAP_FEATURES,
    ROADMAP_STEPS,
)


def test_project_metadata():
    """Verify project name and current operational status."""
    assert PROJECT_NAME == "EcoML"
    assert "Step 4" in PROJECT_STATUS or "Operational" in PROJECT_STATUS


def test_directories_exist():
    """Verify that all core directories exist on disk."""
    required_paths = [
        BASE_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SRC_DIR,
        CONFIG_DIR,
        EXPERIMENTS_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        TESTS_DIR,
    ]
    for directory in required_paths:
        assert isinstance(directory, Path)
        assert directory.exists(), f"Directory missing: {directory}"
        assert directory.is_dir(), f"Path is not a directory: {directory}"


def test_planned_features_count():
    """Verify that all planned evaluation features are registered in settings."""
    feature_ids = [f["id"] for f in PLANNED_FEATURES]
    expected_ids = [
        "model_accuracy",
        "execution_time",
        "memory_usage",
        "energy_estimation",
        "carbon_emissions",
        "model_recommendation",
    ]
    assert feature_ids == expected_ids


def test_feature_implementation_status():
    """Verify that implemented features are labelled as Implemented and future features as Planned."""
    for feat in PLANNED_FEATURES:
        assert "Implemented" in feat["status"], f"Feature {feat['id']} should be marked as Implemented"

    for feat in FUTURE_ROADMAP_FEATURES:
        assert feat["status"] == "Planned", f"Feature {feat['id']} should be marked as Planned"


def test_roadmap_steps_status():
    """Verify that roadmap records Steps 1-4 as Completed and Step 5 as Planned."""
    completed = [s for s in ROADMAP_STEPS if s["status"] == "Completed"]
    planned = [s for s in ROADMAP_STEPS if s["status"] == "Planned"]

    assert len(completed) == 4
    assert len(planned) >= 1
    assert completed[0]["step"] == "Step 1"
    assert completed[3]["step"] == "Step 4"

