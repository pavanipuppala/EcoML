"""
EcoML — Cross-Tab Data Persistence & State Flow Regression Tests.

Validates that dataset ingestion, target variable selection, preprocessing results,
and configuration signatures persist across Streamlit reruns, and that parameter
or dataset changes reliably invalidate stale preprocessed matrices and benchmarks.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from app import build_config_signature

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_build_config_signature_consistency():
    """Verify that configuration fingerprints are deterministic and sensitive to all parameters."""
    sig1 = build_config_signature("demo", "target", "classification", 0.2, 42, True)
    sig2 = build_config_signature("demo", "target", "classification", 0.20, 42, True)
    assert sig1 == sig2

    # Change target
    sig_diff_target = build_config_signature("demo", "label", "classification", 0.2, 42, True)
    assert sig1 != sig_diff_target

    # Change task type
    sig_diff_task = build_config_signature("demo", "target", "regression", 0.2, 42, True)
    assert sig1 != sig_diff_task

    # Change test size
    sig_diff_split = build_config_signature("demo", "target", "classification", 0.3, 42, True)
    assert sig1 != sig_diff_split

    # Change seed
    sig_diff_seed = build_config_signature("demo", "target", "classification", 0.2, 99, True)
    assert sig1 != sig_diff_seed

    # Change scaling
    sig_diff_scale = build_config_signature("demo", "target", "classification", 0.2, 42, False)
    assert sig1 != sig_diff_scale


def test_app_demo_dataset_persistence_across_reruns():
    """Verify that demo dataset and preprocessing results persist across simulated reruns."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # 1. Click Demo CSV button
    for b in at.button:
        if "Demo CSV" in b.label:
            b.click().run()
            break

    assert at.session_state["dataset_source_type"] == "synthetic_demo"
    assert at.session_state["raw_df"] is not None
    assert at.session_state["preprocessed_data"] is None

    # 2. Click Configure & Run Preprocessing Pipeline
    for b in at.button:
        if "Configure & Run Preprocessing" in b.label:
            b.click().run()
            break

    assert at.session_state["preprocessed_data"] is not None
    active_sig = at.session_state["prep_signature"]
    assert active_sig is not None

    # 3. Simulate subsequent reruns (e.g. switching tabs or interacting with controls)
    at.run()
    assert at.session_state["preprocessed_data"] is not None
    assert at.session_state["prep_signature"] == active_sig

    at.run()
    assert at.session_state["preprocessed_data"] is not None


def test_app_parameter_change_invalidates_preprocessed_data():
    """Verify that changing a partition parameter invalidates previous preprocessed data."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Load Demo CSV
    for b in at.button:
        if "Demo CSV" in b.label:
            b.click().run()
            break

    # Run Preprocessing
    for b in at.button:
        if "Configure & Run Preprocessing" in b.label:
            b.click().run()
            break

    assert at.session_state["preprocessed_data"] is not None

    # Modify slider value (Test Partition Ratio)
    for s in at.slider:
        if "Test Partition Ratio" in s.label:
            s.set_value(0.35).run()
            break

    # Stale preprocessing data must be invalidated immediately
    assert at.session_state["preprocessed_data"] is None
    assert at.session_state["prep_signature"] is None


def test_app_dataset_switch_invalidates_downstream_state():
    """Verify that switching datasets invalidates prior preprocessed data and signatures."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Load Demo CSV & Run Preprocessing
    for b in at.button:
        if "Demo CSV" in b.label:
            b.click().run()
            break

    for b in at.button:
        if "Configure & Run Preprocessing" in b.label:
            b.click().run()
            break

    assert at.session_state["dataset_source_type"] == "synthetic_demo"
    assert at.session_state["preprocessed_data"] is not None

    # Switch to Breast Cancer dataset
    for b in at.button:
        if "Breast Cancer" in b.label:
            b.click().run()
            break

    assert at.session_state["dataset_source_type"] == "breast_cancer"
    assert at.session_state["preprocessed_data"] is None
    assert at.session_state["prep_signature"] is None


def test_app_end_to_end_flow_and_step3_step4_access():
    """Verify that preprocessed data in Step 2 is accessible in Step 3 and Step 4 without being lost."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Step 2: Load Demo CSV
    for b in at.button:
        if "Demo CSV" in b.label:
            b.click().run()
            break

    # Step 2: Run Preprocessing
    for b in at.button:
        if "Configure & Run Preprocessing" in b.label:
            b.click().run()
            break

    assert at.session_state["preprocessed_data"] is not None

    # Step 3: Trigger Model Benchmark Suite
    for b in at.button:
        if "Run Empirical Benchmark Suite" in b.label:
            b.click().run()
            break

    assert at.session_state["benchmark_suite_result"] is not None
    suite_res = at.session_state["benchmark_suite_result"]
    assert len(suite_res.models_evaluated) > 0

    # Simulate subsequent tab visit or interaction
    at.run()
    assert at.session_state["preprocessed_data"] is not None
    assert at.session_state["benchmark_suite_result"] is not None

    # Step 4: Verify benchmark results are recognized and not marked stale
    assert at.session_state["benchmark_prep_sig"] == at.session_state["prep_signature"]


def test_app_csv_upload_persistence_across_reruns():
    """Verify that user-uploaded CSV and preprocessing persist across reruns."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    csv_content = b"feature1,feature2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n13,14,0\n15,16,1\n17,18,0\n19,20,1\n"
    at.file_uploader[0].upload(filename="custom_dataset.csv", content=csv_content).run()

    assert at.session_state["dataset_source_type"] == "uploaded"
    assert at.session_state["raw_df"] is not None

    # Run Preprocessing
    for b in at.button:
        if "Configure & Run Preprocessing" in b.label:
            b.click().run()
            break

    assert at.session_state["preprocessed_data"] is not None

    # Subsequent rerun
    at.run()
    assert at.session_state["preprocessed_data"] is not None
    assert at.session_state["raw_df"] is not None

