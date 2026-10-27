"""
Unit tests for EcoML dataset loading, validation, and preprocessing pipeline.

These tests use synthetic in-memory fixtures to verify the preprocessing routines
without fabricating benchmark results or relying on external datasets.
"""

import io
import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    load_dataset,
    validate_dataset,
    get_dataset_summary,
    identify_column_types,
    validate_target_column,
    separate_features_and_target,
    build_preprocessor,
    split_and_preprocess_data,
)


@pytest.fixture
def sample_classification_csv():
    """Create an in-memory CSV string representing a small mixed-type classification dataset."""
    csv_data = """age,income,department,tenure,promoted
25,50000,Sales,1,0
30,62000,Engineering,3,1
35,75000,Engineering,5,1
40,,Marketing,8,0
22,45000,Sales,1,0
45,90000,,12,1
32,68000,Engineering,4,1
28,52000,Marketing,2,0
38,82000,Sales,7,1
50,98000,Engineering,15,1
29,58000,Sales,2,0
42,85000,Marketing,10,1
"""
    return io.StringIO(csv_data)


@pytest.fixture
def sample_regression_csv():
    """Create an in-memory CSV string representing a small regression dataset."""
    csv_data = """area,bedrooms,location,age,price
1200,2,Urban,5,300000
1500,3,Suburban,10,380000
1800,3,Urban,2,450000
900,1,Rural,20,180000
2200,4,Suburban,7,550000
1100,2,Rural,15,220000
1350,3,Urban,8,340000
1650,3,Suburban,12,410000
2000,4,Urban,4,510000
2500,5,Suburban,1,620000
1050,2,Rural,18,205000
1750,3,Urban,6,440000
"""
    return io.StringIO(csv_data)


# ==============================================================================
# 1. Dataset Loading Tests
# ==============================================================================

def test_load_dataset_valid_csv(sample_classification_csv):
    """Test reading a valid CSV buffer."""
    df = load_dataset(sample_classification_csv)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 12
    assert "promoted" in df.columns


def test_load_dataset_empty_file():
    """Test that an empty buffer raises a helpful ValueError."""
    empty_buffer = io.StringIO("")
    with pytest.raises(ValueError, match="empty"):
        load_dataset(empty_buffer)


def test_load_dataset_none():
    """Test that passing None raises ValueError."""
    with pytest.raises(ValueError, match="No file"):
        load_dataset(None)


# ==============================================================================
# 2. Dataset Validation Tests
# ==============================================================================

def test_validate_dataset_valid(sample_classification_csv):
    """Test validation on an adequately sized dataset."""
    df = load_dataset(sample_classification_csv)
    result = validate_dataset(df, min_rows=10, min_cols=2)
    assert result["is_valid"] is True
    assert result["error_message"] is None
    assert result["rows"] == 12
    assert result["cols"] == 5


def test_validate_dataset_too_few_rows():
    """Test validation failure when row count is below threshold."""
    tiny_df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})
    result = validate_dataset(tiny_df, min_rows=10, min_cols=2)
    assert result["is_valid"] is False
    assert "too few rows" in result["error_message"]


def test_validate_dataset_too_few_cols():
    """Test validation failure when column count is below threshold."""
    one_col_df = pd.DataFrame({"only_col": list(range(15))})
    result = validate_dataset(one_col_df, min_rows=10, min_cols=2)
    assert result["is_valid"] is False
    assert "only 1 column" in result["error_message"]


# ==============================================================================
# 3. Exploratory Summary and Column Typing Tests
# ==============================================================================

def test_get_dataset_summary(sample_classification_csv):
    """Verify that summary contains all required diagnostic fields."""
    df = load_dataset(sample_classification_csv)
    summary = get_dataset_summary(df)

    assert summary["rows"] == 12
    assert summary["columns"] == 5
    assert "promoted" in summary["column_names"]
    assert summary["missing_counts"]["income"] == 1
    assert summary["missing_counts"]["department"] == 1
    assert summary["total_missing"] == 2
    assert summary["duplicate_rows"] == 0
    assert "income" in summary["numerical_columns"]
    assert "department" in summary["categorical_columns"]
    assert summary["memory_usage_kb"] > 0


def test_identify_column_types(sample_classification_csv):
    """Test numerical vs categorical identification and target exclusion."""
    df = load_dataset(sample_classification_csv)
    num_cols, cat_cols = identify_column_types(df, target_column="promoted")

    assert set(num_cols) == {"age", "income", "tenure"}
    assert set(cat_cols) == {"department"}
    assert "promoted" not in num_cols
    assert "promoted" not in cat_cols


# ==============================================================================
# 4. Target Column Validation Tests
# ==============================================================================

def test_validate_target_column_classification_success(sample_classification_csv):
    """Test valid binary classification target."""
    df = load_dataset(sample_classification_csv)
    res = validate_target_column(df, target_column="promoted", task_type="classification")
    assert res["is_valid"] is True
    assert res["unique_classes"] == 2
    assert res["error_message"] is None


def test_validate_target_column_single_class():
    """Test classification rejection when target has only 1 distinct class."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    })
    res = validate_target_column(df, target_column="target", task_type="classification")
    assert res["is_valid"] is False
    assert "only 1 unique class" in res["error_message"]


def test_validate_target_column_regression_success(sample_regression_csv):
    """Test valid regression continuous target."""
    df = load_dataset(sample_regression_csv)
    res = validate_target_column(df, target_column="price", task_type="regression")
    assert res["is_valid"] is True
    assert res["unique_classes"] > 2


def test_validate_target_column_regression_non_numeric():
    """Test regression rejection when target contains text strings."""
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "category_target": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })
    res = validate_target_column(df, target_column="category_target", task_type="regression")
    assert res["is_valid"] is False
    assert "non-numeric" in res["error_message"]


def test_validate_target_column_missing_or_invalid():
    """Test rejection when target does not exist or is None."""
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    res1 = validate_target_column(df, target_column=None)
    assert res1["is_valid"] is False
    assert "No target column selected" in res1["error_message"]

    res2 = validate_target_column(df, target_column="non_existent")
    assert res2["is_valid"] is False
    assert "does not exist" in res2["error_message"]


def test_validate_target_column_classification_continuous_rejected():
    """Regression test: continuous floating-point variables must be rejected as classification targets."""
    from sklearn.datasets import load_breast_cancer
    df = load_breast_cancer(as_frame=True).frame
    res = validate_target_column(df, target_column="smoothness error", task_type="classification")
    assert res["is_valid"] is False
    assert "continuous numerical data" in res["error_message"]
    assert res["unique_classes"] > 50


def test_validate_target_column_smoothness_error_valid_for_regression():
    """Continuous biological measurements like 'smoothness error' are valid targets for regression."""
    from sklearn.datasets import load_breast_cancer
    df = load_breast_cancer(as_frame=True).frame
    res = validate_target_column(df, target_column="smoothness error", task_type="regression")
    assert res["is_valid"] is True
    assert res["error_message"] is None


def test_validate_target_column_classification_discrete_floats_accepted():
    """Discrete float classes (e.g. 0.0 and 1.0) must be accepted for classification."""
    df = pd.DataFrame({
        "feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "target": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    })
    res = validate_target_column(df, target_column="target", task_type="classification")
    assert res["is_valid"] is True
    assert res["unique_classes"] == 2



# ==============================================================================
# 5. Separation and Preprocessing Pipeline Tests (No Leakage)
# ==============================================================================

def test_separate_features_and_target(sample_classification_csv):
    """Test separation of X and y and dropping of samples where target is null."""
    df = load_dataset(sample_classification_csv)
    # Inject a null target in one row
    df.loc[0, "promoted"] = np.nan

    X, y = separate_features_and_target(df, target_column="promoted")
    assert "promoted" not in X.columns
    assert len(X) == 11
    assert len(y) == 11
    assert y.isnull().sum() == 0


def test_build_preprocessor_structure():
    """Test preprocessor construction with both numeric and categorical features."""
    preprocessor = build_preprocessor(
        numerical_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        scale_numerical=True,
    )
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "numerical" in transformer_names
    assert "categorical" in transformer_names


def test_split_and_preprocess_no_data_leakage(sample_classification_csv):
    """Verify that preprocessing is fitted strictly on X_train to prevent data leakage."""
    df = load_dataset(sample_classification_csv)
    X, y = separate_features_and_target(df, target_column="promoted")

    res = split_and_preprocess_data(
        X=X,
        y=y,
        task_type="classification",
        test_size=0.25,
        random_state=42,
        scale_numerical=True,
    )

    # Check shapes
    assert res.X_train_processed.shape[0] == len(res.y_train)
    assert res.X_test_processed.shape[0] == len(res.y_test)
    assert res.X_train_processed.shape[1] == res.X_test_processed.shape[1]

    # Check that imputation resolved missing values: no NaNs in processed arrays
    assert not np.isnan(res.X_train_processed).any()
    assert not np.isnan(res.X_test_processed).any()

    # Data leakage check: Verify StandardScaler mean was calculated solely on X_train
    num_pipeline = res.preprocessor.named_transformers_["numerical"]
    scaler = num_pipeline.named_steps["scaler"]
    imputer = num_pipeline.named_steps["imputer"]
    
    # Calculate expected imputed mean on X_train only
    num_train_imputed = imputer.transform(res.X_train_raw[res.numerical_cols])
    expected_means = np.mean(num_train_imputed, axis=0)
    np.testing.assert_allclose(scaler.mean_, expected_means)


def test_unseen_categorical_category_in_test_set():
    """Verify that unseen categories in X_test are ignored gracefully without crashing."""
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "blue", "red", "blue", "red", "blue", "red", "green"],
        "num": [10, 20, 10, 20, 10, 20, 10, 20, 10, 30],
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    X, y = separate_features_and_target(df, target_column="label")

    # Manually split to ensure 'green' only appears in test set
    X_train = X.iloc[:8]  # Only 'red' and 'blue'
    X_test = X.iloc[8:]   # Contains 'green'
    y_train = y.iloc[:8].to_numpy()
    y_test = y.iloc[8:].to_numpy()

    num_cols = ["num"]
    cat_cols = ["color"]
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numerical=True)

    X_train_proc = preprocessor.fit_transform(X_train)
    # Transforming X_test with unseen 'green' should encode as all zeros, not raise error
    X_test_proc = preprocessor.transform(X_test)

    assert X_train_proc.shape[1] == X_test_proc.shape[1]
    assert not np.isnan(X_test_proc).any()


def test_regression_pipeline_execution(sample_regression_csv):
    """Test full preprocessing pipeline on a regression dataset."""
    df = load_dataset(sample_regression_csv)
    X, y = separate_features_and_target(df, target_column="price")

    res = split_and_preprocess_data(
        X=X,
        y=y,
        task_type="regression",
        test_size=0.25,
        random_state=42,
        scale_numerical=True,
    )

    assert res.task_type == "regression"
    assert res.label_encoder is None  # Regression target is continuous float
    assert len(res.y_train) == res.X_train_processed.shape[0]
    assert len(res.y_test) == res.X_test_processed.shape[0]
    assert not np.isnan(res.X_train_processed).any()
    assert not np.isnan(res.X_test_processed).any()

