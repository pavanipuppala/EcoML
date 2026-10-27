"""
EcoML — Data Loading, Validation, and Preprocessing Module.

This module provides modular, leakage-free data processing routines for tabular
datasets, including CSV ingestion, dataset validation, exploratory summarization,
missing value imputation, categorical encoding, feature scaling, and reproducible
train-test splitting.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import io

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


@dataclass
class PreprocessingResult:
    """Container for preprocessed datasets, pipelines, and transformation metadata.
    
    Attributes:
        X_train_raw: Unprocessed training feature DataFrame.
        X_test_raw: Unprocessed testing feature DataFrame.
        X_train_processed: Scaled and encoded training feature array/DataFrame.
        X_test_processed: Scaled and encoded testing feature array/DataFrame.
        y_train: Training target array.
        y_test: Testing target array.
        preprocessor: Fitted scikit-learn ColumnTransformer instance.
        feature_names_out: Names of the features after transformation.
        numerical_cols: List of identified numerical feature column names.
        categorical_cols: List of identified categorical feature column names.
        target_column: Name of the selected target column.
        task_type: Machine learning task type ('classification' or 'regression').
        train_shape: Tuple representing processed X_train dimensions.
        test_shape: Tuple representing processed X_test dimensions.
        label_encoder: Fitted LabelEncoder for classification targets, or None.
    """
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_processed: np.ndarray
    X_test_processed: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names_out: List[str]
    numerical_cols: List[str]
    categorical_cols: List[str]
    target_column: str
    task_type: str
    train_shape: Tuple[int, int]
    test_shape: Tuple[int, int]
    label_encoder: Optional[LabelEncoder] = None


def load_dataset(file_or_path: Union[str, Path, io.IOBase, Any]) -> pd.DataFrame:
    """Load a tabular dataset from a CSV file path, buffer, or file-like object.

    Args:
        file_or_path: Path string, Path object, or file-like buffer (e.g. Streamlit UploadedFile).

    Returns:
        pd.DataFrame: Loaded tabular dataset.

    Raises:
        ValueError: If file is empty, cannot be parsed, or contains no readable data.
    """
    if file_or_path is None:
        raise ValueError("No file or buffer was provided.")

    try:
        # If it's a file-like object with a seek pointer, ensure reading from start
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        
        df = pd.read_csv(file_or_path)
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty and contains no data.")
    except Exception as exc:
        raise ValueError(f"Could not parse the file as a valid CSV: {str(exc)}")

    if df.empty:
        raise ValueError("The uploaded CSV file contains no rows.")

    return df


def validate_dataset(
    df: pd.DataFrame,
    min_rows: int = 10,
    min_cols: int = 2,
) -> Dict[str, Any]:
    """Validate that the dataset meets minimal structural requirements for machine learning.

    Args:
        df: Input pandas DataFrame to validate.
        min_rows: Minimum required number of samples (rows). Default is 10.
        min_cols: Minimum required number of columns (features + target). Default is 2.

    Returns:
        Dict[str, Any]: Validation summary containing:
            - 'is_valid' (bool): True if all checks pass.
            - 'error_message' (Optional[str]): Human-readable message if invalid.
            - 'rows' (int): Total rows.
            - 'cols' (int): Total columns.
    """
    if df is None:
        return {
            "is_valid": False,
            "error_message": "Dataset is None.",
            "rows": 0,
            "cols": 0,
        }

    n_rows, n_cols = df.shape

    if n_rows == 0:
        return {
            "is_valid": False,
            "error_message": "Dataset contains 0 rows. Please provide a non-empty CSV.",
            "rows": 0,
            "cols": n_cols,
        }

    if n_cols == 0:
        return {
            "is_valid": False,
            "error_message": "Dataset contains 0 columns. Please check your CSV format.",
            "rows": n_rows,
            "cols": 0,
        }

    if n_rows < min_rows:
        return {
            "is_valid": False,
            "error_message": (
                f"Dataset has too few rows ({n_rows} rows). "
                f"A minimum of {min_rows} rows is required for train/test splitting."
            ),
            "rows": n_rows,
            "cols": n_cols,
        }

    if n_cols < min_cols:
        return {
            "is_valid": False,
            "error_message": (
                f"Dataset has only {n_cols} column. "
                f"At least {min_cols} columns (at least 1 feature and 1 target) are required."
            ),
            "rows": n_rows,
            "cols": n_cols,
        }

    return {
        "is_valid": True,
        "error_message": None,
        "rows": n_rows,
        "cols": n_cols,
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive, beginner-friendly diagnostic summary of a DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing dimensions, column types,
            missing value metrics, duplicate counts, and memory usage.
    """
    if df.empty:
        return {
            "shape": (0, 0),
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "dtypes": {},
            "missing_counts": {},
            "missing_percentages": {},
            "total_missing": 0,
            "duplicate_rows": 0,
            "numerical_columns": [],
            "categorical_columns": [],
            "memory_usage_kb": 0.0,
        }

    rows, cols = df.shape
    col_names = list(df.columns)
    
    # Missing values breakdown
    missing_series = df.isnull().sum()
    missing_counts = missing_series.to_dict()
    missing_percentages = {
        col: round((count / rows) * 100, 2)
        for col, count in missing_counts.items()
    }
    total_missing = int(missing_series.sum())

    # Duplicate row counts
    duplicate_rows = int(df.duplicated().sum())

    # Column type categorization
    num_cols, cat_cols = identify_column_types(df)

    # Memory footprint
    memory_usage_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)

    return {
        "shape": (rows, cols),
        "rows": rows,
        "columns": cols,
        "column_names": col_names,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_percentages": missing_percentages,
        "total_missing": total_missing,
        "duplicate_rows": duplicate_rows,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "memory_usage_kb": memory_usage_kb,
    }


def identify_column_types(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Classify DataFrame columns into numerical and categorical feature sets.

    Args:
        df: Input pandas DataFrame.
        target_column: Optional target column to exclude from feature grouping.

    Returns:
        Tuple[List[str], List[str]]: (numerical_columns, categorical_columns).
    """
    features_df = df.drop(columns=[target_column]) if target_column and target_column in df.columns else df

    numerical_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in features_df.columns:
        if pd.api.types.is_numeric_dtype(features_df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    return numerical_cols, categorical_cols


def validate_target_column(
    df: pd.DataFrame,
    target_column: Optional[str],
    task_type: str = "classification",
) -> Dict[str, Any]:
    """Verify that the user-selected target column is valid and compatible with the task type.

    Args:
        df: Input pandas DataFrame.
        target_column: Name of the column selected as prediction target.
        task_type: Machine learning task ('classification' or 'regression').

    Returns:
        Dict[str, Any]: Validation report with keys:
            - 'is_valid' (bool)
            - 'error_message' (Optional[str])
            - 'unique_classes' (int)
            - 'null_count' (int)
    """
    if not target_column:
        return {
            "is_valid": False,
            "error_message": "No target column selected. Please select a target column to proceed.",
            "unique_classes": 0,
            "null_count": 0,
        }

    if target_column not in df.columns:
        return {
            "is_valid": False,
            "error_message": f"Target column '{target_column}' does not exist in the dataset.",
            "unique_classes": 0,
            "null_count": 0,
        }

    target_series = df[target_column]
    null_count = int(target_series.isnull().sum())
    valid_values = target_series.dropna()

    if len(valid_values) == 0:
        return {
            "is_valid": False,
            "error_message": f"Target column '{target_column}' contains only missing (NaN) values.",
            "unique_classes": 0,
            "null_count": null_count,
        }

    unique_count = int(valid_values.nunique())

    # Task-specific validations
    clean_task = task_type.strip().lower()

    if clean_task == "classification":
        if unique_count < 2:
            return {
                "is_valid": False,
                "error_message": (
                    f"Target column '{target_column}' has only 1 unique class ({valid_values.iloc[0]}). "
                    "Classification requires at least 2 distinct classes."
                ),
                "unique_classes": unique_count,
                "null_count": null_count,
            }

        # Check for continuous floating-point data or excessive cardinality
        is_float = pd.api.types.is_float_dtype(target_series)
        has_fractional = is_float and not np.all(np.isclose(valid_values, np.round(valid_values), atol=1e-7))
        is_high_cardinality = unique_count > 50 or (len(valid_values) > 30 and (unique_count / len(valid_values)) > 0.25)

        if has_fractional or is_high_cardinality:
            return {
                "is_valid": False,
                "error_message": (
                    f"Target column '{target_column}' contains continuous numerical data "
                    f"({unique_count} unique values). Classification requires discrete class labels "
                    f"(e.g., binary 0/1 or categorical groups). "
                    f"Please select 'Regression' as the task formulation or select a discrete target column."
                ),
                "unique_classes": unique_count,
                "null_count": null_count,
            }
    elif clean_task == "regression":
        if not pd.api.types.is_numeric_dtype(target_series):
            return {
                "is_valid": False,
                "error_message": (
                    f"Target column '{target_column}' has non-numeric data type ('{target_series.dtype}'). "
                    "Regression tasks require continuous numerical target values."
                ),
                "unique_classes": unique_count,
                "null_count": null_count,
            }
        if unique_count < 2:
            return {
                "is_valid": False,
                "error_message": (
                    f"Target column '{target_column}' is constant (all values are {valid_values.iloc[0]}). "
                    "Regression requires variance in the target variable."
                ),
                "unique_classes": unique_count,
                "null_count": null_count,
            }
    else:
        return {
            "is_valid": False,
            "error_message": f"Unsupported task type '{task_type}'. Must be 'classification' or 'regression'.",
            "unique_classes": unique_count,
            "null_count": null_count,
        }

    return {
        "is_valid": True,
        "error_message": None,
        "unique_classes": unique_count,
        "null_count": null_count,
    }


def separate_features_and_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the input feature matrix X and target vector y, removing missing targets.

    Args:
        df: Input pandas DataFrame.
        target_column: Column name designated as the prediction target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) with any rows having null targets safely removed.

    Raises:
        ValueError: If target_column is not found or if dropping nulls leaves 0 rows.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in dataset columns.")

    # Remove samples where the target itself is missing to ensure valid supervision
    valid_mask = df[target_column].notnull()
    cleaned_df = df[valid_mask]

    if cleaned_df.empty:
        raise ValueError(f"Target column '{target_column}' contains only missing values.")

    X = cleaned_df.drop(columns=[target_column]).copy()
    y = cleaned_df[target_column].copy()

    return X, y


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    scale_numerical: bool = True,
) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for numerical and categorical features.

    Numerical pipeline:
      - Median imputation (robust against outliers)
      - Standard scaling (zero mean, unit variance) if scale_numerical is True

    Categorical pipeline:
      - Most frequent value imputation
      - OneHotEncoding (handles unseen categories gracefully in test data)

    Args:
        numerical_cols: List of numerical column names.
        categorical_cols: List of categorical column names.
        scale_numerical: Whether to apply StandardScaler to numerical features. Default True.

    Returns:
        ColumnTransformer: Configured preprocessor pipeline.
    """
    transformers = []

    # 1. Numerical Pipeline
    if numerical_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numerical:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numerical", num_pipeline, numerical_cols))

    # 2. Categorical Pipeline
    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def split_and_preprocess_data(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
    scale_numerical: bool = True,
) -> PreprocessingResult:
    """Split dataset into train and test sets, then fit preprocessors strictly on training data.

    To prevent data leakage, imputation statistics, scalers, and one-hot encoders
    are fitted exclusively on X_train. The fitted pipeline is then used to transform X_test.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        task_type: 'classification' or 'regression'.
        test_size: Proportion of dataset allocated to test set (e.g. 0.2).
        random_state: Reproducibility seed.
        scale_numerical: Whether to apply StandardScaler to numeric features.

    Returns:
        PreprocessingResult: Dataclass holding train/test partitions and fitted transformers.

    Raises:
        ValueError: If input dimensions are invalid or parameters are out of range.
    """
    if len(X) != len(y):
        raise ValueError(f"Feature count ({len(X)}) does not match target count ({len(y)}).")

    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")

    num_cols, cat_cols = identify_column_types(X)

    if not num_cols and not cat_cols:
        raise ValueError("No feature columns available to preprocess.")

    # Determine stratification for classification
    clean_task = task_type.strip().lower()
    stratify_target = None
    label_encoder = None

    if clean_task == "classification":
        # Check if every class has at least 2 instances to enable stratification
        class_counts = y.value_counts()
        if (class_counts >= 2).all() and len(class_counts) > 1:
            stratify_target = y
        
        # Fit LabelEncoder on y for standard 0..K-1 integer classes
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y.to_numpy(dtype=float)

    # 1. Train / Test Split (Strictly before any transformation to avoid leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    # 2. Build Preprocessor Pipeline
    preprocessor = build_preprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        scale_numerical=scale_numerical,
    )

    # 3. Fit ONLY on X_train, then transform both train and test sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Ensure processed arrays are float32/float64 ndarrays
    if hasattr(X_train_processed, "toarray"):
        X_train_processed = X_train_processed.toarray()
    if hasattr(X_test_processed, "toarray"):
        X_test_processed = X_test_processed.toarray()

    X_train_processed = np.asarray(X_train_processed, dtype=np.float64)
    X_test_processed = np.asarray(X_test_processed, dtype=np.float64)

    # Extract output feature names safely
    try:
        feature_names_out = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names_out = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    return PreprocessingResult(
        X_train_raw=X_train,
        X_test_raw=X_test,
        X_train_processed=X_train_processed,
        X_test_processed=X_test_processed,
        y_train=np.asarray(y_train),
        y_test=np.asarray(y_test),
        preprocessor=preprocessor,
        feature_names_out=feature_names_out,
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        target_column=getattr(y, "name", "target") or "target",
        task_type=clean_task,
        train_shape=X_train_processed.shape,
        test_shape=X_test_processed.shape,
        label_encoder=label_encoder,
    )

