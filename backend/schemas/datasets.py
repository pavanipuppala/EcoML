"""
EcoML — Dataset and Preprocessing Schemas.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class DatasetListItem(BaseModel):
    """Metadata item for a selectable dataset."""
    dataset_id: str = Field(..., description="Unique dataset identifier.")
    name: str = Field(..., description="Human-readable dataset display name.")
    source_type: str = Field(..., description="Source type ('built_in' or 'uploaded').")
    description: str = Field(..., description="Overview of dataset domain and characteristics.")
    sample_count: Optional[int] = Field(None, description="Number of observations.")
    feature_count: Optional[int] = Field(None, description="Number of columns.")
    default_task: Optional[str] = Field(None, description="Recommended ML task type ('classification' or 'regression').")
    default_target: Optional[str] = Field(None, description="Recommended default target column.")


class DatasetListResponse(BaseModel):
    """Available datasets response."""
    datasets: List[DatasetListItem]


class ColumnSummary(BaseModel):
    """Statistical summary of dataset columns and memory footprint."""
    shape: List[int]
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_missing: int
    duplicate_rows: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    memory_usage_kb: float


class DatasetPreviewResponse(BaseModel):
    """Dataset preview and diagnostics response."""
    dataset_id: str
    name: str
    rows: int
    columns: int
    column_names: List[str]
    summary: ColumnSummary
    preview_rows: List[Dict[str, Any]]


class TargetValidationRequest(BaseModel):
    """Target variable validation request."""
    target_column: str = Field(..., min_length=1, max_length=200, description="Name of column to test as prediction target.")
    task_type: str = Field("classification", description="Machine learning task type ('classification' or 'regression').")

    @field_validator("task_type")
    @classmethod
    def validate_task(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("classification", "regression"):
            raise ValueError("task_type must be either 'classification' or 'regression'.")
        return clean


class TargetValidationResponse(BaseModel):
    """Target column validation response."""
    target_column: str
    task_type: str
    is_valid: bool
    error_message: Optional[str] = None
    unique_classes: int
    null_count: int


class PreprocessingRequest(BaseModel):
    """Data preprocessing and leakage-free partition request."""
    target_column: str = Field(..., min_length=1, max_length=200, description="Target column name.")
    task_type: str = Field("classification", description="Task type ('classification' or 'regression').")
    test_size: float = Field(0.2, gt=0.0, lt=1.0, description="Test partition ratio (0.0 to 1.0).")
    random_state: int = Field(42, ge=0, le=2147483647, description="Reproducibility seed.")
    scale_numerical: bool = Field(True, description="Apply standard scaling to numerical features.")

    @field_validator("task_type")
    @classmethod
    def validate_task(cls, v: str) -> str:
        clean = v.strip().lower()
        if clean not in ("classification", "regression"):
            raise ValueError("task_type must be either 'classification' or 'regression'.")
        return clean


class PreprocessingResponse(BaseModel):
    """Preprocessing and data partitioning output metadata."""
    dataset_id: str
    config_signature: str
    task_type: str
    target_column: str
    train_samples: int
    test_samples: int
    feature_count: int
    numerical_cols: List[str]
    categorical_cols: List[str]
    feature_names_out: List[str]
    message: str = "Preprocessing pipeline fitted strictly on training partition without data leakage."

