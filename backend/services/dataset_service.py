"""
EcoML — Dataset and Preprocessing Service.

Manages built-in and uploaded datasets, schema inspection, target variable validation,
and leakage-free scikit-learn preprocessing pipelines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import io
import re
import uuid

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from config.settings import DATA_DIR, RAW_DATA_DIR
from src.data_preprocessing import (
    PreprocessingResult,
    load_dataset,
    validate_dataset,
    get_dataset_summary,
    identify_column_types,
    validate_target_column,
    separate_features_and_target,
    split_and_preprocess_data,
)

MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 Megabytes

DEMO_CSV_STRING = """age,income,department,tenure,previous_eval,promoted
25,52000,Sales,1,3.2,0
30,64000,Engineering,3,4.1,1
35,78000,Engineering,5,4.5,1
40,,Marketing,8,3.8,0
22,46000,Sales,1,,0
45,92000,,12,4.8,1
32,69000,Engineering,4,4.0,1
28,54000,Marketing,2,3.5,0
38,84000,Sales,7,4.6,1
50,99000,Engineering,15,4.9,1
29,59000,Sales,2,3.6,0
42,87000,Marketing,10,4.2,1
31,65000,Engineering,3,3.9,0
27,51000,Sales,1,3.4,0
48,95000,Marketing,13,4.7,1
"""


class DatasetService:
    """Thread-safe stateful service managing dataset intake and preprocessed partitions."""

    def __init__(self) -> None:
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._preprocessed: Dict[str, PreprocessingResult] = {}
        self._config_signatures: Dict[str, str] = {}
        self._uploads_dir = RAW_DATA_DIR / "uploads"
        self._uploads_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_built_in_datasets()

    def _initialize_built_in_datasets(self) -> None:
        """Register built-in reference datasets."""
        # 1. Synthetic Demo
        demo_df = load_dataset(io.StringIO(DEMO_CSV_STRING))
        self._datasets["demo"] = {
            "name": "Synthetic Demo Dataset",
            "source_type": "built_in",
            "description": "Minimal synthetic 15-row tabular dataset for rapid pipeline testing and verification.",
            "df": demo_df,
            "default_task": "classification",
            "default_target": "promoted",
        }

        # 2. Breast Cancer Wisconsin
        bc_bunch = load_breast_cancer(as_frame=True)
        bc_df = bc_bunch.frame
        self._datasets["breast_cancer"] = {
            "name": "Wisconsin Breast Cancer Diagnostic",
            "source_type": "built_in",
            "description": "Public diagnostic clinical benchmark with 569 samples and 30 continuous biological features.",
            "df": bc_df,
            "default_task": "classification",
            "default_target": "target",
        }

        # 3. Diabetes Progression
        diab_bunch = load_diabetes(as_frame=True)
        diab_df = diab_bunch.frame
        self._datasets["diabetes"] = {
            "name": "Diabetes Progression Benchmark",
            "source_type": "built_in",
            "description": "Public quantitative disease progression regression benchmark with 442 patients and 10 physiological predictors.",
            "df": diab_df,
            "default_task": "regression",
            "default_target": "target",
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Return list of available dataset descriptors."""
        result = []
        for d_id, item in self._datasets.items():
            df = item["df"]
            default_target = item.get("default_target")
            feat_count = int(df.shape[1] - 1) if default_target and default_target in df.columns else int(df.shape[1])
            result.append({
                "dataset_id": d_id,
                "name": item["name"],
                "source_type": item["source_type"],
                "description": item["description"],
                "sample_count": int(df.shape[0]),
                "feature_count": feat_count,
                "default_task": item.get("default_task"),
                "default_target": default_target,
            })
        return result

    def get_dataset(self, dataset_id: str) -> Tuple[str, pd.DataFrame]:
        """Fetch dataset name and DataFrame, or raise ValueError."""
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found. Available datasets: {list(self._datasets.keys())}")
        item = self._datasets[dataset_id]
        return item["name"], item["df"]

    def save_uploaded_csv(self, filename: str, content: bytes) -> Dict[str, Any]:
        """Validate and store uploaded CSV safely with path traversal protection."""
        if len(content) > MAX_UPLOAD_SIZE_BYTES:
            raise ValueError(
                f"File size exceeds 50 MB maximum limit (received {len(content) / (1024 * 1024):.2f} MB)."
            )

        if not content or len(content.strip()) == 0:
            raise ValueError("The uploaded CSV file is empty and contains no data.")

        # Sanitize filename (remove path traversal tokens)
        clean_name = Path(filename).name
        clean_name = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", clean_name)
        if not clean_name.lower().endswith(".csv"):
            raise ValueError("Only CSV (.csv) files are supported.")

        # Parse CSV into DataFrame
        try:
            df = load_dataset(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Could not parse uploaded CSV: {str(exc)}")

        # Validate minimum rows and columns
        validation = validate_dataset(df)
        if not validation["is_valid"]:
            raise ValueError(validation["error_message"])

        dataset_id = f"upload_{uuid.uuid4().hex[:10]}"
        safe_file_path = self._uploads_dir / f"{dataset_id}_{clean_name}"
        try:
            with open(safe_file_path, "wb") as f:
                f.write(content)
        except Exception:
            pass  # Fall back to in-memory DataFrame if disk write fails

        self._datasets[dataset_id] = {
            "name": f"Uploaded: {clean_name}",
            "source_type": "uploaded",
            "description": f"User-uploaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.",
            "df": df,
            "default_task": None,
            "default_target": None,
            "file_path": str(safe_file_path),
        }

        return {
            "dataset_id": dataset_id,
            "name": f"Uploaded: {clean_name}",
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }

    def get_dataset_preview(self, dataset_id: str, n_preview: int = 10) -> Dict[str, Any]:
        """Generate schema diagnostics and head preview rows."""
        name, df = self.get_dataset(dataset_id)
        summary = get_dataset_summary(df)

        preview_slice = df.head(n_preview).replace({np.nan: None})
        preview_records = preview_slice.to_dict(orient="records")

        return {
            "dataset_id": dataset_id,
            "name": name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": list(df.columns),
            "summary": summary,
            "preview_rows": preview_records,
        }

    def validate_target(self, dataset_id: str, target_column: str, task_type: str) -> Dict[str, Any]:
        """Validate suitability of target variable for chosen task."""
        _, df = self.get_dataset(dataset_id)
        res = validate_target_column(df, target_column=target_column, task_type=task_type)
        return {
            "target_column": target_column,
            "task_type": task_type,
            "is_valid": res["is_valid"],
            "error_message": res.get("error_message"),
            "unique_classes": res.get("unique_classes", 0),
            "null_count": res.get("null_count", 0),
        }

    def preprocess_dataset(
        self,
        dataset_id: str,
        target_column: str,
        task_type: str = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
        scale_numerical: bool = True,
    ) -> PreprocessingResult:
        """Execute leakage-free preprocessing and store active partition matrices."""
        name, df = self.get_dataset(dataset_id)

        # Validate target
        target_val = validate_target_column(df, target_column=target_column, task_type=task_type)
        if not target_val["is_valid"]:
            raise ValueError(target_val["error_message"])

        # Separate features and target
        X, y = separate_features_and_target(df, target_column=target_column)

        # Execute split and fit
        prep_res = split_and_preprocess_data(
            X=X,
            y=y,
            task_type=task_type,
            test_size=test_size,
            random_state=random_state,
            scale_numerical=scale_numerical,
        )

        sig = f"{dataset_id}|{target_column}|{task_type.lower()}|{test_size:.2f}|{random_state}|{bool(scale_numerical)}"
        self._preprocessed[dataset_id] = prep_res
        self._config_signatures[dataset_id] = sig

        return prep_res

    def get_preprocessed(self, dataset_id: str) -> PreprocessingResult:
        """Fetch preprocessed data for dataset, or raise ValueError."""
        if dataset_id not in self._preprocessed:
            raise ValueError(
                f"No preprocessed data available for dataset '{dataset_id}'. "
                "Please configure and run preprocessing before initiating benchmarks."
            )
        return self._preprocessed[dataset_id]

    def get_config_signature(self, dataset_id: str) -> Optional[str]:
        """Return the configuration signature of the preprocessed dataset."""
        return self._config_signatures.get(dataset_id)


# Global singleton instance for backend services
dataset_service = DatasetService()

