"""
EcoML — Dataset Ingestion, Diagnostics & Preprocessing Routes.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from backend.schemas.datasets import (
    DatasetListResponse,
    DatasetPreviewResponse,
    TargetValidationRequest,
    TargetValidationResponse,
    PreprocessingRequest,
    PreprocessingResponse,
)
from backend.services.dataset_service import dataset_service

router = APIRouter(prefix="/api/datasets", tags=["Datasets & Preprocessing"])


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """Retrieve catalog of built-in reference benchmarks and user-uploaded datasets."""
    datasets = dataset_service.list_datasets()
    return DatasetListResponse(datasets=datasets)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_csv_dataset(file: UploadFile = File(...)) -> dict:
    """Upload and validate a custom tabular CSV dataset (maximum 50 MB)."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only standard comma-separated tabular files (.csv) are supported.",
        )

    try:
        content = await file.read()
        res = dataset_service.save_uploaded_csv(filename=file.filename, content=content)
        return {
            "status": "success",
            "message": "Dataset successfully uploaded and validated.",
            **res,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Upload processing failed: {str(exc)}")


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(dataset_id: str, n_preview: int = 10) -> DatasetPreviewResponse:
    """Retrieve schema diagnostics, column dtypes, missingness rates, and preview records."""
    try:
        data = dataset_service.get_dataset_preview(dataset_id=dataset_id, n_preview=n_preview)
        return DatasetPreviewResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/{dataset_id}/validate-target", response_model=TargetValidationResponse)
def validate_target_column(dataset_id: str, req: TargetValidationRequest) -> TargetValidationResponse:
    """Verify that chosen target variable satisfies mathematical constraints for the task."""
    try:
        res = dataset_service.validate_target(
            dataset_id=dataset_id,
            target_column=req.target_column,
            task_type=req.task_type,
        )
        return TargetValidationResponse(**res)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.post("/{dataset_id}/preprocess", response_model=PreprocessingResponse)
def preprocess_dataset(dataset_id: str, req: PreprocessingRequest) -> PreprocessingResponse:
    """Construct and fit scikit-learn transformers strictly on training partition."""
    try:
        prep_res = dataset_service.preprocess_dataset(
            dataset_id=dataset_id,
            target_column=req.target_column,
            task_type=req.task_type,
            test_size=req.test_size,
            random_state=req.random_state,
            scale_numerical=req.scale_numerical,
        )
        sig = dataset_service.get_config_signature(dataset_id) or ""
        return PreprocessingResponse(
            dataset_id=dataset_id,
            config_signature=sig,
            task_type=prep_res.task_type,
            target_column=prep_res.target_column,
            train_samples=prep_res.train_shape[0],
            test_samples=prep_res.test_shape[0],
            feature_count=prep_res.train_shape[1],
            numerical_cols=prep_res.numerical_cols,
            categorical_cols=prep_res.categorical_cols,
            feature_names_out=prep_res.feature_names_out,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

