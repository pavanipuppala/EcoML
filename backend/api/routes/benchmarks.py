"""
EcoML — Model Benchmarking and Execution Routes.
"""

from fastapi import APIRouter, HTTPException, Query, status
from backend.schemas.benchmarks import (
    SupportedModelsResponse,
    BenchmarkSubmissionRequest,
    BenchmarkJobStatusResponse,
    BenchmarkResultsResponse,
)
from backend.services.benchmark_service import benchmark_service

router = APIRouter(tags=["Model Benchmarks"])


@router.get("/api/models", response_model=SupportedModelsResponse)
def get_models(task_type: str = Query("classification", description="'classification' or 'regression'")) -> SupportedModelsResponse:
    """List available algorithm implementations supported for the chosen task."""
    try:
        models = benchmark_service.get_supported_model_list(task_type=task_type)
        return SupportedModelsResponse(task_type=task_type.lower(), models=models)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


@router.post("/api/benchmarks", status_code=status.HTTP_202_ACCEPTED)
def submit_benchmark(req: BenchmarkSubmissionRequest) -> dict:
    """Submit a benchmark run across selected algorithms on preprocessed data partitions."""
    try:
        res = benchmark_service.submit_benchmark_job(
            dataset_id=req.dataset_id,
            selected_models=req.selected_models,
            enable_carbon_tracking=req.enable_carbon_tracking,
            random_state=req.random_state,
        )
        return {
            "status": "accepted",
            "message": "Benchmark job submitted and queued for execution.",
            **res,
        }
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Benchmark submission failed: {str(exc)}")


@router.get("/api/benchmarks/{benchmark_id}", response_model=BenchmarkJobStatusResponse)
def get_benchmark_status(benchmark_id: str) -> BenchmarkJobStatusResponse:
    """Poll current execution lifecycle state and completed model count for a benchmark job."""
    try:
        data = benchmark_service.get_job_status(benchmark_id=benchmark_id)
        return BenchmarkJobStatusResponse(**data)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@router.get("/api/benchmarks/{benchmark_id}/results", response_model=BenchmarkResultsResponse)
def get_benchmark_results(benchmark_id: str) -> BenchmarkResultsResponse:
    """Retrieve complete multi-model benchmark results, latency, RAM, and carbon metrics."""
    try:
        data = benchmark_service.get_job_results(benchmark_id=benchmark_id)
        return BenchmarkResultsResponse(**data)
    except ValueError as exc:
        # If in progress, return 409 Conflict with detail
        err_msg = str(exc)
        if "in progress" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
