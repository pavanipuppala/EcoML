"""
EcoML — Quantum Optimization API Routes.

Exposes endpoints for simulated QAOA quantum circuits and exact classical
baselines for constrained multi-criteria model selection.
"""

from fastapi import APIRouter, HTTPException, status
from backend.schemas.quantum import (
    QuantumOptimizationRequest,
    QuantumOptimizationResponse,
    QuantumInfoResponse,
)
from backend.services.quantum_service import quantum_service

router = APIRouter(prefix="/api/quantum", tags=["quantum"])


@router.get("/info", response_model=QuantumInfoResponse, status_code=status.HTTP_200_OK)
def get_quantum_info() -> QuantumInfoResponse:
    """Return simulator status, Qiskit presence, and parameter bounds."""
    try:
        return quantum_service.get_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve quantum info: {str(e)}",
        )


@router.post("/optimize", response_model=QuantumOptimizationResponse, status_code=status.HTTP_200_OK)
def run_quantum_optimization(request: QuantumOptimizationRequest) -> QuantumOptimizationResponse:
    """Execute classical exhaustive baseline and simulated QAOA on candidate models."""
    try:
        return quantum_service.optimize(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantum optimization execution failed: {str(e)}",
        )

