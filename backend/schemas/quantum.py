"""
EcoML — Pydantic Schemas for Quantum Optimization (QAOA) API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuantumWeightProfile(BaseModel):
    """Priority weights for quantum multi-criteria optimization."""
    w_performance: float = Field(0.40, ge=0.0, description="Predictive quality weight")
    w_train_time: float = Field(0.15, ge=0.0, description="Training time efficiency weight")
    w_inference_latency: float = Field(0.15, ge=0.0, description="Latency efficiency weight")
    w_ram: float = Field(0.15, ge=0.0, description="RAM memory efficiency weight")
    w_carbon: float = Field(0.15, ge=0.0, description="Carbon emission efficiency weight")


class QuantumOptimizationRequest(BaseModel):
    """Request payload to trigger classical baseline and simulated QAOA model selection."""
    benchmark_id: Optional[str] = Field(None, description="Active benchmark ID to pull model results from")
    selected_model_ids: Optional[List[str]] = Field(None, description="Subset of model IDs to optimize (3-10 models)")
    weights: Optional[QuantumWeightProfile] = Field(None, description="Custom multi-criteria weights")
    depth_p: int = Field(2, ge=1, le=4, description="QAOA circuit layer depth (p)")
    shots: int = Field(1024, ge=100, le=4096, description="Quantum Born-rule measurement shots")
    classical_optimizer: str = Field("COBYLA", description="Classical optimizer for variational angles: COBYLA or Nelder-Mead")
    penalty_factor: float = Field(2.0, ge=1.0, le=10.0, description="Constraint penalty multiplier P")


class CandidateModelInfo(BaseModel):
    """Diagnostic profile of candidate model mapped into quantum decision variable."""
    qubit_index: int
    model_id: str
    model_name: str
    performance_score: float
    training_time_sec: float
    inference_latency_ms: float
    ram_mb: float
    carbon_kg: Optional[float] = None
    normalized_utility: float
    linear_cost: float


class ClassicalBaselineResult(BaseModel):
    """Exact classical optimization result via exhaustive 2^n state evaluation."""
    execution_time_sec: float
    total_states_evaluated: int
    selected_model_name: str
    selected_model_id: str
    bitstring: str
    energy: float
    raw_utility: float
    is_feasible: bool


class BitstringSample(BaseModel):
    """Measured computational basis state with Born probability and evaluation."""
    bitstring: str
    counts: int
    shot_frequency: float
    theoretical_probability: float
    is_feasible: bool
    selected_model: Optional[str] = None
    selected_model_id: Optional[str] = None
    raw_utility: float
    energy: float


class ConvergencePoint(BaseModel):
    """Iteration snapshot of classical variational angle optimization."""
    iteration: int
    energy: float


class QAOAResult(BaseModel):
    """QAOA execution telemetry and measurement results."""
    engine: str
    execution_time_sec: float
    depth_p: int
    shots: int
    classical_optimizer: str
    iterations_completed: int
    optimized_gammas: List[float]
    optimized_betas: List[float]
    final_expected_energy: float
    feasibility_rate: float
    best_feasible_model: Optional[BitstringSample] = None
    top_sampled_state: Optional[BitstringSample] = None
    samples: List[BitstringSample]
    convergence_history: List[ConvergencePoint]
    circuit_ascii: str
    total_quantum_gates: int


class OptimizationComparison(BaseModel):
    """Empirical comparison between Exact Classical baseline and QAOA."""
    agreement: bool
    classical_selected_model: str
    qaoa_selected_model: Optional[str] = None
    utility_delta: float
    energy_delta: float
    summary: str


class QuantumOptimizationResponse(BaseModel):
    """Full optimization results payload."""
    benchmark_id: Optional[str] = None
    dataset_name: str
    task_type: str
    penalty_factor: float
    candidates: List[CandidateModelInfo]
    qubo_matrix: List[List[float]]
    ising_h: List[float]
    ising_j: Dict[str, float]
    classical_baseline: ClassicalBaselineResult
    qaoa_result: QAOAResult
    comparison: OptimizationComparison
    meta: Dict[str, Any]


class QuantumInfoResponse(BaseModel):
    """Metadata regarding system quantum optimization capabilities."""
    has_qiskit: bool
    simulator_type: str
    max_qubits: int
    supported_depths: List[int]
    supported_optimizers: List[str]
    description: str

