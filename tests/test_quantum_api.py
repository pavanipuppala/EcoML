"""
EcoML — Quantum API Integration Tests.
"""

import pytest
from backend.api.routes.quantum import get_quantum_info, run_quantum_optimization
from backend.schemas.quantum import QuantumOptimizationRequest, QuantumWeightProfile


def test_quantum_info_endpoint():
    info = get_quantum_info()
    assert info.max_qubits >= 10
    assert 2 in info.supported_depths
    assert "COBYLA" in info.supported_optimizers
    assert "EcoML" in info.description


def test_quantum_optimization_endpoint_with_default_models():
    req = QuantumOptimizationRequest(
        depth_p=1,
        shots=256,
        classical_optimizer="COBYLA",
        penalty_factor=2.0,
        weights=QuantumWeightProfile(
            w_performance=0.5,
            w_train_time=0.2,
            w_inference_latency=0.1,
            w_ram=0.1,
            w_carbon=0.1,
        ),
    )
    res = run_quantum_optimization(req)

    assert len(res.candidates) >= 3
    assert res.classical_baseline.is_feasible is True
    assert res.classical_baseline.selected_model_name != "None"
    assert res.qaoa_result.depth_p == 1
    assert res.qaoa_result.shots == 256
    assert res.qaoa_result.best_feasible_model is not None
    assert res.qaoa_result.best_feasible_model.is_feasible is True
    assert len(res.qubo_matrix) == len(res.candidates)
    assert len(res.ising_h) == len(res.candidates)
    assert res.comparison.summary != ""
    assert res.meta["is_simulated"] is True
    assert res.meta["is_synthetic_demo"] is True
    assert res.meta["data_source"] == "synthetic_demo" 


def test_quantum_optimization_custom_subset():
    req = QuantumOptimizationRequest(
        selected_model_ids=["logistic_regression", "decision_tree", "random_forest"],
        depth_p=2,
        shots=512,
        classical_optimizer="COBYLA",
    )
    res = run_quantum_optimization(req)

    assert len(res.candidates) == 3
    assert res.classical_baseline.total_states_evaluated == 8
    assert res.qaoa_result.depth_p == 2

