"""
Tests for QAOA Quantum Optimizer and Classical Optimization Baseline.
"""

import pytest
import numpy as np

from src.quantum_optimizer import (
    CandidateModelInput,
    OptimizationWeights,
    QuantumModelSelectionProblem,
    ExactClassicalOptimizer,
    QuantumStatevectorSimulator,
    QAOAOptimizer,
)


def sample_candidates():
    return [
        CandidateModelInput(
            model_id="logistic_regression",
            model_name="Logistic Regression",
            performance_score=0.95,
            training_time_sec=0.010,
            inference_latency_ms=0.020,
            ram_mb=45.0,
            carbon_kg=0.00001,
        ),
        CandidateModelInput(
            model_id="decision_tree",
            model_name="Decision Tree",
            performance_score=0.92,
            training_time_sec=0.005,
            inference_latency_ms=0.015,
            ram_mb=42.0,
            carbon_kg=0.000005,
        ),
        CandidateModelInput(
            model_id="random_forest",
            model_name="Random Forest",
            performance_score=0.98,
            training_time_sec=0.150,
            inference_latency_ms=0.080,
            ram_mb=65.0,
            carbon_kg=0.00008,
        ),
    ]


def test_qubo_and_ising_formulation():
    candidates = sample_candidates()
    problem = QuantumModelSelectionProblem(candidates)

    assert problem.num_models == 3
    assert len(problem.utilities) == 3
    assert len(problem.linear_costs) == 3
    assert problem.qubo_matrix.shape == (3, 3)
    assert len(problem.h_coeffs) == 3
    assert len(problem.J_coeffs) == 3  # (0,1), (0,2), (1,2)

    # Check that feasible states (Hamming weight 1) have no penalty
    res_feasible = problem.evaluate_bitstring("010")
    assert res_feasible["is_feasible"] is True
    assert res_feasible["penalty"] == 0.0
    assert res_feasible["selected_model"] == "Decision Tree"

    # Check that infeasible states have positive penalty
    res_infeasible_0 = problem.evaluate_bitstring("000")
    assert res_infeasible_0["is_feasible"] is False
    assert res_infeasible_0["penalty"] > 0.0

    res_infeasible_2 = problem.evaluate_bitstring("110")
    assert res_infeasible_2["is_feasible"] is False
    assert res_infeasible_2["penalty"] > 0.0

    # Feasible state must have lower energy than infeasible states
    assert res_feasible["energy"] < res_infeasible_0["energy"]
    assert res_feasible["energy"] < res_infeasible_2["energy"]


def test_classical_exact_solver():
    candidates = sample_candidates()
    problem = QuantumModelSelectionProblem(candidates)
    solver = ExactClassicalOptimizer(problem)
    res = solver.solve()

    assert res["total_states_evaluated"] == 8
    assert res["best_feasible_solution"] is not None
    assert res["best_feasible_solution"]["is_feasible"] is True
    # Global ground state must be the best feasible solution because penalty P is sufficiently large
    assert res["global_ground_state"]["bitstring"] == res["best_feasible_solution"]["bitstring"]


def test_statevector_simulator_properties():
    candidates = sample_candidates()
    problem = QuantumModelSelectionProblem(candidates)
    sim = QuantumStatevectorSimulator(problem)

    init_state = sim.get_initial_state()
    # Norm of initial state must be 1.0
    norm = np.sum(np.abs(init_state) ** 2)
    assert pytest.approx(norm, 1e-6) == 1.0

    # Unitary evolution must preserve norm
    state_after_cost = sim.apply_cost_unitary(init_state, 0.5)
    assert pytest.approx(np.sum(np.abs(state_after_cost) ** 2), 1e-6) == 1.0

    state_after_mixer = sim.apply_mixer_unitary(state_after_cost, 0.3)
    assert pytest.approx(np.sum(np.abs(state_after_mixer) ** 2), 1e-6) == 1.0

    # Full simulation
    final_state = sim.simulate_circuit([0.5], [0.3])
    assert pytest.approx(np.sum(np.abs(final_state) ** 2), 1e-6) == 1.0

    # Sampling
    shots = 500
    counts = sim.sample_shots(final_state, shots=shots, seed=42)
    assert sum(counts.values()) == shots


def test_qaoa_optimizer_execution():
    candidates = sample_candidates()
    problem = QuantumModelSelectionProblem(candidates)
    qaoa = QAOAOptimizer(
        problem,
        depth_p=1,
        shots=512,
        classical_method="COBYLA",
        maxiter=15,
        random_seed=42,
    )
    res = qaoa.optimize()

    assert res["depth_p"] == 1
    assert res["shots"] == 512
    assert res["best_feasible_model"] is not None
    assert res["best_feasible_model"]["is_feasible"] is True
    assert res["feasibility_rate"] > 0.0
    assert "circuit_ascii" in res
    assert "total_quantum_gates" in res
    assert len(res["convergence_history"]) > 0



def test_exhaustive_16_state_qubo_ising_equivalence():
    """
    Exhaustively enumerate all 16 bitstrings for the 4-candidate problem.
    Verify that QUBO objective, Ising eigenvalue, and state evaluation agree for every state.
    """
    from backend.services.quantum_service import _get_default_candidate_inputs
    candidates = _get_default_candidate_inputs()
    problem = QuantumModelSelectionProblem(candidates, penalty_factor=2.0)
    assert problem.num_models == 4

    Q = problem.qubo_matrix
    qubo_offset = problem.qubo_offset
    h = problem.h_coeffs
    J = problem.J_coeffs
    c_offset = problem.c_offset

    for idx in range(16):
        bitstring = format(idx, '04b')
        x = np.array([int(b) for b in bitstring], dtype=float)

        # 1. QUBO algebraic objective: x^T Q x + offset
        qubo_val = float(x @ Q @ x + qubo_offset)

        # 2. Ising eigenvalue: sum(h_i * s_i) + sum(J_ij * s_i * s_j) + c_offset where s_i = 1 - 2*x_i
        s = 1.0 - 2.0 * x
        ising_val = float(np.sum(h * s))
        for (i, j), j_val in J.items():
            ising_val += j_val * s[i] * s[j]
        ising_val += c_offset

        # 3. Problem evaluate_bitstring
        eval_res = problem.evaluate_bitstring(bitstring)

        assert abs(qubo_val - ising_val) < 1e-12, f"QUBO != Ising for {bitstring}: {qubo_val} vs {ising_val}"
        assert abs(qubo_val - eval_res["energy"]) < 1e-12, f"QUBO != eval for {bitstring}"


def test_measurement_histogram_shot_counts_and_probabilities():
    """
    Verify that displayed bar lengths and labels originate from Born-rule sampling.
    Counts must sum to exactly 1024 shots, probabilities sum to 1.0, and
    the feasibility rate matches the 48.5% measured distribution.
    """
    from backend.services.quantum_service import _get_default_candidate_inputs
    candidates = _get_default_candidate_inputs()
    problem = QuantumModelSelectionProblem(candidates, penalty_factor=2.0)
    qaoa = QAOAOptimizer(
        problem,
        depth_p=2,
        shots=1024,
        classical_method="COBYLA",
        maxiter=60,
        random_seed=42,
    )
    res = qaoa.optimize()

    # Total counts must sum to exactly 1024
    total_shots = sum(s["counts"] for s in res["samples"])
    assert total_shots == 1024

    # Frequency must sum to 1.0
    total_freq = sum(s["shot_frequency"] for s in res["samples"])
    assert pytest.approx(total_freq, 1e-3) == 1.0

    # Theoretical probabilities across all states must sum to 1.0
    sim = QuantumStatevectorSimulator(problem)
    final_state = sim.simulate_circuit(res["optimized_gammas"], res["optimized_betas"])
    probs = np.abs(final_state) ** 2
    assert pytest.approx(np.sum(probs), 1e-6) == 1.0

    # Feasible states breakdown: exactly 4 one-hot states
    feasible_samples = [s for s in res["samples"] if s["is_feasible"]]
    infeasible_samples = [s for s in res["samples"] if not s["is_feasible"]]

    feasible_shots = sum(s["counts"] for s in feasible_samples)
    infeasible_shots = sum(s["counts"] for s in infeasible_samples)
    assert feasible_shots + infeasible_shots == 1024

    # Measured feasibility rate: 497 / 1024 = 48.535%
    feasibility_rate = feasible_shots / 1024.0
    assert pytest.approx(res["feasibility_rate"], 1e-4) == feasibility_rate
    assert pytest.approx(feasibility_rate, 0.01) == 0.485  # ~48.5%


def test_expected_energy_decomposition_and_consistency():
    """
    Confirm why QAOA expected energy is +0.9851 while classical optimum is -0.8498.
    Directly compute <H_C> from the final statevector and cost Hamiltonian.
    """
    from backend.services.quantum_service import _get_default_candidate_inputs
    candidates = _get_default_candidate_inputs()
    problem = QuantumModelSelectionProblem(candidates, penalty_factor=2.0)
    qaoa = QAOAOptimizer(
        problem,
        depth_p=2,
        shots=1024,
        classical_method="COBYLA",
        maxiter=60,
        random_seed=42,
    )
    res = qaoa.optimize()

    sim = QuantumStatevectorSimulator(problem)
    final_state = sim.simulate_circuit(res["optimized_gammas"], res["optimized_betas"])
    expected_energy = sim.compute_energy_expectation(res["optimized_gammas"], res["optimized_betas"])
    direct_expectation = float(np.sum(np.abs(final_state)**2 * sim.eigenvalues))
    assert pytest.approx(expected_energy, 1e-10) == direct_expectation

    # Statevector expectation matches reported final_expected_energy (~0.9851)
    assert pytest.approx(res["final_expected_energy"], 1e-4) == expected_energy
    assert pytest.approx(expected_energy, 1e-3) == 0.9851

    # Classical optimum energy is -0.8498
    solver = ExactClassicalOptimizer(problem)
    cl_res = solver.solve()
    assert pytest.approx(cl_res["best_feasible_solution"]["energy"], 1e-4) == -0.8498

    # QAOA best feasible sampled model achieves identical ground state energy (-0.8498)
    assert res["best_feasible_model"] is not None
    assert res["best_feasible_model"]["bitstring"] == "1000"
    assert pytest.approx(res["best_feasible_model"]["energy"], 1e-4) == -0.8498

    # Decomposition into feasible and infeasible subspaces
    probs = np.abs(final_state) ** 2
    feasible_indices = [1 << (3 - i) for i in range(4)]  # 8, 4, 2, 1
    p_feas = sum(probs[idx] for idx in feasible_indices)
    p_infeas = 1.0 - p_feas

    feas_contrib = sum(probs[idx] * sim.eigenvalues[idx] for idx in feasible_indices)
    infeas_contrib = sum(probs[idx] * sim.eigenvalues[idx] for idx in range(16) if idx not in feasible_indices)

    assert pytest.approx(feas_contrib + infeas_contrib, 1e-6) == expected_energy
    assert feas_contrib < 0.0  # Feasible states have negative energy
    assert infeas_contrib > 0.0  # Penalized infeasible states shift ensemble <H_C> positive


def test_bitstring_decoding_and_one_hot_penalty():
    """
    Verify bitstring decoding, candidate selection, and penalty barrier.
    """
    from backend.services.quantum_service import _get_default_candidate_inputs
    candidates = _get_default_candidate_inputs()
    problem = QuantumModelSelectionProblem(candidates, penalty_factor=2.0)

    # Qubit 0 (MSB) -> Logistic Regression
    ev0 = problem.evaluate_bitstring("1000")
    assert ev0["is_feasible"] is True
    assert ev0["selected_model_id"] == "logistic_regression"
    assert ev0["penalty"] == 0.0

    # Qubit 1 -> Decision Tree
    ev1 = problem.evaluate_bitstring("0100")
    assert ev1["is_feasible"] is True
    assert ev1["selected_model_id"] == "decision_tree"
    assert ev1["penalty"] == 0.0

    # Qubit 2 -> Random Forest
    ev2 = problem.evaluate_bitstring("0010")
    assert ev2["is_feasible"] is True
    assert ev2["selected_model_id"] == "random_forest"
    assert ev2["penalty"] == 0.0

    # Qubit 3 (LSB) -> KNN
    ev3 = problem.evaluate_bitstring("0001")
    assert ev3["is_feasible"] is True
    assert ev3["selected_model_id"] == "knn"
    assert ev3["penalty"] == 0.0

    # Infeasible: Hamming weight != 1
    ev_zero = problem.evaluate_bitstring("0000")
    assert ev_zero["is_feasible"] is False
    assert ev_zero["penalty"] >= 3.0

    ev_multi = problem.evaluate_bitstring("1100")
    assert ev_multi["is_feasible"] is False
    assert ev_multi["penalty"] >= 3.0

    # Every infeasible state has energy strictly higher than any feasible state
    max_feas_energy = max(ev["energy"] for ev in [ev0, ev1, ev2, ev3])
    for idx in range(16):
        bs = format(idx, '04b')
        ev = problem.evaluate_bitstring(bs)
        if not ev["is_feasible"]:
            assert ev["energy"] > max_feas_energy
