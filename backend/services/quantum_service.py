"""
EcoML — Quantum Optimization Service.

Bridges benchmark execution results with QUBO formulation, exact classical baseline,
and simulated QAOA quantum circuits.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from backend.schemas.quantum import (
    QuantumOptimizationRequest,
    QuantumOptimizationResponse,
    QuantumInfoResponse,
    CandidateModelInfo,
    ClassicalBaselineResult,
    QAOAResult,
    BitstringSample,
    ConvergencePoint,
    OptimizationComparison,
)
from backend.services.benchmark_service import benchmark_service
from src.quantum_optimizer import (
    CandidateModelInput,
    OptimizationWeights,
    QuantumModelSelectionProblem,
    ExactClassicalOptimizer,
    QAOAOptimizer,
    HAS_QISKIT,
)


def _get_default_candidate_inputs() -> List[CandidateModelInput]:
    """Default candidate pool used if no benchmark has been executed yet."""
    return [
        CandidateModelInput(
            model_id="logistic_regression",
            model_name="Logistic Regression",
            performance_score=0.9649,
            training_time_sec=0.0072,
            inference_latency_ms=0.0125,
            ram_mb=42.5,
            carbon_kg=0.000001,
        ),
        CandidateModelInput(
            model_id="decision_tree",
            model_name="Decision Tree Classifier",
            performance_score=0.9298,
            training_time_sec=0.0028,
            inference_latency_ms=0.0110,
            ram_mb=41.8,
            carbon_kg=0.0000008,
        ),
        CandidateModelInput(
            model_id="random_forest",
            model_name="Random Forest Classifier",
            performance_score=0.9825,
            training_time_sec=0.1450,
            inference_latency_ms=0.0820,
            ram_mb=56.2,
            carbon_kg=0.000021,
        ),
        CandidateModelInput(
            model_id="knn",
            model_name="K-Nearest Neighbors",
            performance_score=0.9474,
            training_time_sec=0.0041,
            inference_latency_ms=0.0380,
            ram_mb=44.1,
            carbon_kg=0.000002,
        ),
    ]


class QuantumOptimizationService:
    """Service orchestrating QAOA model selection workflows."""

    def get_info(self) -> QuantumInfoResponse:
        """Return engine capabilities and environment state."""
        return QuantumInfoResponse(
            has_qiskit=HAS_QISKIT,
            simulator_type="Qiskit-compatible NumPy/SciPy Statevector" if HAS_QISKIT else "Native NumPy/SciPy Statevector Simulator",
            max_qubits=12,
            supported_depths=[1, 2, 3, 4],
            supported_optimizers=["COBYLA", "Nelder-Mead"],
            description=(
                "EcoML Quantum Optimization Lab executes parameterized QAOA circuits "
                "to find the optimal ML model satisfying resource and predictive constraints. "
                "All quantum statevectors are evolved using exact unitary mathematics and "
                "sampled via the Born rule."
            ),
        )

    def optimize(self, request: QuantumOptimizationRequest) -> QuantumOptimizationResponse:
        """Run classical baseline and simulated QAOA on candidate models."""
        dataset_name = "Wisconsin Diagnostic Breast Cancer (Demo)"
        task_type = "classification"
        candidate_inputs: List[CandidateModelInput] = []

        # 1. Resolve candidate models from benchmark service or defaults
        is_synthetic_demo = True
        target_bench_id = request.benchmark_id
        if not target_bench_id:
            # Look for most recent completed job
            for b_id, job in reversed(list(benchmark_service._jobs.items())):
                if job.get("status") == "COMPLETED" and "results" in job:
                    target_bench_id = b_id
                    break

        if target_bench_id:
            try:
                results_dict = benchmark_service.get_benchmark_results(target_bench_id)
                dataset_name = results_dict.get("dataset_name", "Evaluated Benchmark")
                task_type = results_dict.get("task_type", "classification")
                models_eval = results_dict.get("models_evaluated", [])

                for m in models_eval:
                    # Performance score: accuracy or weighted_f1 for classification, r2 for regression
                    perf = m.get("weighted_f1") or m.get("accuracy")
                    if perf is None:
                        r2_val = m.get("r2")
                        perf = max(0.0, min(1.0, r2_val)) if r2_val is not None else 0.5

                    # Extract telemetry
                    telem = m.get("telemetry", {})
                    train_t = m.get("training_time_sec") or 0.01
                    lat_ms = m.get("inference_latency_ms") or (m.get("inference_time_sec", 0.0001) * 1000.0)
                    ram = telem.get("peak_ram_mb") or 40.0
                    carb = telem.get("emissions_kg")

                    candidate_inputs.append(
                        CandidateModelInput(
                            model_id=m["model_id"],
                            model_name=m["model_name"],
                            performance_score=float(perf),
                            training_time_sec=float(train_t),
                            inference_latency_ms=float(lat_ms),
                            ram_mb=float(ram),
                            carbon_kg=float(carb) if carb is not None else None,
                        )
                    )
                if candidate_inputs:
                    is_synthetic_demo = False
            except Exception:
                # Fallback to default candidates if benchmark results not yet ready
                candidate_inputs = _get_default_candidate_inputs()
        else:
            candidate_inputs = _get_default_candidate_inputs()

        # Filter by selected_model_ids if requested
        if request.selected_model_ids:
            filtered = [c for c in candidate_inputs if c.model_id in request.selected_model_ids]
            if len(filtered) >= 2:
                candidate_inputs = filtered

        # Cap at 10 candidates to keep statevector simulation fast (< 1 second)
        candidate_inputs = candidate_inputs[:10]

        # 2. Formulate Problem
        weights = OptimizationWeights(
            w_performance=request.weights.w_performance if request.weights else 0.40,
            w_train_time=request.weights.w_train_time if request.weights else 0.15,
            w_inference_latency=request.weights.w_inference_latency if request.weights else 0.15,
            w_ram=request.weights.w_ram if request.weights else 0.15,
            w_carbon=request.weights.w_carbon if request.weights else 0.15,
        )

        problem = QuantumModelSelectionProblem(
            candidate_inputs,
            weights=weights,
            penalty_factor=request.penalty_factor,
        )

        # 3. Exact Classical Baseline
        classical_solver = ExactClassicalOptimizer(problem)
        classical_raw = classical_solver.solve()

        best_cl = classical_raw["best_feasible_solution"]
        classical_result = ClassicalBaselineResult(
            execution_time_sec=round(classical_raw["execution_time_sec"], 6),
            total_states_evaluated=classical_raw["total_states_evaluated"],
            selected_model_name=best_cl["selected_model"] if best_cl else "None",
            selected_model_id=best_cl["selected_model_id"] if best_cl else "None",
            bitstring=best_cl["bitstring"] if best_cl else "",
            energy=round(best_cl["energy"], 4) if best_cl else 999.0,
            raw_utility=round(best_cl["raw_utility"], 4) if best_cl else 0.0,
            is_feasible=best_cl["is_feasible"] if best_cl else False,
        )

        # 4. QAOA Optimization
        qaoa_solver = QAOAOptimizer(
            problem,
            depth_p=request.depth_p,
            shots=request.shots,
            classical_method=request.classical_optimizer,
            maxiter=60,
            random_seed=42,
        )
        qaoa_raw = qaoa_solver.optimize()

        # Parse samples
        parsed_samples = [
            BitstringSample(
                bitstring=s["bitstring"],
                counts=s["counts"],
                shot_frequency=s["shot_frequency"],
                theoretical_probability=s["theoretical_probability"],
                is_feasible=s["is_feasible"],
                selected_model=s["selected_model"],
                selected_model_id=s["selected_model_id"],
                raw_utility=s["raw_utility"],
                energy=s["energy"],
            )
            for s in qaoa_raw["samples"]
        ]

        best_q_model = (
            BitstringSample(**qaoa_raw["best_feasible_model"])
            if qaoa_raw["best_feasible_model"]
            else None
        )
        top_state = (
            BitstringSample(**qaoa_raw["top_sampled_state"])
            if qaoa_raw["top_sampled_state"]
            else None
        )

        convergence_pts = [
            ConvergencePoint(iteration=cp["iteration"], energy=cp["energy"])
            for cp in qaoa_raw["convergence_history"]
        ]

        qaoa_result = QAOAResult(
            engine=qaoa_raw["engine"],
            execution_time_sec=round(qaoa_raw["execution_time_sec"], 4),
            depth_p=qaoa_raw["depth_p"],
            shots=qaoa_raw["shots"],
            classical_optimizer=qaoa_raw["classical_optimizer"],
            iterations_completed=qaoa_raw["iterations_completed"],
            optimized_gammas=qaoa_raw["optimized_gammas"],
            optimized_betas=qaoa_raw["optimized_betas"],
            final_expected_energy=qaoa_raw["final_expected_energy"],
            feasibility_rate=qaoa_raw["feasibility_rate"],
            best_feasible_model=best_q_model,
            top_sampled_state=top_state,
            samples=parsed_samples,
            convergence_history=convergence_pts,
            circuit_ascii=qaoa_raw["circuit_ascii"],
            total_quantum_gates=qaoa_raw["total_quantum_gates"],
        )

        # 5. Build Candidate Info Diagnostics
        candidates_out: List[CandidateModelInfo] = []
        for i, c in enumerate(candidate_inputs):
            candidates_out.append(
                CandidateModelInfo(
                    qubit_index=i,
                    model_id=c.model_id,
                    model_name=c.model_name,
                    performance_score=round(c.performance_score, 4),
                    training_time_sec=round(c.training_time_sec, 5),
                    inference_latency_ms=round(c.inference_latency_ms, 4),
                    ram_mb=round(c.ram_mb, 2),
                    carbon_kg=c.carbon_kg,
                    normalized_utility=round(problem.utilities[i], 4),
                    linear_cost=round(problem.linear_costs[i], 4),
                )
            )

        # 6. Empirical Comparison
        agreement = (
            best_cl is not None
            and best_q_model is not None
            and best_cl["selected_model_id"] == best_q_model.selected_model_id
        )

        cl_utility = best_cl["raw_utility"] if best_cl else 0.0
        q_utility = best_q_model.raw_utility if best_q_model else 0.0
        utility_delta = round(q_utility - cl_utility, 4)

        cl_energy = best_cl["energy"] if best_cl else 0.0
        q_energy = best_q_model.energy if best_q_model else 0.0
        energy_delta = round(q_energy - cl_energy, 4)

        if agreement:
            summary = (
                f"QAOA successfully converged to the classical global optimum ({classical_result.selected_model_name}), "
                f"achieving identical composite utility ({cl_utility:.4f}) with a measurement feasibility rate of "
                f"{qaoa_result.feasibility_rate * 100:.1f}%."
            )
        else:
            summary = (
                f"QAOA selected {best_q_model.selected_model if best_q_model else 'None'} "
                f"while Classical Baseline found {classical_result.selected_model_name} (Utility delta: {utility_delta:+.4f}). "
                f"Higher depth p or additional classical iterations can refine quantum convergence."
            )

        comparison = OptimizationComparison(
            agreement=agreement,
            classical_selected_model=classical_result.selected_model_name,
            qaoa_selected_model=best_q_model.selected_model if best_q_model else None,
            utility_delta=utility_delta,
            energy_delta=energy_delta,
            summary=summary,
        )

        # Ising J string dict for JSON serialization
        ising_j_str = {f"{k[0]},{k[1]}": round(v, 4) for k, v in problem.J_coeffs.items()}

        return QuantumOptimizationResponse(
            benchmark_id=target_bench_id,
            dataset_name=dataset_name,
            task_type=task_type,
            penalty_factor=request.penalty_factor,
            candidates=candidates_out,
            qubo_matrix=[[round(float(val), 4) for val in row] for row in problem.qubo_matrix],
            ising_h=[round(float(h), 4) for h in problem.h_coeffs],
            ising_j=ising_j_str,
            classical_baseline=classical_result,
            qaoa_result=qaoa_result,
            comparison=comparison,
            meta={
                "num_qubits": problem.num_models,
                "qubo_offset": round(problem.qubo_offset, 4),
                "c_offset": round(problem.c_offset, 4),
                "penalty_parameter_P": round(problem.penalty, 4),
                "is_simulated": True,
                "is_synthetic_demo": is_synthetic_demo,
                "data_source": "synthetic_demo" if is_synthetic_demo else "active_benchmark",
            },
        )


quantum_service = QuantumOptimizationService()

