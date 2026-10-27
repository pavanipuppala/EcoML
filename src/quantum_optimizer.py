"""
EcoML — QAOA Quantum Approximate Optimization Engine for Green Model Selection.

Formulates constrained multi-criteria model selection as a Quadratic Unconstrained
Binary Optimization (QUBO) problem and maps it to an Ising Cost Hamiltonian (Pauli-Z).
Simulates QAOA parameterized quantum circuits via an exact statevector simulator
and compares empirical results against an exact classical exhaustive baseline.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    import qiskit
    from qiskit import QuantumCircuit
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    QuantumCircuit = None


@dataclass
class CandidateModelInput:
    """Normalized candidate model profile evaluated in EcoML benchmark."""
    model_id: str
    model_name: str
    performance_score: float  # e.g., Accuracy, F1, or R2 in [0, 1]
    training_time_sec: float
    inference_latency_ms: float
    ram_mb: float
    carbon_kg: Optional[float] = None


@dataclass
class OptimizationWeights:
    """Multi-criteria priority weights balancing performance and resource consumption."""
    w_performance: float = 0.40
    w_train_time: float = 0.15
    w_inference_latency: float = 0.15
    w_ram: float = 0.15
    w_carbon: float = 0.15

    def normalized_dict(self, has_carbon: bool = True) -> Dict[str, float]:
        """Return non-negative normalized weights summing to 1.0."""
        w_c = self.w_carbon if has_carbon else 0.0
        total = (
            max(0.0, self.w_performance)
            + max(0.0, self.w_train_time)
            + max(0.0, self.w_inference_latency)
            + max(0.0, self.w_ram)
            + max(0.0, w_c)
        )
        if total <= 1e-9:
            return {
                "performance": 0.4,
                "train_time": 0.2,
                "inference_latency": 0.2,
                "ram": 0.2,
                "carbon": 0.0,
            }
        return {
            "performance": max(0.0, self.w_performance) / total,
            "train_time": max(0.0, self.w_train_time) / total,
            "inference_latency": max(0.0, self.w_inference_latency) / total,
            "ram": max(0.0, self.w_ram) / total,
            "carbon": max(0.0, w_c) / total,
        }


class QuantumModelSelectionProblem:
    """
    Formulates EcoML model selection as a QUBO and Ising Cost Hamiltonian.

    Decision variables: x_i in {0, 1} for i in {0, ..., n-1}.
    Constraint: Exactly one model must be selected, sum_{i=0}^{n-1} x_i = 1.
    Linear Cost: f_i = -U_i (where U_i in [0, 1] is the composite multi-criteria utility).
    Penalty: P * (sum x_i - 1)^2.
    QUBO Objective: min_x x^T Q x + P.
    Ising Mapping: x_i = (I - Z_i) / 2 -> H_C = sum h_i Z_i + sum J_{ij} Z_i Z_j + C_offset.
    """

    def __init__(
        self,
        candidates: List[CandidateModelInput],
        weights: Optional[OptimizationWeights] = None,
        penalty_factor: float = 2.0,
    ) -> None:
        if not candidates:
            raise ValueError("At least one candidate model is required for optimization.")
        if len(candidates) > 14:
            raise ValueError(f"Quantum statevector simulation supports up to 14 qubits (received {len(candidates)}).")

        self.candidates = candidates
        self.num_models = len(candidates)
        self.weights = weights or OptimizationWeights()

        # Check if carbon measurements exist across candidates
        has_carbon = any(c.carbon_kg is not None and c.carbon_kg > 0 for c in candidates)
        self.norm_weights = self.weights.normalized_dict(has_carbon=has_carbon)

        # Compute normalized utilities and linear cost coefficients f_i
        self._compute_utilities()

        # Formulate penalty P and QUBO matrix Q
        max_abs_f = max(abs(f) for f in self.linear_costs) if self.linear_costs else 1.0
        self.penalty = penalty_factor * max(1.0, max_abs_f) + 1.0

        self._build_qubo_and_ising()

    def _compute_utilities(self) -> None:
        """Normalize candidate metrics and compute composite utility U_i in [0, 1]."""
        n = self.num_models
        perfs = [c.performance_score for c in self.candidates]
        trains = [c.training_time_sec for c in self.candidates]
        lats = [c.inference_latency_ms for c in self.candidates]
        rams = [c.ram_mb for c in self.candidates]
        carbs = [c.carbon_kg if c.carbon_kg is not None else 0.0 for c in self.candidates]

        def min_max_norm(vals: List[float]) -> List[float]:
            min_v, max_v = min(vals), max(vals)
            rng = max_v - min_v
            if rng <= 1e-9:
                return [0.0] * len(vals)
            return [(v - min_v) / rng for v in vals]

        norm_perf = min_max_norm(perfs)
        norm_train = min_max_norm(trains)
        norm_lat = min_max_norm(lats)
        norm_ram = min_max_norm(rams)
        norm_carb = min_max_norm(carbs)

        self.utilities: List[float] = []
        self.linear_costs: List[float] = []

        w_p = self.norm_weights["performance"]
        w_t = self.norm_weights["train_time"]
        w_l = self.norm_weights["inference_latency"]
        w_m = self.norm_weights["ram"]
        w_c = self.norm_weights["carbon"]

        for i in range(n):
            # Performance is maximized (norm_perf), resource costs are minimized (1 - norm_cost)
            u_i = (
                w_p * norm_perf[i]
                + w_t * (1.0 - norm_train[i])
                + w_l * (1.0 - norm_lat[i])
                + w_m * (1.0 - norm_ram[i])
                + w_c * (1.0 - norm_carb[i])
            )
            # Bound utility in [0, 1]
            u_i = max(0.0, min(1.0, u_i))
            self.utilities.append(u_i)
            # Minimizing cost f_i = -u_i directly maximizes utility
            self.linear_costs.append(-u_i)

    def _build_qubo_and_ising(self) -> None:
        """
        Build QUBO matrix Q and Pauli-Z Ising Hamiltonian parameters.

        QUBO formulation:
        Cost(x) = sum f_i x_i + P (sum x_i - 1)^2
                = sum (f_i - P) x_i + 2P sum_{i < j} x_i x_j + P
        """
        n = self.num_models
        P = self.penalty
        Q = np.zeros((n, n), dtype=float)

        for i in range(n):
            Q[i, i] = self.linear_costs[i] - P
            for j in range(i + 1, n):
                Q[i, j] = 2.0 * P
                Q[j, i] = 0.0  # Upper-triangular standard

        self.qubo_matrix = Q
        self.qubo_offset = P

        # Ising formulation: x_i = (I - Z_i)/2
        # H_C = sum h_i Z_i + sum_{i < j} J_{ij} Z_i Z_j + C_offset
        self.h_coeffs = np.zeros(n, dtype=float)
        self.J_coeffs: Dict[Tuple[int, int], float] = {}

        for i in range(n):
            # h_i = -1/2 * (f_i - P) - 1/4 * sum_{j != i} 2P
            self.h_coeffs[i] = -0.5 * (self.linear_costs[i] - P) - 0.5 * P * (n - 1)
            for j in range(i + 1, n):
                self.J_coeffs[(i, j)] = 0.5 * P  # 1/4 * (2P)

        # C_offset calculation
        num_pairs = (n * (n - 1)) / 2.0
        self.c_offset = (
            P
            + 0.5 * sum(self.linear_costs[i] - P for i in range(n))
            + 0.5 * P * num_pairs
        )

    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Evaluate objective value, constraint feasibility, and selected model for a bitstring.

        Args:
            bitstring: Binary string of length n (e.g. '0100').
        """
        x = [int(b) for b in bitstring]
        selected_indices = [i for i, val in enumerate(x) if val == 1]
        is_feasible = (len(selected_indices) == 1)

        # Raw multi-criteria utility
        if is_feasible:
            sel_idx = selected_indices[0]
            raw_utility = self.utilities[sel_idx]
            selected_model = self.candidates[sel_idx].model_name
            selected_model_id = self.candidates[sel_idx].model_id
        else:
            raw_utility = 0.0
            selected_model = None
            selected_model_id = None

        # Classical QUBO cost F(x) = sum f_i x_i + P (sum x_i - 1)^2
        sum_x = sum(x)
        penalty_val = self.penalty * ((sum_x - 1) ** 2)
        linear_val = sum(self.linear_costs[i] * x[i] for i in range(self.num_models))
        qubo_energy = linear_val + penalty_val

        return {
            "bitstring": bitstring,
            "is_feasible": is_feasible,
            "selected_model": selected_model,
            "selected_model_id": selected_model_id,
            "selected_indices": selected_indices,
            "raw_utility": raw_utility,
            "penalty": penalty_val,
            "energy": qubo_energy,
        }

    def get_basis_state_energy(self, state_int: int) -> float:
        """Compute the exact Ising eigenvalue E_C(z) = <z| H_C |z> for basis index state_int."""
        # Convert integer to bit array of length n (MSB to LSB: bit 0 is qubit 0)
        # Using convention qubit i is bit i
        n = self.num_models
        z_vals = np.zeros(n, dtype=float)
        for i in range(n):
            bit = (state_int >> (n - 1 - i)) & 1
            z_vals[i] = 1.0 if bit == 0 else -1.0  # |0> -> +1, |1> -> -1

        # Energy = sum h_i Z_i + sum J_ij Z_i Z_j + C_offset
        energy = self.c_offset + np.dot(self.h_coeffs, z_vals)
        for (i, j), J_val in self.J_coeffs.items():
            energy += J_val * z_vals[i] * z_vals[j]
        return float(energy)


class ExactClassicalOptimizer:
    """
    Exhaustive classical baseline solver.

    Evaluates all 2^n bitstrings, computing exact QUBO energies,
    identifying the global ground state and best feasible solution.
    """

    def __init__(self, problem: QuantumModelSelectionProblem) -> None:
        self.problem = problem

    def solve(self) -> Dict[str, Any]:
        start_time = time.perf_counter()
        n = self.problem.num_models
        total_states = 1 << n

        evaluations: List[Dict[str, Any]] = []
        best_feasible: Optional[Dict[str, Any]] = None
        global_min_state: Optional[Dict[str, Any]] = None

        for state_int in range(total_states):
            bitstring = format(state_int, f"0{n}b")
            res = self.problem.evaluate_bitstring(bitstring)
            evaluations.append(res)

            if global_min_state is None or res["energy"] < global_min_state["energy"]:
                global_min_state = res

            if res["is_feasible"]:
                if best_feasible is None or res["energy"] < best_feasible["energy"]:
                    best_feasible = res

        elapsed_sec = time.perf_counter() - start_time

        return {
            "execution_time_sec": elapsed_sec,
            "total_states_evaluated": total_states,
            "global_ground_state": global_min_state,
            "best_feasible_solution": best_feasible,
            "evaluations": evaluations,
        }


class QuantumStatevectorSimulator:
    """
    Exact Statevector Quantum Circuit Simulator for QAOA.

    Implements:
    - Initial state |+>^{tensor n} = 1/sqrt(2^n) sum_z |z>
    - Cost unitary U(C, gamma) = exp(-i * gamma * H_C) via exact diagonal phase shifts
    - Mixer unitary U(M, beta) = prod exp(-i * beta * X_i) = prod R_X(2 * beta)
    - Born rule probability evaluation p(z) = |psi(z)|^2
    - Shot sampling from categorical distribution
    """

    def __init__(self, problem: QuantumModelSelectionProblem) -> None:
        self.problem = problem
        self.num_qubits = problem.num_models
        self.num_states = 1 << self.num_qubits

        # Precompute diagonal eigenvalues E_C(z) for z in 0..2^n - 1
        self.eigenvalues = np.zeros(self.num_states, dtype=float)
        for z in range(self.num_states):
            self.eigenvalues[z] = self.problem.get_basis_state_energy(z)

    def get_initial_state(self) -> np.ndarray:
        """Create equal superposition |+>^{tensor n}."""
        val = 1.0 / math.sqrt(self.num_states)
        return np.full(self.num_states, val, dtype=np.complex128)

    def apply_cost_unitary(self, statevector: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply U(C, gamma) = exp(-i * gamma * H_C) to the statevector.
        Since H_C is diagonal in the computational basis, this is an exact elementwise phase rotation.
        """
        phases = np.exp(-1j * gamma * self.eigenvalues)
        return statevector * phases

    def apply_mixer_unitary(self, statevector: np.ndarray, beta: float) -> np.ndarray:
        """
        Apply transverse mixer U(M, beta) = prod_{k=0}^{n-1} exp(-i * beta * X_k).
        For each qubit k, applies single-qubit rotation R_X(2*beta) across all pairs of basis states.
        """
        n = self.num_qubits
        cos_b = math.cos(beta)
        sin_b = -1j * math.sin(beta)

        # Apply single qubit rotation along X to each qubit k
        state = statevector.copy()
        for k in range(n):
            stride = 1 << (n - 1 - k)
            new_state = np.empty_like(state)
            for z in range(self.num_states):
                if (z & stride) == 0:
                    # Qubit k is 0 in z, and 1 in z | stride
                    z0 = z
                    z1 = z | stride
                    v0 = state[z0]
                    v1 = state[z1]
                    new_state[z0] = cos_b * v0 + sin_b * v1
                    new_state[z1] = sin_b * v0 + cos_b * v1
            state = new_state
        return state

    def simulate_circuit(
        self,
        gammas: List[float],
        betas: List[float],
    ) -> np.ndarray:
        """
        Simulate depth-p QAOA circuit.

        |psi(gamma, beta)> = prod_{l=1}^p [ U(M, beta_l) U(C, gamma_l) ] |+>^{tensor n}
        """
        p = len(gammas)
        state = self.get_initial_state()
        for l in range(p):
            state = self.apply_cost_unitary(state, gammas[l])
            state = self.apply_mixer_unitary(state, betas[l])
        return state

    def compute_energy_expectation(
        self,
        gammas: List[float],
        betas: List[float],
    ) -> float:
        """Compute expectation <H_C> = sum_z |psi(z)|^2 * E_C(z)."""
        state = self.simulate_circuit(gammas, betas)
        probs = np.abs(state) ** 2
        return float(np.sum(probs * self.eigenvalues))

    def sample_shots(
        self,
        statevector: np.ndarray,
        shots: int = 1024,
        seed: Optional[int] = 42,
    ) -> Dict[str, int]:
        """Sample bitstrings according to Born rule probabilities p(z) = |psi(z)|^2."""
        probs = np.abs(statevector) ** 2
        # Normalize probabilities to ensure sum = 1.0 (guarding floating point imprecision)
        total_p = np.sum(probs)
        if total_p > 0:
            probs = probs / total_p
        else:
            probs = np.full(self.num_states, 1.0 / self.num_states)

        rng = np.random.default_rng(seed)
        sampled_indices = rng.choice(self.num_states, size=shots, p=probs)

        counts: Dict[str, int] = {}
        n = self.num_qubits
        for idx in sampled_indices:
            bs = format(idx, f"0{n}b")
            counts[bs] = counts.get(bs, 0) + 1

        # Sort by frequency descending
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


class QAOAOptimizer:
    """
    End-to-End QAOA Optimizer for EcoML.

    Orchestrates:
    - Classical optimization loop (COBYLA / Nelder-Mead) over variational parameters (gamma, beta).
    - Energy convergence tracking across iterations.
    - Post-optimization shot sampling.
    - Extraction of best feasible model and solution quality metrics.
    - Generation of ASCII / structural quantum circuit representation.
    """

    def __init__(
        self,
        problem: QuantumModelSelectionProblem,
        depth_p: int = 2,
        shots: int = 1024,
        classical_method: str = "COBYLA",
        maxiter: int = 60,
        random_seed: int = 42,
    ) -> None:
        self.problem = problem
        self.depth_p = max(1, min(depth_p, 4))
        self.shots = max(100, min(shots, 4096))
        self.classical_method = classical_method if classical_method in ["COBYLA", "Nelder-Mead"] else "COBYLA"
        self.maxiter = max(10, min(maxiter, 150))
        self.random_seed = random_seed
        self.simulator = QuantumStatevectorSimulator(problem)

    def generate_ascii_circuit(self) -> str:
        """Generate a clean ASCII visualization of the parameterized QAOA quantum circuit."""
        n = self.problem.num_models
        p = self.depth_p
        lines: List[str] = []

        header = f"=== QAOA Parameterized Circuit Architecture (p = {p}, n = {n} Qubits) ==="
        lines.append(header)
        lines.append("")

        for q in range(n):
            model_lbl = self.problem.candidates[q].model_id[:8].upper()
            line = f"q{q} [{model_lbl:>8}] : |0> -[ H ]-"
            for step in range(1, p + 1):
                line += f"-[ Rz(g{step}) ]-[ Rzz(g{step}) ]-[ Rx(b{step}) ]-"
            line += "-[ M ]-> Bit"
            lines.append(line)

        lines.append("")
        lines.append("Circuit Gate Summary:")
        lines.append(f"  - Initial Hadamards: {n}")
        num_pairs = (n * (n - 1)) // 2
        lines.append(f"  - Cost Unitary U(C, g): {p} layers x ({n} Rz + {num_pairs} Rzz entangling pairs)")
        lines.append(f"  - Mixer Unitary U(M, b): {p} layers x ({n} Rx single-qubit rotations)")
        lines.append(f"  - Measurements: {n} projective Pauli-Z detectors")
        total_gates = n + p * (n + num_pairs + n)
        lines.append(f"  - Total Quantum Gate Operations: {total_gates}")
        return "\n".join(lines)

    def optimize(self) -> Dict[str, Any]:
        """Execute full QAOA parameter optimization and measurement workflow."""
        start_time = time.perf_counter()
        p = self.depth_p

        # Initial parameter heuristics:
        # gamma in [0, 2*pi], beta in [0, pi]
        rng = np.random.default_rng(self.random_seed)
        init_gammas = rng.uniform(0.1, 1.0, size=p)
        init_betas = rng.uniform(0.1, 0.8, size=p)
        init_params = np.concatenate([init_gammas, init_betas])

        convergence_history: List[Dict[str, Any]] = []
        iteration_counter = 0

        def cost_func(params: np.ndarray) -> float:
            nonlocal iteration_counter
            g = list(params[:p])
            b = list(params[p:])
            energy = self.simulator.compute_energy_expectation(g, b)
            iteration_counter += 1
            if iteration_counter <= self.maxiter:
                convergence_history.append({
                    "iteration": iteration_counter,
                    "energy": round(float(energy), 6),
                })
            return energy

        # Run classical parameter optimization
        opt_res = minimize(
            cost_func,
            init_params,
            method=self.classical_method,
            options={"maxiter": self.maxiter},
        )

        opt_gammas = list(opt_res.x[:p])
        opt_betas = list(opt_res.x[p:])

        # Final statevector evolution
        final_state = self.simulator.simulate_circuit(opt_gammas, opt_betas)
        final_energy = float(self.simulator.compute_energy_expectation(opt_gammas, opt_betas))

        # Sample bitstrings
        sample_counts = self.simulator.sample_shots(
            final_state,
            shots=self.shots,
            seed=self.random_seed,
        )

        # Theoretical probabilities
        probs = np.abs(final_state) ** 2
        n = self.problem.num_models

        # Analyze sampled results
        all_samples: List[Dict[str, Any]] = []
        feasible_count = 0
        best_feasible_sample: Optional[Dict[str, Any]] = None
        top_sampled: Optional[Dict[str, Any]] = None

        for bs, count in sample_counts.items():
            state_idx = int(bs, 2)
            eval_info = self.problem.evaluate_bitstring(bs)
            shot_prob = count / self.shots
            theo_prob = float(probs[state_idx])

            sample_entry = {
                "bitstring": bs,
                "counts": count,
                "shot_frequency": round(shot_prob, 4),
                "theoretical_probability": round(theo_prob, 4),
                "is_feasible": eval_info["is_feasible"],
                "selected_model": eval_info["selected_model"],
                "selected_model_id": eval_info["selected_model_id"],
                "raw_utility": round(eval_info["raw_utility"], 4),
                "energy": round(eval_info["energy"], 4),
            }
            all_samples.append(sample_entry)

            if top_sampled is None:
                top_sampled = sample_entry

            if eval_info["is_feasible"]:
                feasible_count += count
                if best_feasible_sample is None or eval_info["energy"] < best_feasible_sample["energy"]:
                    best_feasible_sample = sample_entry

        feasibility_rate = feasible_count / self.shots
        elapsed_sec = time.perf_counter() - start_time

        # Circuit representation
        circuit_ascii = self.generate_ascii_circuit()

        return {
            "engine": "native_statevector_simulator" if not HAS_QISKIT else "qiskit_compatible_statevector",
            "execution_time_sec": elapsed_sec,
            "depth_p": p,
            "shots": self.shots,
            "classical_optimizer": self.classical_method,
            "iterations_completed": iteration_counter,
            "optimized_gammas": [round(float(g), 5) for g in opt_gammas],
            "optimized_betas": [round(float(b), 5) for b in opt_betas],
            "final_expected_energy": round(final_energy, 4),
            "feasibility_rate": round(feasibility_rate, 4),
            "best_feasible_model": best_feasible_sample,
            "top_sampled_state": top_sampled,
            "samples": all_samples[:16],  # Return top 16 distinct sampled states
            "convergence_history": convergence_history[::max(1, len(convergence_history) // 20)],  # Decimate to ~20 points
            "circuit_ascii": circuit_ascii,
            "total_quantum_gates": n + p * (n + (n * (n - 1)) // 2 + n),
        }

