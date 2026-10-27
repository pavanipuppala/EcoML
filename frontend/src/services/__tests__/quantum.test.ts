import { describe, it, expect, beforeEach, vi } from 'vitest';
import { api } from '../api';
import { QuantumOptimizationRequest } from '../../types';

describe('Quantum Optimization API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches quantum simulator metadata', async () => {
    const mockInfo = {
      has_qiskit: false,
      simulator_type: 'Native NumPy/SciPy Statevector Simulator',
      max_qubits: 12,
      supported_depths: [1, 2, 3, 4],
      supported_optimizers: ['COBYLA', 'Nelder-Mead'],
      description: 'Test quantum info',
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockInfo,
    } as any);

    const res = await api.getQuantumInfo();
    expect(res).toEqual(mockInfo);
    expect(global.fetch).toHaveBeenCalledWith('/api/quantum/info', expect.any(Object));
  });

  it('submits quantum optimization request and parses results payload', async () => {
    const mockResponse = {
      benchmark_id: 'bench_123',
      dataset_name: 'Test Dataset',
      task_type: 'classification',
      penalty_factor: 2.0,
      candidates: [
        {
          qubit_index: 0,
          model_id: 'logistic_regression',
          model_name: 'Logistic Regression',
          performance_score: 0.95,
          training_time_sec: 0.01,
          inference_latency_ms: 0.02,
          ram_mb: 45.0,
          carbon_kg: null,
          normalized_utility: 0.85,
          linear_cost: -0.85,
        },
      ],
      qubo_matrix: [[-2.85]],
      ising_h: [1.425],
      ising_j: {},
      classical_baseline: {
        execution_time_sec: 0.0001,
        total_states_evaluated: 2,
        selected_model_name: 'Logistic Regression',
        selected_model_id: 'logistic_regression',
        bitstring: '1',
        energy: -0.85,
        raw_utility: 0.85,
        is_feasible: true,
      },
      qaoa_result: {
        engine: 'native_statevector_simulator',
        execution_time_sec: 0.05,
        depth_p: 1,
        shots: 512,
        classical_optimizer: 'COBYLA',
        iterations_completed: 10,
        optimized_gammas: [0.5],
        optimized_betas: [0.3],
        final_expected_energy: -0.80,
        feasibility_rate: 0.92,
        best_feasible_model: {
          bitstring: '1',
          counts: 470,
          shot_frequency: 0.92,
          theoretical_probability: 0.93,
          is_feasible: true,
          selected_model: 'Logistic Regression',
          selected_model_id: 'logistic_regression',
          raw_utility: 0.85,
          energy: -0.85,
        },
        samples: [],
        convergence_history: [],
        circuit_ascii: 'ASCII Circuit',
        total_quantum_gates: 6,
      },
      comparison: {
        agreement: true,
        classical_selected_model: 'Logistic Regression',
        qaoa_selected_model: 'Logistic Regression',
        utility_delta: 0.0,
        energy_delta: 0.0,
        summary: 'Agreement test summary',
      },
      meta: {
        num_qubits: 1,
        is_simulated: true,
        is_synthetic_demo: true,
        data_source: 'synthetic_demo',
      },
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as any);

    const payload: QuantumOptimizationRequest = {
      benchmark_id: 'bench_123',
      depth_p: 1,
      shots: 512,
      classical_optimizer: 'COBYLA',
      penalty_factor: 2.0,
    };

    const res = await api.optimizeQuantum(payload);
    expect(res).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/quantum/optimize',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(payload),
      })
    );
  });

  it('validates measurement histogram shot counts and feasibility calculation', () => {
    const samples = [
      { bitstring: '1000', counts: 147, shot_frequency: 0.1436, theoretical_probability: 0.145, is_feasible: true, selected_model: 'LR', selected_model_id: 'lr', raw_utility: 0.85, energy: -0.8498 },
      { bitstring: '0001', counts: 132, shot_frequency: 0.1289, theoretical_probability: 0.130, is_feasible: true, selected_model: 'KNN', selected_model_id: 'knn', raw_utility: 0.67, energy: -0.6714 },
      { bitstring: '0100', counts: 112, shot_frequency: 0.1094, theoretical_probability: 0.110, is_feasible: true, selected_model: 'DT', selected_model_id: 'dt', raw_utility: 0.63, energy: -0.6300 },
      { bitstring: '0010', counts: 106, shot_frequency: 0.1035, theoretical_probability: 0.105, is_feasible: true, selected_model: 'RF', selected_model_id: 'rf', raw_utility: 0.42, energy: -0.4209 },
      { bitstring: '0000', counts: 98, shot_frequency: 0.0957, theoretical_probability: 0.095, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 3.0000 },
      { bitstring: '1001', counts: 84, shot_frequency: 0.0820, theoretical_probability: 0.080, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 1.4788 },
      { bitstring: '0101', counts: 79, shot_frequency: 0.0771, theoretical_probability: 0.075, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 1.6986 },
      { bitstring: '1100', counts: 76, shot_frequency: 0.0742, theoretical_probability: 0.075, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 1.5199 },
      { bitstring: '1010', counts: 71, shot_frequency: 0.0693, theoretical_probability: 0.070, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 1.7291 },
      { bitstring: '0110', counts: 60, shot_frequency: 0.0586, theoretical_probability: 0.060, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 1.9489 },
      { bitstring: '1110', counts: 23, shot_frequency: 0.0225, theoretical_probability: 0.022, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 4.0697 },
      { bitstring: '1101', counts: 17, shot_frequency: 0.0166, theoretical_probability: 0.017, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 4.0284 },
      { bitstring: '1011', counts: 11, shot_frequency: 0.0107, theoretical_probability: 0.011, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 4.2376 },
      { bitstring: '0111', counts: 8, shot_frequency: 0.0078, theoretical_probability: 0.008, is_feasible: false, selected_model: null, selected_model_id: null, raw_utility: 0.0, energy: 4.4574 },
    ];

    const totalShots = samples.reduce((sum, s) => sum + s.counts, 0);
    expect(totalShots).toBe(1024);

    const feasibleShots = samples.filter((s) => s.is_feasible).reduce((sum, s) => sum + s.counts, 0);
    expect(feasibleShots).toBe(497);

    const feasibilityRate = feasibleShots / totalShots;
    expect(feasibilityRate).toBeCloseTo(0.48535, 4);

    const infeasibleShots = samples.filter((s) => !s.is_feasible).reduce((sum, s) => sum + s.counts, 0);
    expect(infeasibleShots).toBe(527);
    expect(feasibleShots + infeasibleShots).toBe(1024);
  });

});
