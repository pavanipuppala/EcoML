import { describe, it, expect, beforeEach, vi } from 'vitest';
import { api } from '../api';

describe('EcoML API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetches health status correctly', async () => {
    const mockHealth = {
      status: 'ok',
      app: 'EcoML REST API',
      version: '0.4.0',
      operational_status: 'HEALTHY',
      tagline: 'Carbon and Energy-Aware Machine Learning Model Selector',
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as any);

    const result = await api.getHealth();
    expect(result).toEqual(mockHealth);
    expect(global.fetch).toHaveBeenCalledWith('/api/health', expect.any(Object));
  });

  it('lists datasets and unwraps response object', async () => {
    const mockDatasets = [
      {
        dataset_id: 'builtin_breast_cancer',
        name: 'Breast Cancer Wisconsin (Diagnostic)',
        source_type: 'built_in',
        description: 'Binary classification benchmark',
        sample_count: 569,
        feature_count: 30,
        default_task: 'classification',
        default_target: 'target',
      },
    ];

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ datasets: mockDatasets }),
    } as any);

    const result = await api.listDatasets();
    expect(result).toEqual(mockDatasets);
    expect(global.fetch).toHaveBeenCalledWith('/api/datasets', expect.any(Object));
  });

  it('sends target validation request with JSON body', async () => {
    const mockValidation = {
      target_column: 'target',
      task_type: 'classification',
      is_valid: true,
      unique_classes: 2,
      null_count: 0,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockValidation,
    } as any);

    const result = await api.validateTarget('ds_1', 'target', 'classification');
    expect(result).toEqual(mockValidation);

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/datasets/ds_1/validate-target',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ target_column: 'target', task_type: 'classification' }),
      })
    );
  });

  it('submits benchmark job with custom parameters', async () => {
    const mockResponse = {
      benchmark_id: 'bench_123',
      dataset_id: 'ds_1',
      status: 'QUEUED',
      total_models: 3,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    } as any);

    const result = await api.submitBenchmark({
      dataset_id: 'ds_1',
      selected_models: ['logistic_regression', 'random_forest'],
      enable_carbon_tracking: true,
      random_state: 99,
    });

    expect(result).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/benchmarks',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          dataset_id: 'ds_1',
          selected_models: ['logistic_regression', 'random_forest'],
          enable_carbon_tracking: true,
          random_state: 99,
        }),
      })
    );
  });

  it('throws ApiError with detail message on non-200 responses', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({ detail: 'Invalid target column specified.' }),
    } as any);

    await expect(api.validateTarget('ds_1', 'bad_col', 'classification')).rejects.toThrow(
      'Invalid target column specified.'
    );
  });
});

