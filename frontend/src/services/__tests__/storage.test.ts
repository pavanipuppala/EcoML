import { describe, it, expect, beforeEach, vi } from 'vitest';
import { storage, EcoMLSessionState } from '../storage';

describe('EcoML Storage Service', () => {
  let mockStore: Record<string, string> = {};

  beforeEach(() => {
    mockStore = {};
    const mockSessionStorage = {
      getItem: vi.fn((key: string) => mockStore[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        mockStore[key] = value;
      }),
      removeItem: vi.fn((key: string) => {
        delete mockStore[key];
      }),
      clear: vi.fn(() => {
        mockStore = {};
      }),
      length: 0,
      key: vi.fn(() => null),
    };

    // Attach to globalThis / window
    (globalThis as any).window = {
      sessionStorage: mockSessionStorage,
    };
  });

  it('returns null when storage is empty', () => {
    const loaded = storage.load();
    expect(loaded).toBeNull();
  });

  it('saves and loads session state accurately', () => {
    const mockState: EcoMLSessionState = {
      currentTab: 'benchmarks',
      selectedDatasetId: 'breast_cancer',
      datasetName: 'Wisconsin Breast Cancer Diagnostic',
      datasetPreview: null,
      targetCol: 'target',
      taskType: 'classification',
      prepData: {
        dataset_id: 'breast_cancer',
        config_signature: 'test_sig',
        task_type: 'classification',
        target_column: 'target',
        train_samples: 455,
        test_samples: 114,
        feature_count: 30,
        numerical_cols: ['mean_radius'],
        categorical_cols: [],
        feature_names_out: ['mean_radius'],
        message: 'Partition complete',
      },
      activeBenchmarkId: 'bench_xyz',
      benchmarkResults: null,
      recommendationReport: null,
    };

    storage.save(mockState);
    const loaded = storage.load();
    expect(loaded).toEqual(mockState);
  });

  it('clears session state on clear()', () => {
    const mockState: EcoMLSessionState = {
      currentTab: 'overview',
      selectedDatasetId: null,
      datasetName: null,
      datasetPreview: null,
      targetCol: '',
      taskType: 'classification',
      prepData: null,
      activeBenchmarkId: null,
      benchmarkResults: null,
      recommendationReport: null,
    };

    storage.save(mockState);
    expect(storage.load()).not.toBeNull();
    storage.clear();
    expect(storage.load()).toBeNull();
  });
});

