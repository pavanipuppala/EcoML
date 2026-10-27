/**
 * EcoML — Typed API Client for FastAPI Backend.
 */

import {
  SystemHealth,
  MetadataFeatures,
  DatasetListItem,
  DatasetPreview,
  TargetValidationResponse,
  PreprocessingResponse,
  ModelInfo,
  BenchmarkJobStatus,
  BenchmarkResults,
  WeightProfile,
  WeightProfilePreset,
  RecommendationReport,
  QuantumOptimizationRequest,
  QuantumOptimizationResponse,
  QuantumInfoResponse,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = new Headers(options.headers || {});

  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson && errJson.detail) {
        errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Ignore JSON parse error on non-JSON response
    }
    throw new ApiError(errorDetail, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // System
  async getHealth(): Promise<SystemHealth> {
    return request<SystemHealth>('/api/health');
  },

  async getMetaFeatures(): Promise<MetadataFeatures> {
    return request<MetadataFeatures>('/api/meta/features');
  },

  // Datasets
  async listDatasets(): Promise<DatasetListItem[]> {
    const res = await request<{ datasets: DatasetListItem[] }>('/api/datasets');
    return res.datasets;
  },

  async uploadDataset(file: File): Promise<{ dataset_id: string; name: string; rows: number; columns: number }> {
    const formData = new FormData();
    formData.append('file', file);
    return request('/api/datasets/upload', {
      method: 'POST',
      body: formData,
    });
  },

  async getDatasetPreview(datasetId: string, nPreview = 10): Promise<DatasetPreview> {
    return request<DatasetPreview>(`/api/datasets/${datasetId}/preview?n_preview=${nPreview}`);
  },

  async validateTarget(
    datasetId: string,
    targetColumn: string,
    taskType: 'classification' | 'regression'
  ): Promise<TargetValidationResponse> {
    return request<TargetValidationResponse>(`/api/datasets/${datasetId}/validate-target`, {
      method: 'POST',
      body: JSON.stringify({ target_column: targetColumn, task_type: taskType }),
    });
  },

  async preprocessDataset(
    datasetId: string,
    params: {
      target_column: string;
      task_type: 'classification' | 'regression';
      test_size?: number;
      random_state?: number;
      scale_numerical?: boolean;
    }
  ): Promise<PreprocessingResponse> {
    return request<PreprocessingResponse>(`/api/datasets/${datasetId}/preprocess`, {
      method: 'POST',
      body: JSON.stringify({
        target_column: params.target_column,
        task_type: params.task_type,
        test_size: params.test_size ?? 0.2,
        random_state: params.random_state ?? 42,
        scale_numerical: params.scale_numerical ?? true,
      }),
    });
  },

  // Benchmarks
  async getSupportedModels(taskType: 'classification' | 'regression'): Promise<ModelInfo[]> {
    const res = await request<{ task_type: string; models: ModelInfo[] }>(`/api/models?task_type=${taskType}`);
    return res.models;
  },

  async submitBenchmark(params: {
    dataset_id: string;
    selected_models?: string[];
    enable_carbon_tracking?: boolean;
    random_state?: number;
  }): Promise<{ benchmark_id: string; dataset_id: string; status: string; total_models: number }> {
    return request('/api/benchmarks', {
      method: 'POST',
      body: JSON.stringify({
        dataset_id: params.dataset_id,
        selected_models: params.selected_models,
        enable_carbon_tracking: params.enable_carbon_tracking ?? false,
        random_state: params.random_state ?? 42,
      }),
    });
  },

  async getBenchmarkStatus(benchmarkId: string): Promise<BenchmarkJobStatus> {
    return request<BenchmarkJobStatus>(`/api/benchmarks/${benchmarkId}`);
  },

  async getBenchmarkResults(benchmarkId: string): Promise<BenchmarkResults> {
    return request<BenchmarkResults>(`/api/benchmarks/${benchmarkId}/results`);
  },

  // Recommendations
  async getRecommendationProfiles(): Promise<WeightProfilePreset[]> {
    const res = await request<{ presets: WeightProfilePreset[] }>('/api/recommendations/profiles');
    return res.presets;
  },

  async calculateRecommendations(
    benchmarkId: string,
    weightProfile?: WeightProfile
  ): Promise<RecommendationReport> {
    return request<RecommendationReport>(`/api/recommendations/${benchmarkId}`, {
      method: 'POST',
      body: JSON.stringify({ weight_profile: weightProfile ?? null }),
    });
  },

  // Quantum Optimization
  async getQuantumInfo(): Promise<QuantumInfoResponse> {
    return request<QuantumInfoResponse>('/api/quantum/info');
  },

  async optimizeQuantum(req: QuantumOptimizationRequest): Promise<QuantumOptimizationResponse> {
    return request<QuantumOptimizationResponse>('/api/quantum/optimize', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },
};

