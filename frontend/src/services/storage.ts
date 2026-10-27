/**
 * EcoML — Session Storage Persistence Service
 * Preserves user dataset selections, preprocessed partitions, benchmarks,
 * and recommendations across page navigation and browser refreshes (F5).
 */

import {
  DatasetPreview,
  PreprocessingResponse,
  BenchmarkResults,
  RecommendationReport,
} from '../types';

export interface EcoMLSessionState {
  currentTab: string;
  selectedDatasetId: string | null;
  datasetName: string | null;
  datasetPreview: DatasetPreview | null;
  targetCol: string;
  taskType: 'classification' | 'regression';
  prepData: PreprocessingResponse | null;
  activeBenchmarkId: string | null;
  benchmarkResults: BenchmarkResults | null;
  recommendationReport: RecommendationReport | null;
}

const STORAGE_KEY = 'ecoml_session_state_v1';

export const storage = {
  load(): Partial<EcoMLSessionState> | null {
    try {
      if (typeof window === 'undefined' || !window.sessionStorage) return null;
      const serialized = window.sessionStorage.getItem(STORAGE_KEY);
      if (!serialized) return null;
      return JSON.parse(serialized) as Partial<EcoMLSessionState>;
    } catch (err) {
      console.warn('Failed to parse EcoML session state from sessionStorage:', err);
      return null;
    }
  },

  save(state: EcoMLSessionState): void {
    try {
      if (typeof window === 'undefined' || !window.sessionStorage) return;
      window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (err) {
      console.warn('Failed to save EcoML session state to sessionStorage:', err);
    }
  },

  clear(): void {
    try {
      if (typeof window === 'undefined' || !window.sessionStorage) return;
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      console.warn('Failed to clear EcoML session state:', err);
    }
  },
};
