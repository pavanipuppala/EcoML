/**
 * EcoML — Frontend TypeScript Type Definitions
 * Exact mirrors of the FastAPI backend Pydantic schemas.
 */

export interface SystemHealth {
  status: string;
  app: string;
  version: string;
  operational_status: string;
  tagline: string;
}

export interface FeatureItem {
  id: string;
  title: string;
  icon: string;
  description: string;
  status: string;
  stage: string;
}

export interface RoadmapStep {
  step: string;
  title: string;
  status: string;
  summary: string;
}

export interface MetadataFeatures {
  implemented_features: FeatureItem[];
  future_features: FeatureItem[];
  roadmap_steps: RoadmapStep[];
}

export interface DatasetListItem {
  dataset_id: string;
  name: string;
  source_type: 'built_in' | 'uploaded';
  description: string;
  sample_count?: number | null;
  feature_count?: number | null;
  default_task?: 'classification' | 'regression' | null;
  default_target?: string | null;
}

export interface ColumnSummary {
  shape: number[];
  rows: number;
  columns: number;
  column_names: string[];
  dtypes: Record<string, string>;
  missing_counts: Record<string, number>;
  missing_percentages: Record<string, number>;
  total_missing: number;
  duplicate_rows: number;
  numerical_columns: string[];
  categorical_columns: string[];
  memory_usage_kb: number;
}

export interface DatasetPreview {
  dataset_id: string;
  name: string;
  rows: number;
  columns: number;
  column_names: string[];
  summary: ColumnSummary;
  preview_rows: Record<string, any>[];
}

export interface TargetValidationResponse {
  target_column: string;
  task_type: string;
  is_valid: boolean;
  error_message?: string | null;
  unique_classes: number;
  null_count: number;
}

export interface PreprocessingResponse {
  dataset_id: string;
  config_signature: string;
  task_type: 'classification' | 'regression';
  target_column: string;
  train_samples: number;
  test_samples: number;
  feature_count: number;
  numerical_cols: string[];
  categorical_cols: string[];
  feature_names_out: string[];
  message: string;
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  task_type: string;
}

export interface TelemetryMetrics {
  baseline_ram_mb?: number | null;
  peak_ram_mb?: number | null;
  ram_delta_mb?: number | null;
  avg_cpu_percent?: number | null;
  peak_cpu_percent?: number | null;
  energy_kwh?: number | null;
  carbon_emissions_g?: number | null;
  energy_status: string;
  telemetry_overhead_ms?: number | null;
}

export interface SingleModelResult {
  model_id: string;
  model_name: string;
  task_type: string;
  status: string;
  error_message?: string | null;
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  confusion_matrix?: number[][] | null;
  mae?: number | null;
  rmse?: number | null;
  r2?: number | null;
  training_time_sec?: number | null;
  prediction_time_sec?: number | null;
  latency_per_sample_ms?: number | null;
  telemetry: TelemetryMetrics;
}

export interface BenchmarkJobStatus {
  benchmark_id: string;
  dataset_id: string;
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  progress_percent: number;
  completed_models: number;
  total_models: number;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface BenchmarkResults {
  benchmark_id: string;
  dataset_id: string;
  dataset_name: string;
  task_type: 'classification' | 'regression';
  target_column: string;
  sample_count_train: number;
  sample_count_test: number;
  feature_count: number;
  benchmark_timestamp: string;
  models_evaluated: SingleModelResult[];
  tradeoff_interpretation?: {
    tradeoff_observation: string;
    best_predictive_model: string;
    best_predictive_metric: string;
    fastest_training_model: string;
    lowest_latency_model: string;
  } | null;
}

export interface WeightProfile {
  name: string;
  w_performance: number;
  w_train_time: number;
  w_inference_latency: number;
  w_ram: number;
  w_carbon: number;
}

export interface WeightProfilePreset {
  profile_id: string;
  name: string;
  description: string;
  weights: WeightProfile;
}

export interface ModelScoreDetail {
  model_id: string;
  model_name: string;
  status: string;
  rank: number;
  composite_score: number;
  is_pareto_optimal: boolean;
  raw_metrics: Record<string, any>;
  normalized_scores: Record<string, number>;
  effective_weights: Record<string, number>;
  strengths: string[];
  tradeoff_notes: string[];
}

export interface RecommendationReport {
  benchmark_id: string;
  dataset_name: string;
  task_type: string;
  profile_name: string;
  ranked_models: ModelScoreDetail[];
  top_recommended_model?: ModelScoreDetail | null;
  pareto_frontier_models: string[];
  effective_weights: Record<string, number>;
  redistribution_note?: string | null;
  tradeoff_synthesis: string;
  scoring_methodology: string;
}

export interface QuantumWeightProfile {
  w_performance: number;
  w_train_time: number;
  w_inference_latency: number;
  w_ram: number;
  w_carbon: number;
}

export interface QuantumOptimizationRequest {
  benchmark_id?: string | null;
  selected_model_ids?: string[] | null;
  weights?: QuantumWeightProfile | null;
  depth_p: number;
  shots: number;
  classical_optimizer: string;
  penalty_factor: number;
}

export interface CandidateModelInfo {
  qubit_index: number;
  model_id: string;
  model_name: string;
  performance_score: number;
  training_time_sec: number;
  inference_latency_ms: number;
  ram_mb: number;
  carbon_kg?: number | null;
  normalized_utility: number;
  linear_cost: number;
}

export interface ClassicalBaselineResult {
  execution_time_sec: number;
  total_states_evaluated: number;
  selected_model_name: string;
  selected_model_id: string;
  bitstring: string;
  energy: number;
  raw_utility: number;
  is_feasible: boolean;
}

export interface BitstringSample {
  bitstring: string;
  counts: number;
  shot_frequency: number;
  theoretical_probability: number;
  is_feasible: boolean;
  selected_model?: string | null;
  selected_model_id?: string | null;
  raw_utility: number;
  energy: number;
}

export interface ConvergencePoint {
  iteration: number;
  energy: number;
}

export interface QAOAResult {
  engine: string;
  execution_time_sec: number;
  depth_p: number;
  shots: number;
  classical_optimizer: string;
  iterations_completed: number;
  optimized_gammas: number[];
  optimized_betas: number[];
  final_expected_energy: number;
  feasibility_rate: number;
  best_feasible_model?: BitstringSample | null;
  top_sampled_state?: BitstringSample | null;
  samples: BitstringSample[];
  convergence_history: ConvergencePoint[];
  circuit_ascii: string;
  total_quantum_gates: number;
}

export interface OptimizationComparison {
  agreement: boolean;
  classical_selected_model: string;
  qaoa_selected_model?: string | null;
  utility_delta: number;
  energy_delta: number;
  summary: string;
}

export interface QuantumOptimizationResponse {
  benchmark_id?: string | null;
  dataset_name: string;
  task_type: string;
  penalty_factor: number;
  candidates: CandidateModelInfo[];
  qubo_matrix: number[][];
  ising_h: number[];
  ising_j: Record<string, number>;
  classical_baseline: ClassicalBaselineResult;
  qaoa_result: QAOAResult;
  comparison: OptimizationComparison;
  meta: Record<string, any>;
}

export interface QuantumInfoResponse {
  has_qiskit: boolean;
  simulator_type: string;
  max_qubits: number;
  supported_depths: number[];
  supported_optimizers: string[];
  description: string;
}


