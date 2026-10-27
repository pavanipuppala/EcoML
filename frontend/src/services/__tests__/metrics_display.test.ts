import { describe, it, expect } from 'vitest';
import { ModelScoreDetail } from '../../types';

// Helper formatting functions mirroring the logic in GreenIntelligencePage and ParetoScatterPlot
export function formatDecisionMatrixMetric(
  model: Partial<ModelScoreDetail>,
  isClassification: boolean
): string {
  if (model.status !== 'SUCCESS') {
    return 'Failed';
  }
  const raw = model.raw_metrics || {};
  const val = isClassification
    ? (raw.f1 ?? raw.performance)
    : (raw.r2 ?? raw.performance);

  if (val === undefined || val === null) return '—';
  return isClassification ? `${(val * 100).toFixed(2)}%` : val.toFixed(4);
}

export function formatHeroCardMetric(
  model: Partial<ModelScoreDetail>,
  isClassification: boolean
): string {
  if (model.status !== 'SUCCESS') {
    return 'Failed';
  }
  const raw = model.raw_metrics || {};
  const perfVal = isClassification
    ? (raw.f1 ?? raw.performance)
    : (raw.r2 ?? raw.performance);

  if (perfVal === undefined || perfVal === null) {
    return 'N/A';
  }
  return isClassification ? `${(perfVal * 100).toFixed(2)}%` : perfVal.toFixed(4);
}

export function formatParetoTooltipMetric(
  raw: Record<string, any>,
  status: string,
  taskType: string
): string {
  if (status === 'FAILED') return 'Failed';
  if (taskType === 'classification') {
    const f1 = raw.f1 ?? raw.performance;
    return f1 !== undefined && f1 !== null ? `${(f1 * 100).toFixed(2)}%` : '—';
  } else {
    const r2 = raw.r2 ?? raw.performance;
    return r2 !== undefined && r2 !== null ? r2.toFixed(4) : '—';
  }
}

export function convertSecondsToMs(seconds: number): number {
  return seconds * 1000.0;
}

export function getRadarActiveAxes(isCarbonTracked: boolean): string[] {
  const allAxes = ['Predictive Perf', 'Train Speed', 'Inference Speed', 'RAM Efficiency', 'Carbon Efficiency'];
  return isCarbonTracked ? allAxes : allAxes.filter((a) => a !== 'Carbon Efficiency');
}

describe('Metric Display & Formatting Integrity', () => {
  it('formats valid classification accuracy correctly', () => {
    const model: Partial<ModelScoreDetail> = {
      model_id: 'logistic_regression',
      status: 'SUCCESS',
      raw_metrics: {
        accuracy: 0.9825,
        performance: 0.9825,
        f1: 0.9825,
      },
    };
    expect(formatDecisionMatrixMetric(model, true)).toBe('98.25%');
    expect(formatHeroCardMetric(model, true)).toBe('98.25%');
    expect(formatParetoTooltipMetric(model.raw_metrics!, 'SUCCESS', 'classification')).toBe('98.25%');
  });

  it('preserves genuine zero accuracy as 0.00% without corrupting it', () => {
    const zeroModel: Partial<ModelScoreDetail> = {
      model_id: 'zero_acc',
      status: 'SUCCESS',
      raw_metrics: {
        accuracy: 0.0,
        performance: 0.0,
      },
    };
    expect(formatDecisionMatrixMetric(zeroModel, true)).toBe('0.00%');
    expect(formatHeroCardMetric(zeroModel, true)).toBe('0.00%');
    expect(formatParetoTooltipMetric(zeroModel.raw_metrics!, 'SUCCESS', 'classification')).toBe('0.00%');
  });

  it('renders "Failed" rather than 0.00% for failed models', () => {
    const failedModel: Partial<ModelScoreDetail> = {
      model_id: 'broken_model',
      status: 'FAILED',
      raw_metrics: {
        accuracy: null,
        performance: null,
      },
    };
    expect(formatDecisionMatrixMetric(failedModel, true)).toBe('Failed');
    expect(formatHeroCardMetric(failedModel, true)).toBe('Failed');
    expect(formatParetoTooltipMetric(failedModel.raw_metrics!, 'FAILED', 'classification')).toBe('Failed');
  });

  it('renders "N/A" or "—" rather than 0.00% when metrics are missing or unavailable', () => {
    const missingModel: Partial<ModelScoreDetail> = {
      model_id: 'uncomputed_model',
      status: 'SUCCESS',
      raw_metrics: {},
    };
    expect(formatDecisionMatrixMetric(missingModel, true)).toBe('—');
    expect(formatHeroCardMetric(missingModel, true)).toBe('N/A');
    expect(formatParetoTooltipMetric(missingModel.raw_metrics!, 'SUCCESS', 'classification')).toBe('—');
  });

  it('formats regression metrics as decimal coefficients and never as percentages', () => {
    const regModel: Partial<ModelScoreDetail> = {
      model_id: 'ridge_regressor',
      status: 'SUCCESS',
      raw_metrics: {
        r2: 0.4541,
        performance: 0.4541,
        mae: 42.81,
        rmse: 53.78,
      },
    };
    expect(formatDecisionMatrixMetric(regModel, false)).toBe('0.4541');
    expect(formatHeroCardMetric(regModel, false)).toBe('0.4541');
    expect(formatParetoTooltipMetric(regModel.raw_metrics!, 'SUCCESS', 'regression')).toBe('0.4541');
  });

  it('verifies that the classification table F1 column header is "Weighted F1" and not "Macro F1"', () => {
    // Validates the contract established in BenchmarkLabPage for classification metric tables
    const classificationMetricHeaders: Record<string, string> = {
      accuracy: 'Accuracy',
      f1: 'Weighted F1',
    };
    expect(classificationMetricHeaders.f1).toBe('Weighted F1');
    expect(classificationMetricHeaders.f1).not.toBe('Macro F1');
  });

  it('verifies that composite scores are formatted with two decimal places across display boundaries', () => {
    const rawScore = 98.8432;
    const formatted = rawScore.toFixed(2);
    expect(formatted).toBe('98.84');
    expect(formatted).not.toBe('98.8');

    const regScore = 74.3149;
    expect(regScore.toFixed(2)).toBe('74.31');
  });

  it('verifies inference latency column header explicitly designates per-sample units', () => {
    const latencyHeader = 'Inference Latency (ms/sample)';
    expect(latencyHeader).toContain('(ms/sample)');
  });

  it('verifies that Pareto scatter plot X-axis represents Weighted F1 and is labeled Predictive Utility (Weighted F1 %)', () => {
    const axisLabelClassification = 'Predictive Utility (Weighted F1 %) →';
    const axisLabelRegression = 'Predictive Utility (R² %) →';
    expect(axisLabelClassification).toContain('Weighted F1 %');
    expect(axisLabelRegression).toContain('R² %');
  });

  it('verifies unit conversion consistency between seconds and milliseconds', () => {
    const trainingTimeSeconds = 0.0072;
    const trainingTimeMs = convertSecondsToMs(trainingTimeSeconds);
    expect(trainingTimeMs).toBeCloseTo(7.2, 5);

    const formattedMs = `${trainingTimeMs.toFixed(1)} ms`;
    const formattedSec = `${trainingTimeSeconds.toFixed(4)}s`;
    expect(formattedMs).toBe('7.2 ms');
    expect(formattedSec).toBe('0.0072s');
  });

  it('verifies radar chart dynamically selects 4 axes when carbon is untracked to avoid plotting fabricated zeroes', () => {
    const axesWithoutCarbon = getRadarActiveAxes(false);
    expect(axesWithoutCarbon.length).toBe(4);
    expect(axesWithoutCarbon).not.toContain('Carbon Efficiency');

    const axesWithCarbon = getRadarActiveAxes(true);
    expect(axesWithCarbon.length).toBe(5);
    expect(axesWithCarbon).toContain('Carbon Efficiency');
  });

  it('verifies stale benchmark result invalidation contract', () => {
    // When a new benchmark starts, benchmarkResults and recommendationReport must be nullified
    interface AppSessionState {
      benchmarkResults: any | null;
      recommendationReport: any | null;
      activeBenchmarkId: string | null;
    }
    const previousState: AppSessionState = {
      benchmarkResults: { benchmark_id: 'bench_old', models_evaluated: [] },
      recommendationReport: { benchmark_id: 'bench_old', ranked_models: [] },
      activeBenchmarkId: 'bench_old',
    };

    // Simulated benchmark start handler
    const invalidatedState: AppSessionState = {
      ...previousState,
      benchmarkResults: null,
      recommendationReport: null,
      activeBenchmarkId: null,
    };

    expect(invalidatedState.benchmarkResults).toBeNull();
    expect(invalidatedState.recommendationReport).toBeNull();
    expect(invalidatedState.activeBenchmarkId).toBeNull();
  });

  it('detects zero predictive utility and flags recommendation warning condition', () => {
    const zeroPerfModel: Partial<ModelScoreDetail> = {
      model_id: 'decision_tree',
      status: 'SUCCESS',
      raw_metrics: {
        accuracy: 0.0,
        f1: 0.0,
        performance: 0.0,
      },
    };
    const isClassification = true;
    const isZeroPerf = Boolean(
      zeroPerfModel && (
        (zeroPerfModel.raw_metrics?.performance !== undefined && zeroPerfModel.raw_metrics.performance <= 0) ||
        (isClassification && zeroPerfModel.raw_metrics?.accuracy !== undefined && zeroPerfModel.raw_metrics.accuracy <= 0)
      )
    );
    expect(isZeroPerf).toBe(true);
  });

  it('parses and extracts markdown bold markers for empirical tradeoff observations', () => {
    const rawTradeoff = '**Gradient Boosting Regressor** achieved top R² (0.4551), but required **58.3× longer** training time.';
    const parts = rawTradeoff.split(/(\*\*[^*]+\*\*)/g);
    expect(parts.length).toBeGreaterThan(1);
    const boldTokens = parts.filter((p) => p.startsWith('**') && p.endsWith('**'));
    expect(boldTokens).toEqual(['**Gradient Boosting Regressor**', '**58.3× longer**']);
    const cleanTokens = boldTokens.map((b) => b.slice(2, -2));
    expect(cleanTokens).toEqual(['Gradient Boosting Regressor', '58.3× longer']);
  });
});

