import React, { useState, useEffect } from 'react';
import {
  Leaf,
  Award,
  ArrowRight,
  Info,
  AlertCircle,
} from 'lucide-react';
import { DataTable, ColumnDef } from '../components/DataTable';
import { ParetoScatterPlot } from '../components/ParetoScatterPlot';
import { RadarChart } from '../components/RadarChart';
import { FormattedText } from '../components/FormattedText';
import {
  BenchmarkResults,
  WeightProfile,
  WeightProfilePreset,
  RecommendationReport,
  ModelScoreDetail,
} from '../types';
import { api } from '../services/api';

interface GreenIntelligencePageProps {
  onNavigate: (tab: string) => void;
  activeBenchmarkId: string | null;
  benchmarkResults: BenchmarkResults | null;
  recommendationReport: RecommendationReport | null;
  setRecommendationReport: (rep: RecommendationReport | null) => void;
}

export const GreenIntelligencePage: React.FC<GreenIntelligencePageProps> = ({
  onNavigate,
  activeBenchmarkId,
  benchmarkResults,
  recommendationReport,
  setRecommendationReport,
}) => {
  const [presets, setPresets] = useState<WeightProfilePreset[]>([]);
  const [selectedProfileId, setSelectedProfileId] = useState<string>('balanced');
  const [isCustom, setIsCustom] = useState<boolean>(false);
  const [customWeights, setCustomWeights] = useState<WeightProfile>({
    name: 'Custom',
    w_performance: 0.4,
    w_train_time: 0.15,
    w_inference_latency: 0.15,
    w_ram: 0.15,
    w_carbon: 0.15,
  });

  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Fetch presets on mount
  useEffect(() => {
    api
      .getRecommendationProfiles()
      .then((res) => {
        setPresets(res);
      })
      .catch((err) => {
        console.error('Failed to load recommendation profiles:', err);
      });
  }, []);

  // Compute recommendation if benchmark is available and either no report yet or report is from a different benchmark
  useEffect(() => {
    if (
      activeBenchmarkId &&
      (!recommendationReport || recommendationReport.benchmark_id !== activeBenchmarkId) &&
      presets.length > 0 &&
      !loading
    ) {
      calculateReport(selectedProfileId);
    }
  }, [activeBenchmarkId, presets, recommendationReport?.benchmark_id]);

  const calculateReport = async (profileId: string, custom?: WeightProfile) => {
    if (!activeBenchmarkId) return;

    setLoading(true);
    setErrorMsg(null);

    try {
      let weightPayload: WeightProfile | undefined = undefined;

      if (profileId === 'custom' && custom) {
        weightPayload = custom;
      } else {
        const found = presets.find((p) => p.profile_id === profileId);
        if (found) weightPayload = found.weights;
      }

      const rep = await api.calculateRecommendations(activeBenchmarkId, weightPayload);
      setRecommendationReport(rep);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to compute recommendations');
    } finally {
      setLoading(false);
    }
  };

  const handleProfileChange = (profileId: string) => {
    setSelectedProfileId(profileId);
    if (profileId === 'custom') {
      setIsCustom(true);
    } else {
      setIsCustom(false);
      calculateReport(profileId);
    }
  };

  const handleCustomSliderChange = (key: keyof WeightProfile, val: number) => {
    const updated = { ...customWeights, [key]: val };
    setCustomWeights(updated);
  };

  const handleApplyCustomWeights = () => {
    // Normalize weights to sum to 1
    const total =
      customWeights.w_performance +
      customWeights.w_train_time +
      customWeights.w_inference_latency +
      customWeights.w_ram +
      customWeights.w_carbon;

    const normalized: WeightProfile = {
      name: 'Custom Profile',
      w_performance: +(customWeights.w_performance / total).toFixed(3),
      w_train_time: +(customWeights.w_train_time / total).toFixed(3),
      w_inference_latency: +(customWeights.w_inference_latency / total).toFixed(3),
      w_ram: +(customWeights.w_ram / total).toFixed(3),
      w_carbon: +(customWeights.w_carbon / total).toFixed(3),
    };

    setCustomWeights(normalized);
    calculateReport('custom', normalized);
  };

  // If no benchmark results, show prerequisite guard
  if (!benchmarkResults || !activeBenchmarkId) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            backgroundColor: 'rgba(16, 185, 129, 0.12)',
            border: '1px solid var(--border-emerald)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem auto',
            color: 'var(--eco-emerald-light)',
          }}
        >
          <Leaf size={24} />
        </div>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
          Execute Benchmarks Before Computing Green Intelligence
        </h3>
        <p style={{ maxWidth: '540px', margin: '0 auto 1.5rem auto', fontSize: '0.88rem', lineHeight: '1.6' }}>
          No model telemetry results are currently available. Please execute model benchmarks in the Benchmark Lab tab
          before analyzing multi-criteria trade-offs and calculating EcoML scores.
        </p>
        <button className="btn btn-primary" onClick={() => onNavigate('benchmarks')}>
          <span>Open Benchmark Lab</span>
          <ArrowRight size={16} />
        </button>
      </div>
    );
  }

  const topModel = recommendationReport?.top_recommended_model;
  const isClassification = benchmarkResults.task_type === 'classification';
  const isZeroPerf = Boolean(
    topModel && (
      (topModel.raw_metrics.performance !== undefined && topModel.raw_metrics.performance <= 0) ||
      (isClassification && topModel.raw_metrics.accuracy !== undefined && topModel.raw_metrics.accuracy <= 0)
    )
  );

  // Decision Matrix Table Columns
  const decisionMatrixCols: ColumnDef<ModelScoreDetail>[] = [
    {
      key: 'rank',
      header: 'Rank',
      align: 'center',
      render: (m) => (
        <span
          style={{
            fontWeight: 800,
            fontSize: '0.9rem',
            color: m.rank === 1 ? 'var(--eco-emerald-light)' : '#FFFFFF',
          }}
        >
          #{m.rank}
        </span>
      ),
    },
    {
      key: 'model_name',
      header: 'Model Architecture',
      render: (m) => (
        <div>
          <div style={{ fontWeight: 700, color: m.rank === 1 ? '#FFFFFF' : 'var(--text-primary)' }}>
            {m.model_name}
          </div>
          <div style={{ display: 'flex', gap: '6px', marginTop: '2px' }}>
            {m.is_pareto_optimal && (
              <span className="pill pill-green" style={{ fontSize: '0.62rem', padding: '0 4px' }}>
                Pareto Optimal
              </span>
            )}
            {m.rank === 1 && (
              <span className="pill pill-sky" style={{ fontSize: '0.62rem', padding: '0 4px' }}>
                Top Pick
              </span>
            )}
          </div>
        </div>
      ),
    },
    {
      key: 'composite_score',
      header: 'EcoML Score',
      align: 'right',
      render: (m) => (
        <span
          style={{
            fontSize: '1.05rem',
            fontWeight: 800,
            color: m.rank === 1 ? 'var(--eco-emerald-light)' : '#FFFFFF',
          }}
        >
          {m.composite_score.toFixed(2)}{' '}
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>/ 100</span>
        </span>
      ),
    },
    {
      key: 'raw_perf',
      header: isClassification ? 'Weighted F1' : 'R² Score',
      align: 'right',
      render: (m) => {
        if (m.status !== 'SUCCESS') {
          return <span style={{ color: 'var(--accent-rose)', fontSize: '0.78rem' }}>Failed</span>;
        }
        const val = isClassification
          ? (m.raw_metrics.f1 ?? m.raw_metrics.performance)
          : (m.raw_metrics.r2 ?? m.raw_metrics.performance);
        if (val === undefined || val === null) return '—';
        return isClassification ? `${(val * 100).toFixed(2)}%` : val.toFixed(4);
      },
    },
    ...(isClassification
      ? [
          {
            key: 'accuracy',
            header: 'Accuracy',
            align: 'right' as const,
            render: (m: ModelScoreDetail) => {
              if (m.status !== 'SUCCESS') return '—';
              const val = m.raw_metrics.accuracy;
              return val !== undefined && val !== null ? `${(val * 100).toFixed(2)}%` : '—';
            },
          },
        ]
      : [
          {
            key: 'rmse',
            header: 'RMSE',
            align: 'right' as const,
            render: (m: ModelScoreDetail) => {
              if (m.status !== 'SUCCESS') return '—';
              const val = m.raw_metrics.rmse;
              return val !== undefined && val !== null ? val.toFixed(2) : '—';
            },
          },
        ]),
    {
      key: 's_performance',
      header: 'Norm Perf (S_P)',
      align: 'right',
      render: (m) => {
        const val = m.normalized_scores.performance ?? m.normalized_scores.s_performance;
        return val !== undefined && val !== null ? val.toFixed(3) : '—';
      },
    },
    {
      key: 's_train_time',
      header: 'Norm Speed (S_T)',
      align: 'right',
      render: (m) => {
        const val = m.normalized_scores.train_time ?? m.normalized_scores.s_train_time;
        return val !== undefined && val !== null ? val.toFixed(3) : '—';
      },
    },
    {
      key: 's_inference_latency',
      header: 'Norm Latency (S_L)',
      align: 'right',
      render: (m) => {
        const val = m.normalized_scores.inference_latency ?? m.normalized_scores.s_inference_latency;
        return val !== undefined && val !== null ? val.toFixed(3) : '—';
      },
    },
    {
      key: 's_ram',
      header: 'Norm RAM (S_M)',
      align: 'right',
      render: (m) => {
        const val = m.normalized_scores.ram ?? m.normalized_scores.s_ram;
        return val !== undefined && val !== null ? val.toFixed(3) : '—';
      },
    },
    {
      key: 'carbon_status',
      header: 'Carbon Telemetry',
      align: 'center',
      render: (m) => {
        const carbonScore = m.normalized_scores.carbon ?? m.normalized_scores.s_carbon;
        const hasCarbonWeight = (m.effective_weights.carbon ?? 0) > 0;
        if (hasCarbonWeight && carbonScore !== undefined) {
          return (
            <span className="pill pill-green" style={{ fontSize: '0.65rem', padding: '1px 6px' }}>
              Active ({carbonScore.toFixed(2)})
            </span>
          );
        }
        return (
          <span
            className="pill pill-muted"
            style={{ fontSize: '0.65rem', padding: '1px 6px' }}
            title="Carbon telemetry was unavailable or inactive; its weight was dynamically redistributed"
          >
            Redistributed
          </span>
        );
      },
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Weight Profile Selection Card */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem' }}>Multi-Objective Decision Profiles</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Select a specialized operational priority or tune custom dimension weights
            </p>
          </div>
          <span className="pill pill-green">
            Active: {recommendationReport?.profile_name || 'Balanced'}
          </span>
        </div>

        {errorMsg && (
          <div
            style={{
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid var(--accent-rose)',
              borderRadius: 'var(--radius-sm)',
              color: '#FDA4AF',
              fontSize: '0.82rem',
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
            }}
          >
            <span>{errorMsg}</span>
            <button
              className="btn btn-secondary"
              onClick={() => calculateReport(selectedProfileId)}
              style={{ fontSize: '0.72rem', padding: '0.25rem 0.6rem', flexShrink: 0 }}
            >
              Retry
            </button>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '10px' }}>
          {presets.map((preset) => {
            const isSelected = selectedProfileId === preset.profile_id;
            return (
              <div
                key={preset.profile_id}
                onClick={() => handleProfileChange(preset.profile_id)}
                style={{
                  padding: '0.9rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  border: isSelected ? '1px solid var(--eco-emerald)' : '1px solid var(--border-subtle)',
                  backgroundColor: isSelected ? 'rgba(16, 185, 129, 0.08)' : 'var(--bg-surface-elevated)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
              >
                <div style={{ fontWeight: 700, fontSize: '0.9rem', color: isSelected ? 'var(--eco-emerald-light)' : '#FFFFFF' }}>
                  {preset.name}
                </div>
                <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.4' }}>
                  {preset.description}
                </p>
              </div>
            );
          })}

          {/* Custom Profile Option */}
          <div
            onClick={() => handleProfileChange('custom')}
            style={{
              padding: '0.9rem 1rem',
              borderRadius: 'var(--radius-sm)',
              border: isCustom ? '1px solid var(--eco-emerald)' : '1px solid var(--border-subtle)',
              backgroundColor: isCustom ? 'rgba(16, 185, 129, 0.08)' : 'var(--bg-surface-elevated)',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: isCustom ? 'var(--eco-emerald-light)' : '#FFFFFF' }}>
              Custom Weights
            </div>
            <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.4' }}>
              Adjust individual criteria sliders manually for custom trade-off targets
            </p>
          </div>
        </div>

        {/* Custom Sliders Drawer */}
        {isCustom && (
          <div
            style={{
              marginTop: '1.5rem',
              padding: '1.25rem',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: '#0A0F1D',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ fontSize: '0.88rem', fontWeight: 700, marginBottom: '1rem', color: '#FFFFFF' }}>
              Tune Custom Criterion Weights
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1.25rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span>Performance</span>
                  <span style={{ color: 'var(--eco-emerald-light)' }}>
                    {customWeights.w_performance.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.8"
                  step="0.05"
                  value={customWeights.w_performance}
                  onChange={(e) => handleCustomSliderChange('w_performance', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span>Train Speed</span>
                  <span style={{ color: 'var(--eco-emerald-light)' }}>
                    {customWeights.w_train_time.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={customWeights.w_train_time}
                  onChange={(e) => handleCustomSliderChange('w_train_time', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span>Latency</span>
                  <span style={{ color: 'var(--eco-emerald-light)' }}>
                    {customWeights.w_inference_latency.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={customWeights.w_inference_latency}
                  onChange={(e) =>
                    handleCustomSliderChange('w_inference_latency', parseFloat(e.target.value))
                  }
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span>RAM Memory</span>
                  <span style={{ color: 'var(--eco-emerald-light)' }}>
                    {customWeights.w_ram.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={customWeights.w_ram}
                  onChange={(e) => handleCustomSliderChange('w_ram', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                  <span>Carbon / Energy</span>
                  <span style={{ color: 'var(--eco-emerald-light)' }}>
                    {customWeights.w_carbon.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={customWeights.w_carbon}
                  onChange={(e) => handleCustomSliderChange('w_carbon', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1rem' }}>
              <button className="btn btn-primary" onClick={handleApplyCustomWeights} disabled={loading}>
                <span>Normalize &amp; Apply Weights</span>
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Zero Predictive Utility Warning Alert */}
      {isZeroPerf && (
        <div
          className="card"
          style={{
            backgroundColor: 'rgba(244, 63, 94, 0.08)',
            border: '1px solid var(--accent-rose)',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
          }}
        >
          <AlertCircle size={22} color="var(--accent-rose)" style={{ flexShrink: 0 }} />
          <div>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.92rem' }}>
              Zero Predictive Utility Warning
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              All evaluated models achieved 0.00% predictive accuracy or utility. This candidate cannot be recommended for production deployment. Please verify that the correct discrete target column and task formulation (Classification vs. Regression) were configured in Dataset Studio.
            </p>
          </div>
        </div>
      )}

      {/* Top Recommended Model Hero Card */}
      {topModel && (
        <div
          className="hero-card"
          style={{
            borderLeftColor: isZeroPerf ? 'var(--accent-rose)' : 'var(--eco-emerald)',
          }}
        >
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                {isZeroPerf ? (
                  <span className="pill pill-rose">
                    <AlertCircle size={12} />
                    <span>Ineffective Predictor (0.00% Utility)</span>
                  </span>
                ) : (
                  <span className="pill pill-green">
                    <Award size={12} />
                    <span>Rank #1 Recommended Model</span>
                  </span>
                )}
                {topModel.is_pareto_optimal && (
                  <span className="pill pill-sky">Pareto Optimal Frontier</span>
                )}
                <span className="pill pill-muted">
                  Profile: {recommendationReport?.profile_name}
                </span>
              </div>

              <h2 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
                {topModel.model_name}
              </h2>
              {!isClassification && topModel.status === 'SUCCESS' && (
                <div style={{ display: 'flex', gap: '14px', marginTop: '6px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  {topModel.raw_metrics.rmse !== undefined && topModel.raw_metrics.rmse !== null && (
                    <span>RMSE: <strong style={{ color: '#FFFFFF' }}>{topModel.raw_metrics.rmse.toFixed(2)}</strong></span>
                  )}
                  {topModel.raw_metrics.mae !== undefined && topModel.raw_metrics.mae !== null && (
                    <span>MAE: <strong style={{ color: '#FFFFFF' }}>{topModel.raw_metrics.mae.toFixed(2)}</strong></span>
                  )}
                </div>
              )}
              {isClassification && topModel.status === 'SUCCESS' && topModel.raw_metrics.accuracy !== undefined && (
                <div style={{ display: 'flex', gap: '14px', marginTop: '6px', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  <span>Accuracy: <strong style={{ color: '#FFFFFF' }}>{(topModel.raw_metrics.accuracy * 100).toFixed(2)}%</strong></span>
                  {topModel.raw_metrics.precision !== undefined && (
                    <span>Precision: <strong style={{ color: '#FFFFFF' }}>{(topModel.raw_metrics.precision * 100).toFixed(2)}%</strong></span>
                  )}
                  {topModel.raw_metrics.recall !== undefined && (
                    <span>Recall: <strong style={{ color: '#FFFFFF' }}>{(topModel.raw_metrics.recall * 100).toFixed(2)}%</strong></span>
                  )}
                </div>
              )}
            </div>

            {/* Dual Score Display */}
            <div style={{ display: 'flex', gap: '16px' }}>
              <div
                style={{
                  backgroundColor: 'var(--bg-surface-elevated)',
                  border: '1px solid var(--border-emerald)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.85rem 1.4rem',
                  textAlign: 'right',
                }}
              >
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 600 }}>
                  EcoML Decision Score
                </div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--eco-emerald-light)', lineHeight: '1.2' }}>
                  {topModel.composite_score.toFixed(2)}
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>/100</span>
                </div>
              </div>

              <div
                style={{
                  backgroundColor: 'var(--bg-surface-elevated)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.85rem 1.4rem',
                  textAlign: 'right',
                }}
              >
                <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 600 }}>
                  {isClassification ? 'Weighted F1' : 'R² Score'}
                </div>
                <div style={{ fontSize: '1.85rem', fontWeight: 800, color: '#FFFFFF', lineHeight: '1.2' }}>
                  {(() => {
                    if (topModel.status !== 'SUCCESS') {
                      return <span style={{ color: 'var(--accent-rose)', fontSize: '1.25rem' }}>Failed</span>;
                    }
                    const perfVal = isClassification
                      ? (topModel.raw_metrics.f1 ?? topModel.raw_metrics.performance)
                      : (topModel.raw_metrics.r2 ?? topModel.raw_metrics.performance);
                    if (perfVal === undefined || perfVal === null) {
                      return <span style={{ color: 'var(--text-muted)', fontSize: '1.25rem' }}>N/A</span>;
                    }
                    return isClassification ? `${(perfVal * 100).toFixed(2)}%` : perfVal.toFixed(4);
                  })()}
                </div>
              </div>
            </div>
          </div>

          {/* Strengths and Notes */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1.25rem' }}>
            <div
              style={{
                backgroundColor: 'rgba(16, 185, 129, 0.06)',
                border: '1px solid var(--border-emerald)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.9rem 1.1rem',
              }}
            >
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--eco-emerald-light)', marginBottom: '6px' }}>
                Observed Strengths
              </div>
              <ul style={{ paddingLeft: '1.2rem', fontSize: '0.8rem', color: '#E2E8F0', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {topModel.strengths.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ul>
            </div>

            <div
              style={{
                backgroundColor: 'rgba(56, 189, 248, 0.06)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.9rem 1.1rem',
              }}
            >
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--accent-sky)', marginBottom: '6px' }}>
                Trade-off Notes
              </div>
              <ul style={{ paddingLeft: '1.2rem', fontSize: '0.8rem', color: '#E2E8F0', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {topModel.tradeoff_notes.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Visualizations Section: Pareto Frontier & Radar Spider Chart */}
      {recommendationReport && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
          {/* Pareto Frontier Scatter Plot */}
          <div className="card">
            <div style={{ marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.1rem' }}>2D Pareto Efficiency Frontier</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Predictive Utility vs. System Efficiency Index. Emerald diamonds mark non-dominated models.
              </p>
            </div>
            <ParetoScatterPlot
              models={recommendationReport.ranked_models}
              taskType={benchmarkResults.task_type}
            />
          </div>

          {/* Multi-Criteria Radar Spider Chart */}
          <div className="card">
            <div style={{ marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.1rem' }}>Multi-Criteria Radar Comparison</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Normalized scores (0.0 – 1.0) across all 5 evaluation dimensions for top contenders
              </p>
            </div>
            <RadarChart models={recommendationReport.ranked_models} maxModels={3} />
          </div>
        </div>
      )}

      {/* Multi-Criteria Decision Matrix Table */}
      {recommendationReport && (
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.15rem' }}>Multi-Criteria Decision Matrix</h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Complete ranking with normalized scores and composite EcoML index
              </p>
            </div>
            <span className="pill pill-muted">
              {recommendationReport.ranked_models.length} Models Ranked
            </span>
          </div>

          <DataTable
            columns={decisionMatrixCols}
            data={recommendationReport.ranked_models}
            pageSize={10}
            keyExtractor={(m) => m.model_id}
          />
        </div>
      )}

      {/* Weight Redistribution & Synthesis Callout */}
      {recommendationReport && (
        <div
          className="card"
          style={{
            backgroundColor: 'var(--bg-surface-elevated)',
            border: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
            <Info size={16} color="var(--accent-sky)" />
            <span style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.92rem' }}>
              Scoring Integrity &amp; Synthesis Note
            </span>
          </div>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.5', marginBottom: '6px' }}>
            <FormattedText text={recommendationReport.tradeoff_synthesis} />
          </p>
          {recommendationReport.redistribution_note && (
            <p style={{ fontSize: '0.78rem', color: 'var(--accent-amber)', fontStyle: 'italic' }}>
              Notice: <FormattedText text={recommendationReport.redistribution_note} />
            </p>
          )}
        </div>
      )}
    </div>
  );
};
