import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Play,
  RotateCw,
  AlertCircle,
  ArrowRight,
  Zap,
  Activity,
  HardDrive,
  Layers,
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { DataTable, ColumnDef } from '../components/DataTable';
import { FormattedText } from '../components/FormattedText';
import {
  PreprocessingResponse,
  BenchmarkResults,
  SingleModelResult,
  ModelInfo,
  RecommendationReport,
} from '../types';
import { api } from '../services/api';

interface BenchmarkLabPageProps {
  onNavigate: (tab: string) => void;
  selectedDatasetId: string | null;
  datasetName: string | null;
  prepData: PreprocessingResponse | null;
  benchmarkResults: BenchmarkResults | null;
  setBenchmarkResults: (results: BenchmarkResults | null) => void;
  activeBenchmarkId: string | null;
  setActiveBenchmarkId: (id: string | null) => void;
  setRecommendationReport?: (report: RecommendationReport | null) => void;
}

export const BenchmarkLabPage: React.FC<BenchmarkLabPageProps> = ({
  onNavigate,
  selectedDatasetId,
  datasetName,
  prepData,
  benchmarkResults,
  setBenchmarkResults,
  activeBenchmarkId: _activeBenchmarkId,
  setActiveBenchmarkId,
  setRecommendationReport,
}) => {
  const [supportedModels, setSupportedModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [enableCarbon, setEnableCarbon] = useState<boolean>(false);
  const [benchmarkSeed, setBenchmarkSeed] = useState<number>(42);

  // Execution state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [progressPct, setProgressPct] = useState<number>(0);
  const [progressMsg, setProgressMsg] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const pollIntervalRef = useRef<any>(null);

  // Load supported models when prepData changes
  useEffect(() => {
    if (!prepData) return;
    api
      .getSupportedModels(prepData.task_type)
      .then((models) => {
        setSupportedModels(models);
        // Default select all supported models
        setSelectedModels(models.map((m) => m.model_id));
      })
      .catch((err) => {
        console.error('Failed to load supported models:', err);
      });
  }, [prepData]);

  // Clean up polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Check active benchmark status on mount or if activeBenchmarkId changed
  useEffect(() => {
    if (!_activeBenchmarkId || benchmarkResults) return;

    let isMounted = true;
    api
      .getBenchmarkStatus(_activeBenchmarkId)
      .then((status) => {
        if (!isMounted) return;
        if (status.status === 'COMPLETED') {
          api.getBenchmarkResults(_activeBenchmarkId).then(setBenchmarkResults);
        } else if (status.status === 'RUNNING' || status.status === 'QUEUED') {
          setIsRunning(true);
          setProgressPct(status.progress_percent);
          setProgressMsg(
            `Evaluating model ${status.completed_models} of ${status.total_models} (${status.status})...`
          );

          if (!pollIntervalRef.current) {
            pollIntervalRef.current = setInterval(async () => {
              try {
                const cur = await api.getBenchmarkStatus(_activeBenchmarkId);
                setProgressPct(cur.progress_percent);
                setProgressMsg(
                  `Evaluating model ${cur.completed_models} of ${cur.total_models} (${cur.status})...`
                );

                if (cur.status === 'COMPLETED') {
                  clearInterval(pollIntervalRef.current);
                  pollIntervalRef.current = null;
                  setIsRunning(false);
                  const res = await api.getBenchmarkResults(_activeBenchmarkId);
                  setBenchmarkResults(res);
                } else if (cur.status === 'FAILED') {
                  clearInterval(pollIntervalRef.current);
                  pollIntervalRef.current = null;
                  setIsRunning(false);
                  setErrorMessage(cur.error_message || 'Benchmark execution failed.');
                }
              } catch (err: any) {
                clearInterval(pollIntervalRef.current);
                pollIntervalRef.current = null;
                setIsRunning(false);
                setErrorMessage(err.message || 'Error checking benchmark status.');
              }
            }, 500);
          }
        } else if (status.status === 'FAILED') {
          setErrorMessage(status.error_message || 'Benchmark execution failed.');
        }
      })
      .catch(() => {
        setActiveBenchmarkId(null);
      });

    return () => {
      isMounted = false;
    };
  }, [_activeBenchmarkId, benchmarkResults]);

  const handleToggleModel = (modelId: string) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter((id) => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const handleSelectAll = () => {
    if (selectedModels.length === supportedModels.length) {
      setSelectedModels([]);
    } else {
      setSelectedModels(supportedModels.map((m) => m.model_id));
    }
  };

  const handleStartBenchmark = async () => {
    if (!selectedDatasetId || selectedModels.length === 0) return;

    // Invalidate stale previous benchmark results and recommendations immediately upon starting a new run
    setBenchmarkResults(null);
    if (setRecommendationReport) {
      setRecommendationReport(null);
    }
    setActiveBenchmarkId(null);

    setIsRunning(true);
    setProgressPct(0);
    setProgressMsg('Submitting benchmark job to worker thread...');
    setErrorMessage(null);

    try {
      const res = await api.submitBenchmark({
        dataset_id: selectedDatasetId,
        selected_models: selectedModels,
        enable_carbon_tracking: enableCarbon,
        random_state: benchmarkSeed,
      });

      const bId = res.benchmark_id;
      setActiveBenchmarkId(bId);

      // Start polling
      pollIntervalRef.current = setInterval(async () => {
        try {
          const status = await api.getBenchmarkStatus(bId);
          setProgressPct(status.progress_percent);
          setProgressMsg(
            `Evaluating model ${status.completed_models} of ${status.total_models} (${status.status})...`
          );

          if (status.status === 'COMPLETED') {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
            setIsRunning(false);
            setProgressMsg('Benchmark completed successfully!');

            // Fetch results
            const results = await api.getBenchmarkResults(bId);
            setBenchmarkResults(results);
          } else if (status.status === 'FAILED') {
            clearInterval(pollIntervalRef.current);
            pollIntervalRef.current = null;
            setIsRunning(false);
            setErrorMessage(status.error_message || 'Benchmark execution failed.');
          }
        } catch (err: any) {
          clearInterval(pollIntervalRef.current);
          pollIntervalRef.current = null;
          setIsRunning(false);
          setErrorMessage(err.message || 'Error checking benchmark status.');
        }
      }, 500);
    } catch (err: any) {
      setIsRunning(false);
      setErrorMessage(err.message || 'Failed to submit benchmark.');
    }
  };

  // If no preprocessed data, show prerequisite guard
  if (!prepData) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
        <div
          style={{
            width: '48px',
            height: '48px',
            borderRadius: '50%',
            backgroundColor: 'rgba(251, 191, 36, 0.12)',
            border: '1px solid rgba(251, 191, 36, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1.25rem auto',
            color: 'var(--accent-amber)',
          }}
        >
          <AlertCircle size={24} />
        </div>
        <h3 style={{ fontSize: '1.25rem', marginBottom: '0.5rem' }}>
          Preprocessing Required Before Benchmarking
        </h3>
        <p style={{ maxWidth: '540px', margin: '0 auto 1.5rem auto', fontSize: '0.88rem', lineHeight: '1.6' }}>
          No preprocessed data partition is currently active. Please complete Step 2 in the Dataset Studio tab
          to select your target column and partition train/test sets before executing model benchmarks.
        </p>
        <button className="btn btn-primary" onClick={() => onNavigate('datasets')}>
          <span>Open Dataset Studio</span>
          <ArrowRight size={16} />
        </button>
      </div>
    );
  }

  // Define results table columns
  const isClassification = prepData.task_type === 'classification';

  const CarbonTelemetryBadge: React.FC<{ m: SingleModelResult }> = ({ m }) => {
    const telem = m.telemetry;
    const val = telem?.carbon_emissions_g;
    const statusStr = (telem?.energy_status || '').toLowerCase();

    if (m.status !== 'SUCCESS') {
      return <span style={{ color: 'var(--text-muted)' }}>—</span>;
    }

    if (statusStr.includes('measured') && val !== undefined && val !== null && val > 0) {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
          <span className="pill pill-green" style={{ fontSize: '0.64rem', padding: '1px 6px' }}>
            Measured (RAPL)
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#FFFFFF' }}>
            {val.toExponential(3)} g
          </span>
        </div>
      );
    }

    if (val !== undefined && val !== null && val > 0) {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
          <span className="pill pill-sky" style={{ fontSize: '0.64rem', padding: '1px 6px' }}>
            Estimated (CodeCarbon)
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: '#FFFFFF' }}>
            {val.toExponential(3)} g
          </span>
        </div>
      );
    }

    if (statusStr.includes('disabled')) {
      return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
          <span
            className="pill pill-muted"
            style={{ fontSize: '0.64rem', padding: '1px 6px' }}
            title="Sensor tracking was not enabled for this benchmark run"
          >
            Disabled
          </span>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Opt-in off</span>
        </div>
      );
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '2px' }}>
        <span
          className="pill pill-amber"
          style={{ fontSize: '0.64rem', padding: '1px 6px' }}
          title={telem?.energy_status || 'Execution time below hardware detection threshold'}
        >
          Unavailable
        </span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Below threshold</span>
      </div>
    );
  };

  const benchmarkColumns: ColumnDef<SingleModelResult>[] = [
    {
      key: 'model_name',
      header: 'Model Architecture',
      render: (m) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ fontWeight: 700, color: '#FFFFFF' }}>{m.model_name}</span>
            {m.status !== 'SUCCESS' && (
              <span className="pill pill-rose" style={{ fontSize: '0.62rem', padding: '1px 5px' }}>
                Failed
              </span>
            )}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{m.model_id}</div>
        </div>
      ),
    },
    ...(isClassification
      ? [
          {
            key: 'accuracy',
            header: 'Accuracy',
            align: 'right' as const,
            render: (m: SingleModelResult) =>
              m.status !== 'SUCCESS' ? (
                <span style={{ color: 'var(--accent-rose)', fontSize: '0.76rem' }}>Failed</span>
              ) : (
                <span style={{ fontWeight: 600, color: 'var(--eco-emerald-light)' }}>
                  {m.accuracy !== undefined && m.accuracy !== null ? `${(m.accuracy * 100).toFixed(2)}%` : '—'}
                </span>
              ),
          },
          {
            key: 'f1',
            header: 'Weighted F1',
            align: 'right' as const,
            render: (m: SingleModelResult) =>
              m.status !== 'SUCCESS' ? (
                <span style={{ color: 'var(--text-muted)' }}>—</span>
              ) : m.f1 !== undefined && m.f1 !== null ? (
                (m.f1 * 100).toFixed(2) + '%'
              ) : (
                '—'
              ),
          },
        ]
      : [
          {
            key: 'r2',
            header: 'R² Score',
            align: 'right' as const,
            render: (m: SingleModelResult) =>
              m.status !== 'SUCCESS' ? (
                <span style={{ color: 'var(--accent-rose)', fontSize: '0.76rem' }}>Failed</span>
              ) : (
                <span style={{ fontWeight: 600, color: 'var(--eco-emerald-light)' }}>
                  {m.r2 !== undefined && m.r2 !== null ? m.r2.toFixed(4) : '—'}
                </span>
              ),
          },
          {
            key: 'rmse',
            header: 'RMSE',
            align: 'right' as const,
            render: (m: SingleModelResult) =>
              m.status !== 'SUCCESS' ? (
                <span style={{ color: 'var(--text-muted)' }}>—</span>
              ) : m.rmse !== undefined && m.rmse !== null ? (
                m.rmse.toFixed(4)
              ) : (
                '—'
              ),
          },
        ]),
    {
      key: 'training_time_sec',
      header: 'Train Time',
      align: 'right',
      render: (m) =>
        m.status !== 'SUCCESS'
          ? '—'
          : m.training_time_sec !== undefined && m.training_time_sec !== null
          ? `${(m.training_time_sec * 1000).toFixed(1)} ms`
          : '—',
    },
    {
      key: 'latency_per_sample_ms',
      header: 'Inference Latency (ms/sample)',
      align: 'right',
      render: (m) =>
        m.status !== 'SUCCESS'
          ? '—'
          : m.latency_per_sample_ms !== undefined && m.latency_per_sample_ms !== null
          ? `${m.latency_per_sample_ms.toFixed(4)} ms`
          : '—',
    },
    {
      key: 'peak_ram_mb',
      header: 'Peak RAM',
      align: 'right',
      render: (m) =>
        m.status !== 'SUCCESS'
          ? '—'
          : m.telemetry?.peak_ram_mb !== undefined && m.telemetry?.peak_ram_mb !== null
          ? `${m.telemetry.peak_ram_mb.toFixed(1)} MB`
          : '—',
    },
    {
      key: 'ram_delta_mb',
      header: 'RAM Delta (Δ)',
      align: 'right',
      render: (m) =>
        m.status !== 'SUCCESS'
          ? '—'
          : m.telemetry?.ram_delta_mb !== undefined && m.telemetry?.ram_delta_mb !== null
          ? `${m.telemetry.ram_delta_mb.toFixed(2)} MB`
          : '—',
    },
    {
      key: 'carbon_emissions_g',
      header: 'Carbon Telemetry',
      align: 'right',
      render: (m) => <CarbonTelemetryBadge m={m} />,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Active Configuration Banner */}
      <div
        className="card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          backgroundColor: 'var(--bg-surface-elevated)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            style={{
              width: '40px',
              height: '40px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--eco-emerald)',
            }}
          >
            <Cpu size={22} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontWeight: 700, fontSize: '1.05rem', color: '#FFFFFF' }}>
                {datasetName}
              </span>
              <span className="pill pill-green" style={{ textTransform: 'uppercase' }}>
                {prepData.task_type}
              </span>
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Target: <strong style={{ color: '#FFFFFF' }}>{prepData.target_column}</strong> •{' '}
              {prepData.train_samples} Train • {prepData.test_samples} Test • {prepData.feature_count} Features
            </div>
          </div>
        </div>

        <button
          className="btn btn-secondary"
          onClick={() => onNavigate('datasets')}
          style={{ fontSize: '0.8rem', padding: '0.45rem 0.85rem' }}
        >
          Modify Dataset Split
        </button>
      </div>

      {/* Model Selection and Execution Options */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem' }}>Select Models for Benchmark</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Choose algorithm architectures to train and evaluate under standardized telemetry conditions
            </p>
          </div>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={handleSelectAll}
            style={{ fontSize: '0.8rem' }}
          >
            {selectedModels.length === supportedModels.length ? 'Deselect All' : 'Select All'}
          </button>
        </div>

        {/* Model Card Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px', marginBottom: '1.5rem' }}>
          {supportedModels.map((m) => {
            const isChecked = selectedModels.includes(m.model_id);
            return (
              <div
                key={m.model_id}
                onClick={() => handleToggleModel(m.model_id)}
                style={{
                  padding: '0.85rem 1rem',
                  borderRadius: 'var(--radius-sm)',
                  border: isChecked ? '1px solid var(--eco-emerald)' : '1px solid var(--border-subtle)',
                  backgroundColor: isChecked ? 'rgba(16, 185, 129, 0.08)' : 'var(--bg-surface-elevated)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  transition: 'all 0.15s ease',
                }}
              >
                <input
                  type="checkbox"
                  checked={isChecked}
                  onChange={() => {}} // Handled by parent div
                  style={{ accentColor: 'var(--eco-emerald)', width: '16px', height: '16px' }}
                />
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.88rem', color: isChecked ? '#FFFFFF' : 'var(--text-secondary)' }}>
                    {m.model_name}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{m.model_id}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Benchmark Options */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '1rem',
            paddingTop: '1.25rem',
            borderTop: '1px solid var(--border-subtle)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <input
                type="checkbox"
                id="carbon_track"
                checked={enableCarbon}
                onChange={(e) => setEnableCarbon(e.target.checked)}
                style={{ accentColor: 'var(--eco-emerald)', width: '16px', height: '16px' }}
              />
              <label htmlFor="carbon_track" style={{ fontSize: '0.84rem', color: 'var(--text-primary)', cursor: 'pointer' }}>
                Enable CodeCarbon Emission Tracking (RAPL / Architecture Estimate)
              </label>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Seed:</label>
              <input
                type="number"
                className="form-input"
                value={benchmarkSeed}
                onChange={(e) => setBenchmarkSeed(parseInt(e.target.value, 10) || 42)}
                style={{ width: '80px', padding: '0.3rem 0.5rem' }}
              />
            </div>
          </div>

          <button
            className="btn btn-primary"
            onClick={handleStartBenchmark}
            disabled={isRunning || selectedModels.length === 0}
            style={{ minWidth: '220px' }}
          >
            {isRunning ? (
              <>
                <RotateCw className="spinning" size={16} />
                <span>Running Benchmark ({progressPct}%)...</span>
              </>
            ) : (
              <>
                <Play size={16} />
                <span>Execute {selectedModels.length} Model Benchmarks</span>
              </>
            )}
          </button>
        </div>

        {/* Live Progress Bar */}
        {isRunning && (
          <div style={{ marginTop: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '6px' }}>
              <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 600 }}>{progressMsg}</span>
              <span style={{ color: '#FFFFFF', fontWeight: 700 }}>{progressPct}%</span>
            </div>
            <div
              style={{
                width: '100%',
                height: '8px',
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
                borderRadius: 'var(--radius-full)',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${progressPct}%`,
                  height: '100%',
                  backgroundColor: 'var(--eco-emerald)',
                  boxShadow: '0 0 10px var(--eco-emerald)',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        )}

        {errorMessage && (
          <div
            style={{
              marginTop: '1rem',
              padding: '0.85rem 1.15rem',
              backgroundColor: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid var(--accent-rose)',
              borderRadius: 'var(--radius-sm)',
              color: '#FDA4AF',
              fontSize: '0.82rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '12px',
            }}
          >
            <div>
              <strong>Execution Error: </strong>
              <span>{errorMessage}</span>
            </div>
            <button
              className="btn btn-secondary"
              onClick={handleStartBenchmark}
              style={{ fontSize: '0.75rem', padding: '0.35rem 0.75rem', flexShrink: 0 }}
            >
              Retry Benchmark
            </button>
          </div>
        )}
      </div>

      {/* Benchmark Results Display */}
      {benchmarkResults && (
        <>
          {/* Quick Metrics Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <StatCard
              label="Models Evaluated"
              value={benchmarkResults.models_evaluated.length}
              subtext="Under identical split"
              icon={Layers}
              badge={{ text: 'Verified', variant: 'green' }}
            />
            {benchmarkResults.tradeoff_interpretation && (
              <>
                <StatCard
                  label="Best Predictive Model"
                  value={benchmarkResults.tradeoff_interpretation.best_predictive_model}
                  subtext={benchmarkResults.tradeoff_interpretation.best_predictive_metric}
                  icon={Zap}
                  accentColor="var(--eco-emerald-light)"
                />
                <StatCard
                  label="Fastest Training"
                  value={benchmarkResults.tradeoff_interpretation.fastest_training_model}
                  subtext="Lowest wall-clock fit time"
                  icon={Activity}
                  accentColor="var(--accent-sky)"
                />
                <StatCard
                  label="Lowest Latency"
                  value={benchmarkResults.tradeoff_interpretation.lowest_latency_model}
                  subtext="Fastest per-sample inference"
                  icon={HardDrive}
                  accentColor="var(--accent-amber)"
                />
              </>
            )}
          </div>

          {/* Detailed Benchmark Results Table */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div>
                <h3 style={{ fontSize: '1.15rem' }}>Benchmark Telemetry &amp; Performance Matrix</h3>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Hardware-sampled resource utilization, runtime latencies, and predictive validation scores
                </p>
              </div>
              <span className="pill pill-green">
                Completed at {new Date(benchmarkResults.benchmark_timestamp).toLocaleTimeString()}
              </span>
            </div>

            <DataTable
              columns={benchmarkColumns}
              data={benchmarkResults.models_evaluated}
              pageSize={10}
              keyExtractor={(m) => m.model_id}
            />
          </div>

          {/* Tradeoff Observation Synthesis */}
          {benchmarkResults.tradeoff_interpretation?.tradeoff_observation && (
            <div
              className="card"
              style={{
                backgroundColor: 'rgba(56, 189, 248, 0.05)',
                border: '1px solid rgba(56, 189, 248, 0.25)',
                padding: '1.25rem 1.5rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                <Zap size={16} color="var(--accent-sky)" />
                <span style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.92rem' }}>
                  Empirical Trade-off Observation
                </span>
              </div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
                <FormattedText text={benchmarkResults.tradeoff_interpretation.tradeoff_observation} />
              </p>
            </div>
          )}

          {/* Action to proceed to Recommender */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '1.25rem 1.75rem',
              backgroundColor: 'rgba(16, 185, 129, 0.08)',
              border: '1px solid var(--border-emerald)',
              borderRadius: 'var(--radius-md)',
            }}
          >
            <div>
              <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '1rem' }}>
                Ready for Multi-Criteria Green Intelligence
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Calculate weighted EcoML scores, explore the Pareto efficiency frontier, and review radar trade-offs.
              </div>
            </div>
            <button className="btn btn-primary" onClick={() => onNavigate('intelligence')}>
              <span>Open Green Intelligence</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </>
      )}
    </div>
  );
};
