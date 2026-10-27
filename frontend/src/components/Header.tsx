import React from 'react';
import {
  Database,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Layers,
  RotateCcw,
  Check,
} from 'lucide-react';
import { PreprocessingResponse, BenchmarkResults, RecommendationReport } from '../types';

interface HeaderProps {
  currentTab: string;
  datasetName: string | null;
  prepData: PreprocessingResponse | null;
  benchmarkResults: BenchmarkResults | null;
  recommendationReport: RecommendationReport | null;
  backendOnline: boolean;
  onRefresh?: () => void;
  onNavigate?: (tab: string) => void;
  onResetSession?: () => void;
}

const TAB_TITLES: Record<string, { title: string; subtitle: string }> = {
  overview: {
    title: 'Platform Overview & Executive Cockpit',
    subtitle: 'Green AI Model Selection Framework for Carbon, Energy, and Compute Efficiency',
  },
  datasets: {
    title: 'Dataset Studio & Pipeline Preprocessing',
    subtitle: 'Upload, inspect data schemas, validate target features, and partition test splits',
  },
  benchmarks: {
    title: 'Model Benchmark Lab & Telemetry Profiler',
    subtitle: 'Run scikit-learn models under live psutil memory and CodeCarbon emission tracking',
  },
  intelligence: {
    title: 'Green Intelligence & Multi-Criteria Recommender',
    subtitle: 'Pareto-optimal efficiency frontier analysis and weighted multi-objective scoring',
  },
  quantum: {
    title: 'Quantum Optimization Lab & QAOA Solver',
    subtitle: 'Ising Hamiltonian mapping, variational parameter optimization, and Born-rule sampling',
  },
  methodology: {
    title: 'Academic Methodology & Engineering Architecture',
    subtitle: 'Formula specifications, measurement telemetry caveats, and implementation roadmap',
  },
};

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  datasetName,
  prepData,
  benchmarkResults,
  recommendationReport,
  backendOnline,
  onRefresh,
  onNavigate,
  onResetSession,
}) => {
  const meta = TAB_TITLES[currentTab] || {
    title: 'EcoML Platform',
    subtitle: 'Carbon & Energy-Aware Machine Learning Selector',
  };

  const steps = [
    { id: 'step1', num: 1, label: 'Dataset', done: !!datasetName, tab: 'datasets' },
    { id: 'step2', num: 2, label: 'Partition', done: !!prepData, tab: 'datasets' },
    { id: 'step3', num: 3, label: 'Benchmark', done: !!benchmarkResults, tab: 'benchmarks' },
    { id: 'step4', num: 4, label: 'Intelligence', done: !!recommendationReport, tab: 'intelligence' },
  ];

  return (
    <header
      style={{
        display: 'flex',
        flexDirection: 'column',
        borderBottom: '1px solid var(--border-subtle)',
        backgroundColor: 'var(--bg-canvas)',
        position: 'sticky',
        top: 0,
        zIndex: 20,
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* Top Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '1.1rem 2rem',
        }}
      >
        <div>
          <h1 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
            {meta.title}
          </h1>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
            {meta.subtitle}
          </p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Active Dataset Pill */}
          {datasetName ? (
            <div className="pill pill-sky" title={`Active dataset: ${datasetName}`}>
              <Database size={12} />
              <span>{datasetName}</span>
            </div>
          ) : (
            <div className="pill pill-muted" title="No dataset currently selected">
              <Database size={12} />
              <span>No Dataset</span>
            </div>
          )}

          {/* Preprocessing status */}
          {prepData ? (
            <div
              className="pill pill-green"
              title={`Preprocessed: ${prepData.train_samples} train / ${prepData.test_samples} test`}
            >
              <CheckCircle2 size={12} />
              <span>Partitioned ({prepData.task_type})</span>
            </div>
          ) : (
            <div className="pill pill-muted" title="Preprocessing pending">
              <span>Unprocessed</span>
            </div>
          )}

          {/* Benchmark status */}
          {benchmarkResults && (
            <div
              className="pill pill-green"
              title={`Benchmark completed with ${benchmarkResults.models_evaluated.length} models`}
            >
              <Layers size={12} />
              <span>{benchmarkResults.models_evaluated.length} Models</span>
            </div>
          )}

          {/* Backend health status */}
          <div
            className={`pill ${backendOnline ? 'pill-green' : 'pill-amber'}`}
            title={backendOnline ? 'FastAPI Backend Online' : 'Connecting to FastAPI Backend...'}
          >
            {backendOnline ? (
              <>
                <span
                  style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    backgroundColor: 'var(--eco-emerald-light)',
                    boxShadow: '0 0 6px var(--eco-emerald)',
                  }}
                />
                <span>FastAPI Ready</span>
              </>
            ) : (
              <>
                <AlertCircle size={12} />
                <span>Backend Offline</span>
              </>
            )}
          </div>

          {/* Reset session button */}
          {onResetSession && (
            <button
              className="btn btn-ghost"
              onClick={() => {
                if (window.confirm('Reset EcoML session state? This will clear active dataset partitions and benchmark results.')) {
                  onResetSession();
                }
              }}
              title="Reset Session Data"
              style={{ padding: '0.4rem 0.6rem', fontSize: '0.78rem' }}
            >
              <RotateCcw size={13} />
            </button>
          )}

          {/* Refresh health button */}
          {onRefresh && (
            <button
              className="btn btn-ghost"
              onClick={onRefresh}
              title="Refresh connection status"
              style={{ padding: '0.4rem 0.6rem', fontSize: '0.78rem' }}
            >
              <RefreshCw size={13} />
            </button>
          )}
        </div>
      </div>

      {/* Workflow Stepper Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '0.45rem 2rem',
          backgroundColor: '#090E1C',
          borderTop: '1px solid rgba(148, 163, 184, 0.08)',
          fontSize: '0.74rem',
        }}
      >
        <span style={{ color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Workflow Pipeline:
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {steps.map((st, i) => {
            const isClickable = !!onNavigate;
            return (
              <React.Fragment key={st.id}>
                {i > 0 && <span style={{ color: 'var(--border-light)' }}>→</span>}
                <div
                  onClick={() => isClickable && onNavigate(st.tab)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '5px',
                    cursor: isClickable ? 'pointer' : 'default',
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: st.done ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255, 255, 255, 0.03)',
                    border: `1px solid ${st.done ? 'var(--border-emerald)' : 'var(--border-subtle)'}`,
                    color: st.done ? 'var(--eco-emerald-light)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <span
                    style={{
                      width: '14px',
                      height: '14px',
                      borderRadius: '50%',
                      backgroundColor: st.done ? 'var(--eco-emerald)' : '#1E293B',
                      color: st.done ? '#080D1A' : 'var(--text-secondary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '0.62rem',
                      fontWeight: 800,
                    }}
                  >
                    {st.done ? <Check size={9} strokeWidth={3} /> : st.num}
                  </span>
                  <span style={{ fontWeight: 600 }}>{st.label}</span>
                </div>
              </React.Fragment>
            );
          })}
        </div>
      </div>
    </header>
  );
};
