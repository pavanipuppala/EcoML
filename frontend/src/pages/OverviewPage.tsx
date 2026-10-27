import React, { useEffect, useState } from 'react';
import {
  Database,
  Cpu,
  Leaf,
  Atom,
  BookOpen,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  Sparkles,
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import {
  DatasetPreview,
  PreprocessingResponse,
  BenchmarkResults,
  RecommendationReport,
  MetadataFeatures,
} from '../types';
import { api } from '../services/api';

interface OverviewPageProps {
  onNavigate: (tab: string) => void;
  datasetName: string | null;
  datasetPreview: DatasetPreview | null;
  prepData: PreprocessingResponse | null;
  benchmarkResults: BenchmarkResults | null;
  recommendationReport: RecommendationReport | null;
}

export const OverviewPage: React.FC<OverviewPageProps> = ({
  onNavigate,
  datasetName,
  datasetPreview,
  prepData,
  benchmarkResults,
  recommendationReport,
}) => {
  const [meta, setMeta] = useState<MetadataFeatures | null>(null);

  useEffect(() => {
    api.getMetaFeatures().then(setMeta).catch(() => {});
  }, []);

  const topModel = recommendationReport?.top_recommended_model;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Hero Banner */}
      <div className="hero-card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span className="pill pill-green">
            <Sparkles size={12} />
            <span>Green AI Decision Cockpit</span>
          </span>
          <span className="pill pill-muted">FastAPI + React Production Architecture</span>
        </div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
          EcoML — Carbon &amp; Energy-Aware Machine Learning Selector
        </h2>
        <p style={{ maxWidth: '820px', marginTop: '8px', fontSize: '0.92rem', lineHeight: '1.6' }}>
          Evaluate predictive accuracy alongside computational telemetry: training time, per-sample inference latency,
          peak RAM footprint, and hardware carbon emissions. EcoML calculates a multi-criteria decision score to recommend
          the optimal trade-off model balancing predictive quality and computational resource efficiency.
        </p>

        <div style={{ display: 'flex', gap: '12px', marginTop: '1.25rem' }}>
          <button className="btn btn-primary" onClick={() => onNavigate('datasets')}>
            <span>{datasetName ? 'Configure Preprocessing' : 'Load Dataset in Studio'}</span>
            <ArrowRight size={16} />
          </button>
          <button className="btn btn-secondary" onClick={() => onNavigate('methodology')}>
            <BookOpen size={15} />
            <span>Methodology &amp; Formula</span>
          </button>
        </div>
      </div>

      {/* Real-time Session Stat Tiles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
        <StatCard
          label="Active Dataset"
          value={datasetName || 'None Selected'}
          subtext={
            datasetPreview
              ? `${datasetPreview.rows.toLocaleString()} rows × ${datasetPreview.columns} cols`
              : 'Select a dataset in Studio'
          }
          icon={Database}
          badge={{
            text: datasetPreview ? 'Loaded' : 'Pending',
            variant: datasetPreview ? 'sky' : 'muted',
          }}
          accentColor={datasetPreview ? 'var(--accent-sky)' : undefined}
        />

        <StatCard
          label="Pipeline Status"
          value={prepData ? prepData.task_type.toUpperCase() : 'Unprocessed'}
          subtext={
            prepData
              ? `${prepData.train_samples} train / ${prepData.test_samples} test`
              : 'Configure split in Studio'
          }
          icon={ShieldCheck}
          badge={{
            text: prepData ? 'Partitioned' : 'Awaiting Split',
            variant: prepData ? 'green' : 'muted',
          }}
          accentColor={prepData ? 'var(--eco-emerald)' : undefined}
        />

        <StatCard
          label="Models Evaluated"
          value={benchmarkResults ? benchmarkResults.models_evaluated.length : 0}
          subtext={
            benchmarkResults
              ? `Completed at ${new Date(benchmarkResults.benchmark_timestamp).toLocaleTimeString()}`
              : 'Run benchmark in Lab'
          }
          icon={Cpu}
          badge={{
            text: benchmarkResults ? 'Benchmarked' : 'Not Run',
            variant: benchmarkResults ? 'green' : 'muted',
          }}
          accentColor={benchmarkResults ? 'var(--accent-amber)' : undefined}
        />

        <StatCard
          label="Top Recommendation"
          value={topModel ? topModel.model_name : 'Pending'}
          subtext={
            topModel
              ? `Score: ${topModel.composite_score.toFixed(2)}/100 (${recommendationReport?.profile_name})`
              : 'Compute in Intelligence'
          }
          icon={Leaf}
          badge={{
            text: topModel?.is_pareto_optimal ? 'Pareto Optimal' : 'Computed',
            variant: topModel ? 'green' : 'muted',
          }}
          accentColor={topModel ? 'var(--eco-emerald-light)' : undefined}
        />
      </div>

      {/* Guided 4-Step Architecture Flow */}
      <div>
        <h3 style={{ fontSize: '1.15rem', marginBottom: '0.85rem' }}>
          Standardized Experimentation Workflow
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          {/* Step 1 */}
          <div
            className="card"
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
            onClick={() => onNavigate('datasets')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="pill pill-sky">Step 1</span>
                <Database size={18} color="var(--accent-sky)" />
              </div>
              <h4 style={{ fontSize: '1rem', color: '#FFFFFF', marginBottom: '6px' }}>
                Dataset Studio
              </h4>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.5' }}>
                Load built-in datasets (Breast Cancer, Diabetes, Demo Synthetic) or upload custom CSV. Inspect missing values,
                validate target balance, and create reproducible train/test partitions.
              </p>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-sky)', fontSize: '0.8rem', fontWeight: 600 }}>
              <span>Go to Studio</span>
              <ArrowRight size={14} />
            </div>
          </div>

          {/* Step 2 */}
          <div
            className="card"
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
            onClick={() => onNavigate('benchmarks')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="pill pill-green">Step 2</span>
                <Cpu size={18} color="var(--eco-emerald)" />
              </div>
              <h4 style={{ fontSize: '1rem', color: '#FFFFFF', marginBottom: '6px' }}>
                Model Benchmark Lab
              </h4>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.5' }}>
                Execute scikit-learn classification or regression models under live telemetry sampling. Measure wall-clock training time,
                per-sample latency, and peak RAM delta via psutil.
              </p>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--eco-emerald)', fontSize: '0.8rem', fontWeight: 600 }}>
              <span>Run Benchmarks</span>
              <ArrowRight size={14} />
            </div>
          </div>

          {/* Step 3 */}
          <div
            className="card"
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
            onClick={() => onNavigate('intelligence')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="pill pill-green">Step 3</span>
                <Leaf size={18} color="var(--eco-emerald-light)" />
              </div>
              <h4 style={{ fontSize: '1rem', color: '#FFFFFF', marginBottom: '6px' }}>
                Green Intelligence
              </h4>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.5' }}>
                Analyze multi-objective trade-offs via normalized scoring. Explore the 2D Pareto efficiency frontier and 5-axis
                radar charts. Apply customizable weighting profiles (Balanced, Green-First, Edge).
              </p>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--eco-emerald-light)', fontSize: '0.8rem', fontWeight: 600 }}>
              <span>Analyze Intelligence</span>
              <ArrowRight size={14} />
            </div>
          </div>

          {/* Step 4 */}
          <div
            className="card"
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}
            onClick={() => onNavigate('methodology')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="pill pill-muted">Step 4</span>
                <BookOpen size={18} color="var(--text-secondary)" />
              </div>
              <h4 style={{ fontSize: '1rem', color: '#FFFFFF', marginBottom: '6px' }}>
                Methodology &amp; Standards
              </h4>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.5' }}>
                Inspect the mathematical formulation $Score = \sum w_i S_i$, CodeCarbon hardware emission caveats, psutil measurement
                protocols, and academic research framing.
              </p>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-secondary)', fontSize: '0.8rem', fontWeight: 600 }}>
              <span>View Methodology</span>
              <ArrowRight size={14} />
            </div>
          </div>

          {/* Step 5: Quantum Lab */}
          <div
            className="card"
            style={{ cursor: 'pointer', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', border: '1px solid rgba(16, 185, 129, 0.3)' }}
            onClick={() => onNavigate('quantum')}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                <span className="pill pill-green">Step 5 (Advanced)</span>
                <Atom size={18} color="var(--eco-emerald-light)" />
              </div>
              <h4 style={{ fontSize: '1rem', color: '#FFFFFF', marginBottom: '6px' }}>
                Quantum Lab (QAOA)
              </h4>
              <p style={{ fontSize: '0.82rem', lineHeight: '1.5' }}>
                Formulate constrained model selection as an Ising Spin Glass QUBO and solve it using simulated Quantum Approximate
                Optimization (QAOA) vs. exact classical enumeration.
              </p>
            </div>
            <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--eco-emerald-light)', fontSize: '0.8rem', fontWeight: 600 }}>
              <span>Launch Quantum Lab</span>
              <ArrowRight size={14} />
            </div>
          </div>
        </div>
      </div>

      {/* Platform Features and Capabilities */}
      {meta && (
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem' }}>Verified Platform Capabilities</h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                Core ML algorithms and REST services verified via comprehensive automated test suites
              </p>
            </div>
            <span className="pill pill-green">Verified Suite Snapshot</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
            {meta.implemented_features.map((feat) => (
              <div
                key={feat.id}
                style={{
                  padding: '10px 12px',
                  backgroundColor: 'var(--bg-surface-elevated)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '10px',
                }}
              >
                <CheckCircle2 size={16} color="var(--eco-emerald)" style={{ marginTop: '2px', flexShrink: 0 }} />
                <div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#FFFFFF' }}>{feat.title}</div>
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    {feat.description}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
