import React, { useState, useEffect } from 'react';
import {
  Atom,
  Play,
  Sliders,
  AlertCircle,
  Cpu,
  Zap,
  TrendingDown,
  Layers,
  Activity,
  Code2,
  ArrowLeft,
} from 'lucide-react';
import {
  BenchmarkResults,
  QuantumOptimizationRequest,
  QuantumOptimizationResponse,
  QuantumWeightProfile,
  QuantumInfoResponse,
} from '../types';
import { api } from '../services/api';

interface QuantumLabPageProps {
  onNavigate: (tab: string) => void;
  activeBenchmarkId: string | null;
  benchmarkResults: BenchmarkResults | null;
}

export const QuantumLabPage: React.FC<QuantumLabPageProps> = ({
  onNavigate,
  activeBenchmarkId,
  benchmarkResults,
}) => {
  // Simulator metadata
  const [quantumInfo, setQuantumInfo] = useState<QuantumInfoResponse | null>(null);

  // Configuration state
  const [depthP, setDepthP] = useState<number>(2);
  const [shots, setShots] = useState<number>(1024);
  const [classicalOpt, setClassicalOpt] = useState<string>('COBYLA');
  const [penaltyFactor, setPenaltyFactor] = useState<number>(2.0);

  // Multi-criteria weights
  const [weights, setWeights] = useState<QuantumWeightProfile>({
    w_performance: 0.40,
    w_train_time: 0.15,
    w_inference_latency: 0.15,
    w_ram: 0.15,
    w_carbon: 0.15,
  });

  // Selected models filter
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

  // Execution & results state
  const [isOptimizing, setIsOptimizing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<QuantumOptimizationResponse | null>(null);
  const [activeTabSub, setActiveTabSub] = useState<'comparison' | 'distribution' | 'circuit' | 'math'>('comparison');

  // Load simulator info on mount
  useEffect(() => {
    api.getQuantumInfo()
      .then(setQuantumInfo)
      .catch((err) => console.error('Failed to load quantum info:', err));
  }, []);

  // Execute optimization
  const handleRunOptimization = async (customIds?: string[] | React.MouseEvent) => {
    setIsOptimizing(true);
    setErrorMsg(null);
    try {
      const targetIds = Array.isArray(customIds) ? customIds : selectedModelIds;
      const payload: QuantumOptimizationRequest = {
        benchmark_id: activeBenchmarkId || undefined,
        selected_model_ids: targetIds.length >= 2 ? targetIds : undefined,
        weights,
        depth_p: depthP,
        shots,
        classical_optimizer: classicalOpt,
        penalty_factor: penaltyFactor,
      };
      const res = await api.optimizeQuantum(payload);
      setResult(res);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute quantum optimization workflow.');
    } finally {
      setIsOptimizing(false);
    }
  };

  // Synchronize candidate models and trigger optimization when benchmark results or activeBenchmarkId changes
  useEffect(() => {
    if (benchmarkResults?.models_evaluated?.length) {
      const ids = benchmarkResults.models_evaluated.map((m) => m.model_id).slice(0, 5);
      setSelectedModelIds(ids);
      handleRunOptimization(ids);
    } else {
      handleRunOptimization();
    }
  }, [activeBenchmarkId, benchmarkResults]);

  const handleWeightChange = (key: keyof QuantumWeightProfile, val: number) => {
    setWeights((prev) => ({ ...prev, [key]: val }));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Page Title & Breadcrumbs */}
      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span className="pill pill-muted" style={{ fontSize: '0.72rem' }}>Module V</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>/</span>
            <span style={{ color: 'var(--eco-emerald-light)', fontSize: '0.75rem', fontWeight: 600 }}>
              Combinatorial Optimization
            </span>
            {quantumInfo && (
              <span className="pill pill-sky" style={{ fontSize: '0.68rem', marginLeft: '6px' }}>
                {quantumInfo.simulator_type}
              </span>
            )}
            {result?.meta?.is_synthetic_demo ? (
              <span className="pill pill-amber" style={{ fontSize: '0.68rem', marginLeft: '6px' }}>
                Demo Baseline (Reference Models)
              </span>
            ) : result?.dataset_name ? (
              <span className="pill pill-green" style={{ fontSize: '0.68rem', marginLeft: '6px' }}>
                Active Benchmark: {result.dataset_name}
              </span>
            ) : null}
          </div>
          <button
            onClick={() => onNavigate('benchmarks')}
            className="btn btn-ghost"
            style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.78rem', padding: '3px 8px' }}
          >
            <ArrowLeft size={13} />
            <span>Back to Benchmarks</span>
          </button>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 style={{ fontSize: '1.85rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '10px' }}>
              <Atom size={28} color="var(--eco-emerald-light)" />
              Quantum Optimization Lab
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginTop: '4px', maxWidth: '800px' }}>
              Formulates constrained Green ML model selection as an Ising Cost Hamiltonian and solves the QUBO
              using variational Quantum Approximate Optimization (QAOA) alongside an exact classical baseline.
            </p>
          </div>
          <button
            onClick={handleRunOptimization}
            disabled={isOptimizing}
            className="btn btn-primary"
            style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '0.65rem 1.4rem' }}
          >
            {isOptimizing ? <Activity className="spin" size={17} /> : <Play size={17} />}
            <span>{isOptimizing ? 'Evolving Statevector...' : 'Run QAOA & Baseline'}</span>
          </button>
        </div>
      </div>

      {/* Scientific Transparency Alert */}
      <div style={{
        padding: '1rem 1.25rem',
        borderRadius: 'var(--radius-md)',
        backgroundColor: 'rgba(30, 41, 59, 0.5)',
        border: '1px solid rgba(16, 185, 129, 0.25)',
        display: 'flex',
        alignItems: 'flex-start',
        gap: '12px',
      }}>
        <Atom size={20} color="var(--eco-emerald-light)" style={{ flexShrink: 0, marginTop: '2px' }} />
        <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
          <strong style={{ color: '#FFFFFF' }}>Simulated Quantum Execution: </strong>
          EcoML executes parameterized QAOA circuits via an exact mathematical statevector simulator ($U(C, \gamma)$ and $U(M, \beta)$)
          with Born-rule categorical shot sampling. Demonstrates how constrained quadratic binary optimization (QUBO) maps into
          quantum spin systems without fabricating physical quantum advantage or noisy hardware claims.
        </div>
      </div>

      {/* Error Banner */}
      {errorMsg && (
        <div style={{
          padding: '1rem 1.25rem',
          borderRadius: 'var(--radius-md)',
          backgroundColor: 'rgba(244, 63, 94, 0.1)',
          border: '1px solid var(--border-rose)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          color: 'var(--accent-rose)',
          fontSize: '0.85rem',
        }}>
          <AlertCircle size={18} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Grid: Left Controls, Right Output */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(320px, 380px) 1fr', gap: '1.75rem', alignItems: 'start' }}>
        {/* Left Column: Configuration & Controls */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {/* Circuit & Algorithm Parameters Card */}
          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '1.2rem', paddingBottom: '0.6rem', borderBottom: '1px solid var(--border-subtle)' }}>
              <Sliders size={18} color="var(--eco-emerald-light)" />
              <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                QAOA Parameters
              </h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {/* Circuit Depth p */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Circuit Depth (p layers):</span>
                  <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 700 }}>p = {depthP}</span>
                </div>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[1, 2, 3, 4].map((pVal) => (
                    <button
                      key={pVal}
                      type="button"
                      onClick={() => setDepthP(pVal)}
                      className={`btn ${depthP === pVal ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ flex: 1, padding: '0.4rem 0', fontSize: '0.8rem' }}
                    >
                      p={pVal}
                    </button>
                  ))}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Alternating cost unitary exp(-i&gamma;H_C) and mixer exp(-i&beta;H_M) steps.
                </div>
              </div>

              {/* Measurement Shots */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Measurement Shots:</span>
                  <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 700 }}>{shots} shots</span>
                </div>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {[256, 512, 1024, 2048].map((sVal) => (
                    <button
                      key={sVal}
                      type="button"
                      onClick={() => setShots(sVal)}
                      className={`btn ${shots === sVal ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ flex: 1, padding: '0.4rem 0', fontSize: '0.75rem' }}
                    >
                      {sVal}
                    </button>
                  ))}
                </div>
              </div>

              {/* Classical Angle Optimizer */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '6px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Classical Optimizer:</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{classicalOpt}</span>
                </div>
                <div style={{ display: 'flex', gap: '6px' }}>
                  {['COBYLA', 'Nelder-Mead'].map((opt) => (
                    <button
                      key={opt}
                      type="button"
                      onClick={() => setClassicalOpt(opt)}
                      className={`btn ${classicalOpt === opt ? 'btn-primary' : 'btn-ghost'}`}
                      style={{ flex: 1, padding: '0.4rem 0', fontSize: '0.78rem' }}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>

              {/* Penalty Multiplier P */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Penalty Multiplier (P):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{penaltyFactor.toFixed(1)}x</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="5.0"
                  step="0.5"
                  value={penaltyFactor}
                  onChange={(e) => setPenaltyFactor(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  Enforces strict one-hot constraint $P(\sum x_i - 1)^2$.
                </div>
              </div>
            </div>
          </div>

          {/* Multi-Criteria Optimization Weights Card */}
          <div className="card" style={{ padding: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', paddingBottom: '0.6rem', borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Zap size={18} color="var(--eco-emerald-light)" />
                <h3 style={{ fontSize: '0.98rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                  Objective Weights
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setWeights({ w_performance: 0.40, w_train_time: 0.15, w_inference_latency: 0.15, w_ram: 0.15, w_carbon: 0.15 })}
                className="btn btn-ghost"
                style={{ padding: '2px 6px', fontSize: '0.72rem' }}
                title="Reset to Balanced weights"
              >
                Reset
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '2px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Predictive Quality ($w_P$):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{weights.w_performance.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.w_performance}
                  onChange={(e) => handleWeightChange('w_performance', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '2px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Training Speed ($w_T$):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{weights.w_train_time.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.w_train_time}
                  onChange={(e) => handleWeightChange('w_train_time', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '2px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Inference Latency ($w_L$):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{weights.w_inference_latency.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.w_inference_latency}
                  onChange={(e) => handleWeightChange('w_inference_latency', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '2px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>RAM Footprint ($w_M$):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{weights.w_ram.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.w_ram}
                  onChange={(e) => handleWeightChange('w_ram', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '2px' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Carbon Emissions ($w_C$):</span>
                  <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{weights.w_carbon.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={weights.w_carbon}
                  onChange={(e) => handleWeightChange('w_carbon', parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Comparison, Histogram, Circuit, Math Inspector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {result ? (
            <>
              {/* Winner Banner / Comparative Alignment */}
              <div className="card" style={{
                padding: '1.4rem 1.6rem',
                border: result.comparison.agreement ? '1px solid var(--border-emerald)' : '1px solid var(--border-amber)',
                backgroundColor: result.comparison.agreement ? 'rgba(16, 185, 129, 0.05)' : 'rgba(245, 158, 11, 0.05)',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
                  <div>
                    <span className={`pill ${result.comparison.agreement ? 'pill-green' : 'pill-amber'}`} style={{ marginBottom: '6px' }}>
                      {result.comparison.agreement ? 'Exact Optimum Agreement' : 'Suboptimal Quantum Exploration'}
                    </span>
                    <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#FFFFFF', margin: '4px 0' }}>
                      QAOA Recommendation: {result.qaoa_result.best_feasible_model?.selected_model || 'None'}
                    </h2>
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Classical Exhaustive Baseline selected: <strong style={{ color: '#FFFFFF' }}>{result.classical_baseline.selected_model_name}</strong>
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Measurement Feasibility</div>
                    <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--eco-emerald-light)' }}>
                      {(result.qaoa_result.feasibility_rate * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <p style={{ fontSize: '0.86rem', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5, borderTop: '1px solid var(--border-subtle)', paddingTop: '0.8rem' }}>
                  {result.comparison.summary}
                </p>
              </div>

              {/* Sub-Tabs Navigation */}
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)', gap: '8px' }}>
                {[
                  { id: 'comparison', label: 'Optimization Comparison', icon: Cpu },
                  { id: 'distribution', label: 'Measurement Histogram', icon: TrendingDown },
                  { id: 'circuit', label: 'Quantum Circuit', icon: Layers },
                  { id: 'math', label: 'QUBO & Ising Hamiltonian', icon: Code2 },
                ].map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTabSub === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTabSub(tab.id as any)}
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px',
                        padding: '0.6rem 0.9rem',
                        border: 'none',
                        borderBottom: isActive ? '2px solid var(--eco-emerald)' : '2px solid transparent',
                        backgroundColor: 'transparent',
                        color: isActive ? 'var(--eco-emerald-light)' : 'var(--text-secondary)',
                        fontWeight: isActive ? 600 : 500,
                        fontSize: '0.85rem',
                        cursor: 'pointer',
                      }}
                    >
                      <Icon size={15} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>

              {/* Tab 1: Comparison View */}
              {activeTabSub === 'comparison' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                  {/* Side-by-Side Diagnostic Cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem' }}>
                    {/* Classical Card */}
                    <div className="card" style={{ padding: '1.25rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem' }}>
                        <Cpu size={18} color="var(--accent-blue)" />
                        <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                          Exact Classical Baseline
                        </h4>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.82rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Selected Model:</span>
                          <strong style={{ color: '#FFFFFF' }}>{result.classical_baseline.selected_model_name}</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Ground State Bitstring:</span>
                          <code style={{ color: 'var(--accent-blue)', backgroundColor: 'rgba(59, 130, 246, 0.1)', padding: '1px 6px', borderRadius: '4px' }}>
                            |{result.classical_baseline.bitstring}⟩
                          </code>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Composite Utility (U):</span>
                          <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 700 }}>
                            {result.classical_baseline.raw_utility.toFixed(4)}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>QUBO Energy (F):</span>
                          <span style={{ color: '#FFFFFF' }}>{result.classical_baseline.energy.toFixed(4)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>States Evaluated:</span>
                          <span style={{ color: '#FFFFFF' }}>{result.classical_baseline.total_states_evaluated} (2^{result.candidates.length})</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Execution Time:</span>
                          <span style={{ color: 'var(--text-muted)' }}>
                            {(result.classical_baseline.execution_time_sec * 1000).toFixed(2)} ms
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* QAOA Card */}
                    <div className="card" style={{ padding: '1.25rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '0.75rem' }}>
                        <Atom size={18} color="var(--eco-emerald-light)" />
                        <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                          Simulated QAOA Circuit
                        </h4>
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.82rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Best Feasible Model:</span>
                          <strong style={{ color: '#FFFFFF' }}>
                            {result.qaoa_result.best_feasible_model?.selected_model || 'None'}
                          </strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Optimal Bitstring:</span>
                          <code style={{ color: 'var(--eco-emerald-light)', backgroundColor: 'rgba(16, 185, 129, 0.1)', padding: '1px 6px', borderRadius: '4px' }}>
                            |{result.qaoa_result.best_feasible_model?.bitstring || 'None'}⟩
                          </code>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Composite Utility (U):</span>
                          <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 700 }}>
                            {result.qaoa_result.best_feasible_model?.raw_utility.toFixed(4) || '0.0000'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Best Feasible Energy (F):</span>
                          <span style={{ color: '#FFFFFF' }}>
                            {result.qaoa_result.best_feasible_model?.energy.toFixed(4) || 'N/A'}
                          </span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }} title="Statevector expectation value <H_C> across full superposition, including penalized infeasible states">
                            Statevector &lang;H_C&rang;:
                          </span>
                          <span style={{ color: '#FFFFFF' }}>{result.qaoa_result.final_expected_energy.toFixed(4)}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Born Measurement Shots:</span>
                          <span style={{ color: '#FFFFFF' }}>{result.qaoa_result.shots}</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ color: 'var(--text-secondary)' }}>Variational Gates:</span>
                          <span style={{ color: 'var(--text-muted)' }}>{result.qaoa_result.total_quantum_gates} quantum gates</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Candidate Model Decision Variable Mapping */}
                  <div className="card" style={{ padding: '1.25rem' }}>
                    <h4 style={{ fontSize: '0.92rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.75rem' }}>
                      Candidate Model Decision Variables (x_i &isin; {'{0, 1}'})
                    </h4>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse', textAlign: 'left' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)' }}>
                            <th style={{ padding: '6px 8px' }}>Qubit</th>
                            <th style={{ padding: '6px 8px' }}>Model</th>
                            <th style={{ padding: '6px 8px' }}>Score</th>
                            <th style={{ padding: '6px 8px' }}>Train Time</th>
                            <th style={{ padding: '6px 8px' }}>Latency</th>
                            <th style={{ padding: '6px 8px' }}>RAM</th>
                            <th style={{ padding: '6px 8px' }}>Utility ($U_i$)</th>
                            <th style={{ padding: '6px 8px' }}>Linear Cost ($f_i$)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.candidates.map((cand) => {
                            const isClWinner = result.classical_baseline.selected_model_id === cand.model_id;
                            const isQWinner = result.qaoa_result.best_feasible_model?.selected_model_id === cand.model_id;
                            return (
                              <tr
                                key={cand.model_id}
                                style={{
                                  borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                                  backgroundColor: isQWinner ? 'rgba(16, 185, 129, 0.06)' : 'transparent',
                                }}
                              >
                                <td style={{ padding: '8px' }}>
                                  <code style={{ color: 'var(--eco-emerald-light)' }}>q{cand.qubit_index}</code>
                                </td>
                                <td style={{ padding: '8px', fontWeight: 600, color: '#FFFFFF' }}>
                                  {cand.model_name}
                                  {isClWinner && <span className="pill pill-blue" style={{ marginLeft: '6px', fontSize: '0.65rem' }}>Classical</span>}
                                  {isQWinner && <span className="pill pill-green" style={{ marginLeft: '6px', fontSize: '0.65rem' }}>QAOA</span>}
                                </td>
                                <td style={{ padding: '8px' }}>{(cand.performance_score * 100).toFixed(2)}%</td>
                                <td style={{ padding: '8px' }}>{(cand.training_time_sec * 1000).toFixed(1)} ms</td>
                                <td style={{ padding: '8px' }}>{cand.inference_latency_ms.toFixed(3)} ms</td>
                                <td style={{ padding: '8px' }}>{cand.ram_mb.toFixed(1)} MB</td>
                                <td style={{ padding: '8px', color: 'var(--eco-emerald-light)', fontWeight: 700 }}>
                                  {cand.normalized_utility.toFixed(4)}
                                </td>
                                <td style={{ padding: '8px', color: 'var(--accent-rose)' }}>
                                  {cand.linear_cost.toFixed(4)}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab 2: Measurement Distribution Histogram */}
              {activeTabSub === 'distribution' && (
                <div className="card" style={{ padding: '1.4rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                        Born-Rule Measurement Distribution
                      </h4>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '2px 0 0 0' }}>
                        Evaluated across {result.qaoa_result.shots} quantum projective measurement shots.
                      </p>
                    </div>
                    <div style={{ display: 'flex', gap: '12px', fontSize: '0.75rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--eco-emerald-light)' }}>
                        <span style={{ width: '10px', height: '10px', backgroundColor: '#10B981', borderRadius: '2px' }} />
                        Feasible ($\sum x_i = 1$)
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '5px', color: 'var(--accent-rose)' }}>
                        <span style={{ width: '10px', height: '10px', backgroundColor: '#F43F5E', borderRadius: '2px' }} />
                        Infeasible ($\sum x_i \neq 1$)
                      </span>
                    </div>
                  </div>

                  {/* SVG Bar Chart of Sampled States */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {result.qaoa_result.samples.map((sample) => {
                      const maxFreq = Math.max(...result.qaoa_result.samples.map((s) => s.shot_frequency), 0.01);
                      const barWidth = Math.max(4, (sample.shot_frequency / maxFreq) * 100);
                      return (
                        <div key={sample.bitstring} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem' }}>
                          <div style={{ width: '70px', fontFamily: 'monospace', color: sample.is_feasible ? 'var(--eco-emerald-light)' : 'var(--text-muted)' }}>
                            |{sample.bitstring}⟩
                          </div>
                          <div style={{ width: '140px', color: sample.is_feasible ? '#FFFFFF' : 'var(--text-muted)', fontSize: '0.78rem', textOverflow: 'ellipsis', whiteSpace: 'nowrap', overflow: 'hidden' }}>
                            {sample.selected_model || '(Infeasible State)'}
                          </div>
                          <div style={{ flex: 1, backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: '4px', height: '22px', position: 'relative', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${barWidth}%`,
                                height: '100%',
                                backgroundColor: sample.is_feasible ? 'rgba(16, 185, 129, 0.75)' : 'rgba(244, 63, 94, 0.45)',
                                borderRadius: '4px',
                                transition: 'width 0.4s ease',
                              }}
                            />
                            <span style={{ position: 'absolute', right: '8px', top: '2px', fontSize: '0.72rem', color: '#FFFFFF', fontWeight: 600 }}>
                              {(sample.shot_frequency * 100).toFixed(1)}% ({sample.counts} shots)
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Total Measured Shots Footer */}
                  <div
                    style={{
                      marginTop: '1rem',
                      paddingTop: '0.75rem',
                      borderTop: '1px solid var(--border-subtle)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      gap: '10px',
                      fontSize: '0.78rem',
                    }}
                  >
                    <span style={{ color: 'var(--text-secondary)' }}>
                      Total Measured Shots:{' '}
                      <strong style={{ color: '#FFFFFF' }}>
                        {result.qaoa_result.samples.reduce((sum, s) => sum + s.counts, 0)} / {result.qaoa_result.shots}
                      </strong>{' '}
                      (100.0%)
                    </span>
                    <span style={{ color: 'var(--eco-emerald-light)' }}>
                      Feasible ({result.qaoa_result.samples.filter((s) => s.is_feasible).reduce((sum, s) => sum + s.counts, 0)} shots):{' '}
                      <strong>{(result.qaoa_result.feasibility_rate * 100).toFixed(1)}%</strong>
                    </span>
                    <span style={{ color: 'var(--accent-rose)' }}>
                      Infeasible ({result.qaoa_result.samples.filter((s) => !s.is_feasible).reduce((sum, s) => sum + s.counts, 0)} shots):{' '}
                      <strong>{((1 - result.qaoa_result.feasibility_rate) * 100).toFixed(1)}%</strong>
                    </span>
                  </div>

                  {/* Convergence Trace */}
                  {result.qaoa_result.convergence_history.length > 0 && (
                    <div style={{ marginTop: '1.75rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
                      <h5 style={{ fontSize: '0.85rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.6rem' }}>
                        Classical Variational Convergence Trace (min &lang;H_C&rang;)
                      </h5>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflowX: 'auto', padding: '0.5rem 0' }}>
                        {result.qaoa_result.convergence_history.map((pt) => (
                          <div
                            key={pt.iteration}
                            style={{
                              padding: '4px 8px',
                              backgroundColor: 'rgba(0, 0, 0, 0.25)',
                              borderRadius: '4px',
                              border: '1px solid rgba(255, 255, 255, 0.05)',
                              fontSize: '0.72rem',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            <span style={{ color: 'var(--text-muted)' }}>iter {pt.iteration}:</span>{' '}
                            <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 600 }}>{pt.energy.toFixed(3)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 3: Quantum Circuit Representation */}
              {activeTabSub === 'circuit' && (
                <div className="card" style={{ padding: '1.4rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <div>
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF', margin: 0 }}>
                        QAOA Parameterized Quantum Circuit Architecture
                      </h4>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '2px 0 0 0' }}>
                        Total gates: {result.qaoa_result.total_quantum_gates} | Depth p = {result.qaoa_result.depth_p} | Qubits n = {result.candidates.length}
                      </p>
                    </div>
                    <span className="pill pill-green" style={{ fontSize: '0.72rem' }}>
                      {result.qaoa_result.engine}
                    </span>
                  </div>

                  {/* Variational Parameters Values */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
                    <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
                        Optimized Cost Angles (&gamma;*)
                      </div>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {result.qaoa_result.optimized_gammas.map((g, idx) => (
                          <code key={idx} style={{ color: 'var(--eco-emerald-light)', fontSize: '0.82rem' }}>
                            &gamma;{idx + 1} = {g}
                          </code>
                        ))}
                      </div>
                    </div>

                    <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.3)', padding: '0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '4px' }}>
                        Optimized Mixer Angles (&beta;*)
                      </div>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {result.qaoa_result.optimized_betas.map((b, idx) => (
                          <code key={idx} style={{ color: 'var(--accent-blue)', fontSize: '0.82rem' }}>
                            &beta;{idx + 1} = {b}
                          </code>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* ASCII Circuit Wire Representation */}
                  <pre
                    style={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      padding: '1.2rem',
                      borderRadius: 'var(--radius-sm)',
                      overflowX: 'auto',
                      fontSize: '0.8rem',
                      color: 'var(--eco-emerald-light)',
                      border: '1px solid var(--border-subtle)',
                      lineHeight: 1.6,
                      fontFamily: '"Fira Code", monospace',
                    }}
                  >
                    {result.qaoa_result.circuit_ascii}
                  </pre>
                </div>
              )}

              {/* Tab 4: QUBO & Ising Math Inspector */}
              {activeTabSub === 'math' && (
                <div className="card" style={{ padding: '1.4rem' }}>
                  <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '0.6rem' }}>
                    Mathematical QUBO &amp; Ising Spin Glass Formulation
                  </h4>
                  <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.6, marginBottom: '1.25rem' }}>
                    The model selection problem is framed as:
                    <div style={{ padding: '0.6rem 0.8rem', backgroundColor: 'rgba(0, 0, 0, 0.3)', borderRadius: '4px', margin: '0.5rem 0', fontFamily: 'monospace', color: 'var(--eco-emerald-light)' }}>
                      min_x &nbsp; &sum; f_i x_i + P (&sum; x_i - 1)&sup2; &nbsp; subject to &nbsp; x_i &isin; {'{0, 1}'}
                    </div>
                    Mapping binary variables to Pauli-Z operators via x_i = (I - Z_i)/2 yields the Ising Cost Hamiltonian:
                    <div style={{ padding: '0.6rem 0.8rem', backgroundColor: 'rgba(0, 0, 0, 0.3)', borderRadius: '4px', margin: '0.5rem 0', fontFamily: 'monospace', color: 'var(--eco-emerald-light)' }}>
                      H_C = &sum; h_i Z_i + &sum; J_ij Z_i Z_j + C_offset
                    </div>
                  </div>

                  {/* Parameter Values */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
                    <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.25)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Constraint Penalty (P)</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF' }}>{result.meta.penalty_parameter_P}</div>
                    </div>
                    <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.25)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>QUBO Constant Offset</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF' }}>{result.meta.qubo_offset}</div>
                    </div>
                    <div style={{ backgroundColor: 'rgba(0, 0, 0, 0.25)', padding: '0.75rem', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Ising C_offset</div>
                      <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#FFFFFF' }}>{result.meta.c_offset}</div>
                    </div>
                  </div>

                  {/* Local fields h_i */}
                  <div style={{ marginBottom: '1rem' }}>
                    <h5 style={{ fontSize: '0.82rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '4px' }}>
                      Single-Qubit Local Fields ($h_i Z_i$)
                    </h5>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {result.ising_h.map((hVal, idx) => (
                        <div key={idx} style={{ padding: '3px 8px', backgroundColor: 'rgba(255, 255, 255, 0.04)', borderRadius: '4px', fontSize: '0.78rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>h_{idx}:</span> <strong style={{ color: '#FFFFFF' }}>{hVal}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Two-qubit couplings J_ij */}
                  <div>
                    <h5 style={{ fontSize: '0.82rem', fontWeight: 700, color: '#FFFFFF', marginBottom: '4px' }}>
                      Two-Qubit Couplings (J_ij Z_i Z_j)
                    </h5>
                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      {Object.entries(result.ising_j).map(([pair, jVal]) => (
                        <div key={pair} style={{ padding: '3px 8px', backgroundColor: 'rgba(255, 255, 255, 0.04)', borderRadius: '4px', fontSize: '0.78rem' }}>
                          <span style={{ color: 'var(--text-muted)' }}>J_({pair}):</span> <strong style={{ color: 'var(--eco-emerald-light)' }}>{jVal}</strong>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              <Atom size={40} color="var(--eco-emerald-light)" style={{ margin: '0 auto 1rem auto', opacity: 0.6 }} />
              <h3 style={{ fontSize: '1.1rem', color: '#FFFFFF', marginBottom: '0.5rem' }}>
                Quantum Optimization Pending
              </h3>
              <p style={{ fontSize: '0.85rem', maxWidth: '420px', margin: '0 auto' }}>
                Click "Run QAOA &amp; Baseline" to formulate the QUBO Hamiltonian and simulate the quantum circuit.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
