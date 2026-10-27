import React, { useEffect, useState } from 'react';
import {
  Layers,
  CheckCircle2,
  Activity,
  Award,
} from 'lucide-react';
import { MetadataFeatures } from '../types';
import { api } from '../services/api';

export const MethodologyPage: React.FC = () => {
  const [meta, setMeta] = useState<MetadataFeatures | null>(null);

  useEffect(() => {
    api.getMetaFeatures().then(setMeta).catch(() => {});
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Page Header */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <span className="pill pill-green">Academic Documentation</span>
          <span className="pill pill-muted">Green AI Decision Science</span>
        </div>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
          Methodology, Telemetry &amp; System Architecture
        </h2>
        <p style={{ maxWidth: '820px', marginTop: '8px', fontSize: '0.9rem', lineHeight: '1.6' }}>
          This page documents the formal mathematical models, computational telemetry measurement protocols,
          hardware sensor limitations, and engineering design of the EcoML platform.
        </p>
      </div>

      {/* Mathematical Scoring Formula */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--eco-emerald)',
            }}
          >
            <Award size={18} />
          </div>
          <h3 style={{ fontSize: '1.15rem' }}>1. Multi-Criteria Composite Scoring Formula</h3>
        </div>

        <p style={{ fontSize: '0.86rem', lineHeight: '1.6', marginBottom: '1.25rem' }}>
          EcoML addresses model selection as a formal Multi-Attribute Utility Theory (MAUT) decision problem.
          Rather than selecting models exclusively based on predictive accuracy, EcoML calculates an empirical composite
          score across five standardized dimensions:
        </p>

        {/* Math Display Box */}
        <div
          style={{
            backgroundColor: '#0A0F1D',
            padding: '1.25rem 1.75rem',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.92rem',
            color: 'var(--eco-emerald-light)',
            marginBottom: '1.25rem',
            overflowX: 'auto',
          }}
        >
          EcoML Score = 100 × [ w_P · S_P + w_T · S_T + w_L · S_L + w_M · S_M + w_C · S_C ]
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.86rem', marginBottom: '4px' }}>
              Benefit Dimension (Higher is Better)
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Predictive Utility (S_P: Bounded Utility [0, 1]):
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-sky)', marginTop: '4px' }}>
                S_P = F1_weighted (Classification) | max(0, min(1, R²)) (Regression)
              </div>
            </div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.86rem', marginBottom: '4px' }}>
              Cost Dimensions (Lower is Better)
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Train Time (S_T), Latency (S_L), RAM (S_M), Carbon (S_C):
              <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-amber)', marginTop: '4px' }}>
                S_i = (max(x) - x_i) / (max(x) - min(x))
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Telemetry Measurement Protocol & Caveats */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(56, 189, 248, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-sky)',
            }}
          >
            <Activity size={18} />
          </div>
          <h3 style={{ fontSize: '1.15rem' }}>2. Telemetry Profiling &amp; Measurement Integrity</h3>
        </div>

        <p style={{ fontSize: '0.86rem', lineHeight: '1.6', marginBottom: '1rem' }}>
          Accurate hardware telemetry on commodity hardware poses distinct measurement challenges. EcoML adheres to
          strict scientific integrity principles:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.9rem', marginBottom: '6px' }}>
              Memory Profiling (psutil)
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Measures Resident Set Size (RSS) memory delta in megabytes. Captures baseline process RAM before training,
              polls peak consumption during model fitting, and records net memory allocation overhead.
            </p>
          </div>

          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.9rem', marginBottom: '6px' }}>
              Emission Tracking (CodeCarbon)
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              Monitors hardware power via Intel RAPL or regional grid carbon intensity coefficients (g CO₂e/kWh).
              Hardware-dependent: where sensors are absent or sub-millisecond execution completes below detection thresholds,
              values are honestly labeled as Unavailable or Disabled, never fabricated or substituted with zeros.
            </p>
          </div>

          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
            <div style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.9rem', marginBottom: '6px' }}>
              Weight Redistribution Integrity
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.5' }}>
              If carbon tracking is unavailable or yields zero differential across all models, its assigned weight is
              proportionally redistributed across the remaining active criteria, guaranteeing mathematical normalization
              without penalizing compliant hardware.
            </p>
          </div>
        </div>
      </div>

      {/* Engineering Architecture Diagram */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '1rem' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: 'var(--radius-sm)',
              backgroundColor: 'rgba(251, 191, 36, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--accent-amber)',
            }}
          >
            <Layers size={18} />
          </div>
          <h3 style={{ fontSize: '1.15rem' }}>3. System Architecture &amp; Execution Pipeline</h3>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '12px',
            textAlign: 'center',
            marginBottom: '1.5rem',
          }}
        >
          <div style={{ padding: '1.25rem', backgroundColor: '#0A0F1D', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--accent-sky)', fontWeight: 700, textTransform: 'uppercase' }}>
              Layer 1: Frontend
            </div>
            <div style={{ fontWeight: 700, color: '#FFFFFF', marginTop: '4px' }}>React + Vite (TS)</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Custom Dashboard, High-Contrast UI, SVG Pareto &amp; Radar Charts
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#0A0F1D', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--eco-emerald)', fontWeight: 700, textTransform: 'uppercase' }}>
              Layer 2: REST API
            </div>
            <div style={{ fontWeight: 700, color: '#FFFFFF', marginTop: '4px' }}>FastAPI Backend</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Validated Pydantic Schemas, CORS Middleware, Async Job Status
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#0A0F1D', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--accent-amber)', fontWeight: 700, textTransform: 'uppercase' }}>
              Layer 3: Threading &amp; Telemetry
            </div>
            <div style={{ fontWeight: 700, color: '#FFFFFF', marginTop: '4px' }}>Benchmark Worker</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Sequential Thread Lock, Psutil Sampler, CodeCarbon Tracker
            </div>
          </div>

          <div style={{ padding: '1.25rem', backgroundColor: '#0A0F1D', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--eco-emerald-light)', fontWeight: 700, textTransform: 'uppercase' }}>
              Layer 4: ML Engine
            </div>
            <div style={{ fontWeight: 700, color: '#FFFFFF', marginTop: '4px' }}>Scikit-Learn Core</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
              Shared Partitioning, Standard Scaler, Multi-Criteria Recommender
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
          <CheckCircle2 size={16} color="var(--eco-emerald)" />
          <span>
            Verified by <strong>automated test suites (60+ backend &amp; 14 frontend tests)</strong> covering data preprocessing, benchmark isolation,
            telemetry calculations, and REST API contracts (verified test suite snapshot).
          </span>
        </div>
      </div>

      {/* Roadmap & Feature Implementation Status */}
      {meta && (
        <div className="card">
          <h3 style={{ fontSize: '1.15rem', marginBottom: '1rem' }}>
            Implementation Roadmap &amp; Feature Status
          </h3>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {meta.roadmap_steps.map((s, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.85rem 1.15rem',
                  backgroundColor: 'var(--bg-surface-elevated)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontWeight: 700, color: '#FFFFFF', fontSize: '0.9rem' }}>
                      {s.step}: {s.title}
                    </span>
                    <span
                      className={`pill ${s.status === 'Implemented' ? 'pill-green' : 'pill-muted'}`}
                      style={{ fontSize: '0.65rem' }}
                    >
                      {s.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                    {s.summary}
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
