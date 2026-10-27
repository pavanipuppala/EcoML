import React from 'react';
import {
  Compass,
  Database,
  Cpu,
  Leaf,
  Atom,
  BookOpen,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { PreprocessingResponse, BenchmarkResults } from '../types';

interface SidebarProps {
  currentTab: string;
  onSelectTab: (tab: string) => void;
  datasetName: string | null;
  prepData: PreprocessingResponse | null;
  benchmarkResults: BenchmarkResults | null;
  backendOnline: boolean;
  onResetSession?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onSelectTab,
  datasetName,
  prepData,
  benchmarkResults,
  backendOnline,
  onResetSession,
}) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: Compass },
    { id: 'datasets', label: 'Dataset Studio', icon: Database },
    { id: 'benchmarks', label: 'Benchmark Lab', icon: Cpu },
    { id: 'intelligence', label: 'Green Intelligence', icon: Leaf },
    { id: 'quantum', label: 'Quantum Lab', icon: Atom },
    { id: 'methodology', label: 'Methodology', icon: BookOpen },
  ];

  return (
    <aside style={{
      width: '260px',
      backgroundColor: 'var(--bg-sidebar)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      position: 'sticky',
      top: 0,
      flexShrink: 0,
    }}>
      {/* Brand Header */}
      <div style={{ padding: '1.25rem 1.4rem', borderBottom: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '32px',
            height: '32px',
            borderRadius: 'var(--radius-sm)',
            backgroundColor: 'rgba(16, 185, 129, 0.15)',
            border: '1px solid var(--border-emerald)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--eco-emerald-light)',
            fontWeight: 800,
            fontSize: '1.1rem',
          }}>
            🌱
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#FFFFFF', letterSpacing: '-0.02em' }}>
                EcoML
              </span>
              <span className="pill pill-green" style={{ fontSize: '0.65rem', padding: '1px 5px' }}>
                v0.4.0
              </span>
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Carbon &amp; Energy-Aware ML
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Links */}
      <nav style={{ padding: '1rem 0.75rem', flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
        <div style={{ fontSize: '0.68rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', padding: '0 0.65rem 0.4rem 0.65rem' }}>
          Platform Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '0.6rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                backgroundColor: isActive ? 'rgba(16, 185, 129, 0.12)' : 'transparent',
                color: isActive ? 'var(--eco-emerald-light)' : 'var(--text-secondary)',
                fontWeight: isActive ? 600 : 500,
                fontSize: '0.88rem',
                border: 'none',
                cursor: 'pointer',
                textAlign: 'left',
                width: '100%',
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.03)';
                  e.currentTarget.style.color = '#FFFFFF';
                }
              }}
              onMouseLeave={(e) => {
                if (!isActive) {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.color = 'var(--text-secondary)';
                }
              }}
            >
              <Icon size={17} color={isActive ? 'var(--eco-emerald-light)' : 'var(--text-muted)'} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Session Context Snapshot */}
      <div style={{ padding: '1rem', borderTop: '1px solid var(--border-subtle)', backgroundColor: 'rgba(0, 0, 0, 0.2)' }}>
        <div style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', marginBottom: '0.6rem' }}>
          Session State
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.78rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Dataset:</span>
            <span className={`pill ${datasetName ? 'pill-green' : 'pill-muted'}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
              {datasetName ? (datasetName.length > 14 ? datasetName.slice(0, 14) + '...' : datasetName) : 'None'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Pipeline:</span>
            <span className={`pill ${prepData ? 'pill-green' : 'pill-muted'}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
              {prepData ? 'Configured' : 'Pending'}
            </span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Benchmarks:</span>
            <span className={`pill ${benchmarkResults ? 'pill-green' : 'pill-muted'}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
              {benchmarkResults ? `${benchmarkResults.models_evaluated.length} Models` : 'Idle'}
            </span>
          </div>
        </div>

        {/* Backend health status pill */}
        <div style={{ marginTop: '0.8rem', paddingTop: '0.8rem', borderTop: '1px solid var(--border-subtle)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.72rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>FastAPI Engine:</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '4px', color: backendOnline ? 'var(--eco-emerald-light)' : 'var(--accent-rose)' }}>
            {backendOnline ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
            {backendOnline ? 'Online' : 'Offline'}
          </span>
        </div>

        {onResetSession && (
          <button
            onClick={() => {
              if (window.confirm('Reset EcoML session state?')) {
                onResetSession();
              }
            }}
            className="btn btn-ghost"
            style={{
              width: '100%',
              marginTop: '0.75rem',
              padding: '0.35rem 0.6rem',
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              border: '1px solid rgba(148, 163, 184, 0.1)',
            }}
          >
            Reset Session
          </button>
        )}
      </div>
    </aside>
  );
};

