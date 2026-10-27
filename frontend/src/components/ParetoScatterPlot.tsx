import React, { useState } from 'react';
import { ModelScoreDetail } from '../types';

interface ParetoScatterPlotProps {
  models: ModelScoreDetail[];
  taskType: string;
}

interface PointData {
  id: string;
  name: string;
  x: number; // Predictive Utility % (0 - 100)
  y: number; // System Efficiency Index % (0 - 100)
  isPareto: boolean;
  score: number;
  rank: number;
  raw: Record<string, any>;
}

export const ParetoScatterPlot: React.FC<ParetoScatterPlotProps> = ({ models, taskType }) => {
  const [hoveredPoint, setHoveredPoint] = useState<PointData | null>(null);

  if (!models || models.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No model evaluation data available for Pareto analysis.
      </div>
    );
  }

  // Calculate points
  const points: PointData[] = models.map((m) => {
    // Predictive utility % (Weighted F1 for classification, R² for regression)
    const rawPerf = taskType === 'classification'
      ? (m.raw_metrics.f1 ?? m.raw_metrics.performance ?? m.raw_metrics.accuracy ?? 0)
      : (m.raw_metrics.r2 ?? m.raw_metrics.performance ?? 0);
    const utilPct = Math.max(0, Math.min(100, rawPerf * 100));

    // System Efficiency Index: average of active normalized efficiency scores
    const norm = m.normalized_scores || {};
    const effKeys = ['train_time', 'inference_latency', 'ram'];
    if (norm.carbon !== undefined || norm.s_carbon !== undefined) {
      effKeys.push('carbon');
    }
    let sumEff = 0;
    let countEff = 0;
    effKeys.forEach((k) => {
      const val = norm[k] ?? norm[`s_${k}`];
      if (typeof val === 'number') {
        sumEff += val;
        countEff++;
      }
    });
    const effPct = countEff > 0 ? (sumEff / countEff) * 100 : 50;

    return {
      id: m.model_id,
      name: m.model_name,
      x: utilPct,
      y: effPct,
      isPareto: m.is_pareto_optimal,
      score: m.composite_score,
      rank: m.rank,
      raw: m.raw_metrics,
    };
  });

  // SVG dimensions
  const width = 640;
  const height = 380;
  const margin = { top: 30, right: 40, bottom: 50, left: 60 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  // Domain extents with padding
  const xVals = points.map((p) => p.x);
  const yVals = points.map((p) => p.y);
  const minX = Math.max(0, Math.floor(Math.min(...xVals) / 10) * 10 - 10);
  const maxX = 100;
  const minY = Math.max(0, Math.floor(Math.min(...yVals) / 10) * 10 - 10);
  const maxY = 100;

  const scaleX = (val: number) => {
    const range = maxX - minX || 1;
    return margin.left + ((val - minX) / range) * innerWidth;
  };

  const scaleY = (val: number) => {
    const range = maxY - minY || 1;
    return margin.top + innerHeight - ((val - minY) / range) * innerHeight;
  };

  // Sort Pareto points by X to draw a subtle frontier curve
  const paretoPoints = points
    .filter((p) => p.isPareto)
    .sort((a, b) => a.x - b.x);

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: `${width}px`, margin: '0 auto' }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block', overflow: 'visible' }}
      >
        {/* Background Grid */}
        <rect
          x={margin.left}
          y={margin.top}
          width={innerWidth}
          height={innerHeight}
          fill="#0A0F1D"
          stroke="var(--border-subtle)"
          strokeWidth="1"
          rx="6"
        />

        {/* Horizontal grid lines */}
        {[0, 25, 50, 75, 100].map((tick) => {
          if (tick < minY || tick > maxY) return null;
          const y = scaleY(tick);
          return (
            <g key={`y-grid-${tick}`}>
              <line
                x1={margin.left}
                y1={y}
                x2={margin.left + innerWidth}
                y2={y}
                stroke="rgba(148, 163, 184, 0.1)"
                strokeDasharray="4 4"
              />
              <text
                x={margin.left - 10}
                y={y + 4}
                textAnchor="end"
                fontSize="10"
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
              >
                {tick}%
              </text>
            </g>
          );
        })}

        {/* Vertical grid lines */}
        {[0, 25, 50, 75, 100].map((tick) => {
          if (tick < minX || tick > maxX) return null;
          const x = scaleX(tick);
          return (
            <g key={`x-grid-${tick}`}>
              <line
                x1={x}
                y1={margin.top}
                x2={x}
                y2={margin.top + innerHeight}
                stroke="rgba(148, 163, 184, 0.1)"
                strokeDasharray="4 4"
              />
              <text
                x={x}
                y={margin.top + innerHeight + 18}
                textAnchor="middle"
                fontSize="10"
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
              >
                {tick}%
              </text>
            </g>
          );
        })}

        {/* Pareto Frontier Connecting Line */}
        {paretoPoints.length > 1 && (
          <path
            d={paretoPoints
              .map((p, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(p.x)} ${scaleY(p.y)}`)
              .join(' ')}
            fill="none"
            stroke="var(--eco-emerald)"
            strokeWidth="2"
            strokeDasharray="5 3"
            opacity="0.75"
          />
        )}

        {/* Data Points */}
        {points.map((p) => {
          const cx = scaleX(p.x);
          const cy = scaleY(p.y);
          const isHovered = hoveredPoint?.id === p.id;

          if (p.isPareto) {
            // Emerald diamond for Pareto optimal
            const size = isHovered ? 9 : 7;
            return (
              <g
                key={p.id}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint(p)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                <polygon
                  points={`${cx},${cy - size} ${cx + size},${cy} ${cx},${cy + size} ${cx - size},${cy}`}
                  fill="#10B981"
                  stroke="#FFFFFF"
                  strokeWidth={isHovered ? 2.5 : 1.5}
                  filter="drop-shadow(0 0 6px rgba(16, 185, 129, 0.6))"
                />
                <text
                  x={cx + (cx > width - 120 ? -12 : 12)}
                  y={cy - 8}
                  fill="#34D399"
                  fontSize="10"
                  fontWeight="600"
                  textAnchor={cx > width - 120 ? 'end' : 'start'}
                >
                  {p.name}
                </text>
              </g>
            );
          } else {
            // Subordinate circle
            return (
              <g
                key={p.id}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHoveredPoint(p)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered ? 7 : 5}
                  fill="#1E293B"
                  stroke="#94A3B8"
                  strokeWidth={isHovered ? 2 : 1.5}
                />
                <text
                  x={cx + (cx > width - 120 ? -10 : 10)}
                  y={cy + 4}
                  fill="var(--text-muted)"
                  fontSize="9.5"
                  textAnchor={cx > width - 120 ? 'end' : 'start'}
                >
                  {p.name}
                </text>
              </g>
            );
          }
        })}

        {/* Axis Labels */}
        <text
          x={margin.left + innerWidth / 2}
          y={height - 12}
          textAnchor="middle"
          fontSize="11"
          fontWeight="600"
          fill="var(--text-secondary)"
        >
          Predictive Utility ({taskType === 'classification' ? 'Weighted F1 %' : 'R² %'}) →
        </text>

        <text
          transform={`rotate(-90)`}
          x={-(margin.top + innerHeight / 2)}
          y={16}
          textAnchor="middle"
          fontSize="11"
          fontWeight="600"
          fill="var(--text-secondary)"
        >
          System Efficiency Index % →
        </text>
      </svg>

      {/* Floating Hover Tooltip */}
      {hoveredPoint && (
        <div
          style={{
            position: 'absolute',
            bottom: '20px',
            right: '20px',
            backgroundColor: 'rgba(15, 23, 42, 0.95)',
            border: `1px solid ${hoveredPoint.isPareto ? 'var(--eco-emerald)' : 'var(--border-light)'}`,
            borderRadius: 'var(--radius-sm)',
            padding: '10px 14px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.6)',
            pointerEvents: 'none',
            fontSize: '0.8rem',
            minWidth: '220px',
            zIndex: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '4px' }}>
            <span style={{ fontWeight: 700, color: '#FFFFFF' }}>{hoveredPoint.name}</span>
            <span
              className={`pill ${hoveredPoint.isPareto ? 'pill-green' : 'pill-muted'}`}
              style={{ fontSize: '0.65rem', padding: '1px 5px' }}
            >
              {hoveredPoint.isPareto ? 'Pareto Optimal' : 'Subordinate'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '2px', color: 'var(--text-secondary)' }}>
            <div>
              Composite Score:{' '}
              <span style={{ color: 'var(--eco-emerald-light)', fontWeight: 700 }}>
                {hoveredPoint.score.toFixed(2)} / 100
              </span>
            </div>
            <div>
              {taskType === 'classification' ? 'Weighted F1' : 'R² Score'}:{' '}
              <span style={{ color: '#FFFFFF', fontWeight: 600 }}>
                {(() => {
                  if (hoveredPoint.raw.status === 'FAILED') return 'Failed';
                  if (taskType === 'classification') {
                    const f1 = hoveredPoint.raw.f1 ?? hoveredPoint.raw.performance;
                    return f1 !== undefined && f1 !== null ? `${(f1 * 100).toFixed(2)}%` : '—';
                  } else {
                    const r2 = hoveredPoint.raw.r2 ?? hoveredPoint.raw.performance;
                    return r2 !== undefined && r2 !== null ? r2.toFixed(4) : '—';
                  }
                })()}
              </span>
            </div>
            {taskType === 'classification' && hoveredPoint.raw.accuracy !== undefined && hoveredPoint.raw.accuracy !== null && (
              <div>
                Accuracy:{' '}
                <span style={{ color: '#FFFFFF' }}>{(hoveredPoint.raw.accuracy * 100).toFixed(2)}%</span>
              </div>
            )}
            {taskType !== 'classification' && hoveredPoint.raw.rmse !== undefined && hoveredPoint.raw.rmse !== null && (
              <div>
                RMSE:{' '}
                <span style={{ color: '#FFFFFF' }}>{hoveredPoint.raw.rmse.toFixed(2)}</span>
              </div>
            )}
            {taskType !== 'classification' && hoveredPoint.raw.mae !== undefined && hoveredPoint.raw.mae !== null && (
              <div>
                MAE:{' '}
                <span style={{ color: '#FFFFFF' }}>{hoveredPoint.raw.mae.toFixed(2)}</span>
              </div>
            )}
            <div>
              Efficiency Index:{' '}
              <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{hoveredPoint.y.toFixed(1)}%</span>
            </div>
            {hoveredPoint.raw.training_time_sec !== undefined && (
              <div>
                Train Time:{' '}
                <span style={{ color: '#FFFFFF' }}>
                  {(hoveredPoint.raw.training_time_sec * 1000).toFixed(1)} ms
                </span>
              </div>
            )}
            {hoveredPoint.raw.latency_per_sample_ms !== undefined && (
              <div>
                Sample Latency:{' '}
                <span style={{ color: '#FFFFFF' }}>
                  {hoveredPoint.raw.latency_per_sample_ms.toFixed(4)} ms/sample
                </span>
              </div>
            )}
            {hoveredPoint.raw.ram_current_mb !== undefined && (
              <div>
                Peak RAM:{' '}
                <span style={{ color: '#FFFFFF' }}>
                  {hoveredPoint.raw.ram_current_mb} MB
                </span>
              </div>
            )}
            <div>
              Carbon Telemetry:{' '}
              <span style={{ color: hoveredPoint.raw.carbon_emissions_g ? 'var(--eco-emerald-light)' : 'var(--text-muted)' }}>
                {hoveredPoint.raw.carbon_emissions_g !== undefined && hoveredPoint.raw.carbon_emissions_g !== null && hoveredPoint.raw.carbon_emissions_g > 0
                  ? `${hoveredPoint.raw.carbon_emissions_g.toExponential(3)} g`
                  : 'Unavailable (Redistributed)'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '24px',
          marginTop: '8px',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              width: '10px',
              height: '10px',
              transform: 'rotate(45deg)',
              backgroundColor: 'var(--eco-emerald)',
              display: 'inline-block',
            }}
          />
          <span style={{ color: '#FFFFFF', fontWeight: 600 }}>Pareto Frontier</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: '#1E293B',
              border: '1.5px solid #94A3B8',
              display: 'inline-block',
            }}
          />
          <span>Dominated Models</span>
        </div>
      </div>

      {models.some((m) => (m.effective_weights?.carbon ?? 0) === 0) && (
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '6px', fontStyle: 'italic' }}>
          * System Efficiency Index computed over active computational metrics (train time, latency, RAM). Carbon weight dynamically redistributed.
        </div>
      )}
    </div>
  );
};
