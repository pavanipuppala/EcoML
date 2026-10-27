import React from 'react';
import { ModelScoreDetail } from '../types';

interface RadarChartProps {
  models: ModelScoreDetail[];
  maxModels?: number;
}

const ALL_AXES = [
  { key: 's_performance', label: 'Predictive Perf' },
  { key: 's_train_time', label: 'Train Speed' },
  { key: 's_inference_latency', label: 'Inference Speed' },
  { key: 's_ram', label: 'RAM Efficiency' },
  { key: 's_carbon', label: 'Carbon Efficiency' },
];

const MODEL_COLORS = [
  { stroke: '#10B981', fill: 'rgba(16, 185, 129, 0.22)', text: '#34D399' }, // Emerald (Rank 1)
  { stroke: '#38BDF8', fill: 'rgba(56, 189, 248, 0.18)', text: '#7DD3FC' }, // Sky (Rank 2)
  { stroke: '#FBBF24', fill: 'rgba(251, 191, 36, 0.16)', text: '#FDE68A' }, // Amber (Rank 3)
  { stroke: '#A855F7', fill: 'rgba(168, 85, 247, 0.15)', text: '#D8B4FE' }, // Purple
];

export const RadarChart: React.FC<RadarChartProps> = ({ models, maxModels = 3 }) => {
  const displayModels = models.slice(0, maxModels);

  if (!displayModels || displayModels.length === 0) {
    return (
      <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        No model evaluation data available for radar comparison.
      </div>
    );
  }

  const isCarbonTracked = displayModels.some((m) => {
    const norm = m.normalized_scores || {};
    return norm.carbon !== undefined || norm.s_carbon !== undefined;
  });

  // Dynamically include only active dimensions to prevent plotting fabricated zeroes
  const activeAxes = isCarbonTracked
    ? ALL_AXES
    : ALL_AXES.filter((a) => a.key !== 's_carbon');

  const size = 360;
  const center = size / 2;
  const radius = size * 0.36;
  const numAxes = activeAxes.length;
  const angleStep = (2 * Math.PI) / numAxes;

  // Web rings (25%, 50%, 75%, 100%)
  const levels = [0.25, 0.5, 0.75, 1.0];

  const getCoordinates = (value: number, axisIndex: number) => {
    // Start at top (-PI/2) and rotate clockwise
    const angle = -Math.PI / 2 + axisIndex * angleStep;
    const r = radius * Math.max(0, Math.min(1, value));
    return {
      x: center + r * Math.cos(angle),
      y: center + r * Math.sin(angle),
    };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg
        viewBox={`0 0 ${size} ${size}`}
        style={{ width: '100%', maxWidth: `${size}px`, height: 'auto', display: 'block' }}
      >
        {/* Background Concentric Webs */}
        {levels.map((level) => {
          const points = activeAxes.map((_, i) => {
            const { x, y } = getCoordinates(level, i);
            return `${x},${y}`;
          }).join(' ');

          return (
            <polygon
              key={`level-${level}`}
              points={points}
              fill="none"
              stroke="rgba(148, 163, 184, 0.15)"
              strokeWidth="1"
            />
          );
        })}

        {/* Axis Rays */}
        {activeAxes.map((axis, i) => {
          const { x, y } = getCoordinates(1.0, i);
          const labelCoords = getCoordinates(1.18, i);

          return (
            <g key={axis.key}>
              <line
                x1={center}
                y1={center}
                x2={x}
                y2={y}
                stroke="rgba(148, 163, 184, 0.2)"
                strokeWidth="1"
              />
              <text
                x={labelCoords.x}
                y={labelCoords.y + (i === 0 ? -4 : i === 2 || i === 3 ? 10 : 4)}
                textAnchor="middle"
                fontSize="10"
                fontWeight="600"
                fill="var(--text-secondary)"
              >
                {axis.label}
              </text>
            </g>
          );
        })}

        {/* Model Polygons */}
        {displayModels.map((m, modelIdx) => {
          const color = MODEL_COLORS[modelIdx % MODEL_COLORS.length];
          const norm = m.normalized_scores || {};

          const points = activeAxes.map((axis, i) => {
            const rawVal = norm[axis.key] ?? norm[axis.key.replace(/^s_/, '')];
            const val = typeof rawVal === 'number' ? rawVal : 0.5;
            const { x, y } = getCoordinates(val, i);
            return `${x},${y}`;
          }).join(' ');

          return (
            <g key={m.model_id}>
              <polygon
                points={points}
                fill={color.fill}
                stroke={color.stroke}
                strokeWidth="2"
              />
              {activeAxes.map((axis, i) => {
                const rawVal = norm[axis.key] ?? norm[axis.key.replace(/^s_/, '')];
                const val = typeof rawVal === 'number' ? rawVal : 0.5;
                const { x, y } = getCoordinates(val, i);
                return (
                  <circle
                    key={`${m.model_id}-${axis.key}`}
                    cx={x}
                    cy={y}
                    r="3.5"
                    fill={color.stroke}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '12px',
          justifyContent: 'center',
          marginTop: '10px',
        }}
      >
        {displayModels.map((m, idx) => {
          const color = MODEL_COLORS[idx % MODEL_COLORS.length];
          return (
            <div
              key={m.model_id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                fontSize: '0.78rem',
                backgroundColor: 'var(--bg-surface-elevated)',
                padding: '4px 10px',
                borderRadius: 'var(--radius-sm)',
                border: `1px solid ${color.stroke}44`,
              }}
            >
              <span
                style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: color.stroke,
                }}
              />
              <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{m.model_name}</span>
              <span style={{ color: color.text, fontSize: '0.72rem' }}>
                ({m.composite_score.toFixed(2)}/100)
              </span>
            </div>
          );
        })}
      </div>

      {!isCarbonTracked && (
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '8px', textAlign: 'center', fontStyle: 'italic', maxWidth: '420px' }}>
          * Carbon telemetry is unavailable or inactive on this platform. Carbon efficiency is excluded from radar axes and its scoring weight is dynamically redistributed across the 4 active criteria.
        </div>
      )}
    </div>
  );
};
