import React from 'react';
import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  icon?: LucideIcon;
  badge?: {
    text: string;
    variant?: 'green' | 'sky' | 'amber' | 'muted';
  };
  accentColor?: string;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subtext,
  icon: Icon,
  badge,
  accentColor,
}) => {
  return (
    <div
      className="stat-tile"
      style={{
        position: 'relative',
        overflow: 'hidden',
        borderLeft: accentColor ? `3px solid ${accentColor}` : undefined,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <span className="stat-tile-lbl">{label}</span>
        {badge && (
          <span className={`pill pill-${badge.variant || 'green'}`} style={{ fontSize: '0.68rem', padding: '1px 6px' }}>
            {badge.text}
          </span>
        )}
      </div>

      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between' }}>
        <div className="stat-tile-val" style={{ color: accentColor || '#FFFFFF' }}>
          {value}
        </div>
        {Icon && (
          <div style={{ color: 'var(--text-muted)', opacity: 0.7 }}>
            <Icon size={20} />
          </div>
        )}
      </div>

      {subtext && (
        <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
          {subtext}
        </div>
      )}
    </div>
  );
};

