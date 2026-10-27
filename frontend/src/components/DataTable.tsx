import React, { useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';

export interface ColumnDef<T> {
  key: string;
  header: string;
  render?: (row: T, index: number) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
  width?: string;
}

interface DataTableProps<T> {
  columns: ColumnDef<T>[];
  data: T[];
  pageSize?: number;
  emptyMessage?: string;
  keyExtractor?: (row: T, index: number) => string | number;
}

export function DataTable<T extends Record<string, any>>({
  columns,
  data,
  pageSize = 10,
  emptyMessage = 'No data available to display.',
  keyExtractor,
}: DataTableProps<T>) {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(data.length / pageSize));
  const effectivePage = Math.min(currentPage, totalPages);
  const startIndex = (effectivePage - 1) * pageSize;
  const currentData = data.slice(startIndex, startIndex + pageSize);

  return (
    <div className="data-table-wrapper">
      <div style={{ overflowX: 'auto' }}>
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  style={{
                    textAlign: col.align || 'left',
                    width: col.width,
                  }}
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {currentData.length === 0 ? (
              <tr>
                <td
                  colSpan={columns.length}
                  style={{
                    textAlign: 'center',
                    padding: '2.5rem 1rem',
                    color: 'var(--text-muted)',
                    fontStyle: 'italic',
                  }}
                >
                  {emptyMessage}
                </td>
              </tr>
            ) : (
              currentData.map((row, idx) => {
                const actualIdx = startIndex + idx;
                const rowKey = keyExtractor ? keyExtractor(row, actualIdx) : actualIdx;
                return (
                  <tr key={rowKey}>
                    {columns.map((col) => (
                      <td
                        key={col.key}
                        style={{
                          textAlign: col.align || 'left',
                          fontFamily: typeof row[col.key] === 'number' ? 'var(--font-mono)' : 'inherit',
                        }}
                      >
                        {col.render ? col.render(row, actualIdx) : row[col.key] ?? '—'}
                      </td>
                    ))}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {data.length > pageSize && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.75rem 1rem',
            borderTop: '1px solid var(--border-subtle)',
            backgroundColor: '#0A0F1D',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
          }}
        >
          <div>
            Showing <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{startIndex + 1}</span> to{' '}
            <span style={{ color: '#FFFFFF', fontWeight: 600 }}>
              {Math.min(startIndex + pageSize, data.length)}
            </span>{' '}
            of <span style={{ color: '#FFFFFF', fontWeight: 600 }}>{data.length}</span> entries
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
              disabled={effectivePage === 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft size={14} /> Previous
            </button>
            <span style={{ padding: '0 4px', fontWeight: 600, color: '#FFFFFF' }}>
              {effectivePage} / {totalPages}
            </span>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
              disabled={effectivePage >= totalPages}
              onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            >
              Next <ChevronRight size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

