import React, { useState, useEffect } from 'react';
import {
  Upload,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  Sparkles,
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { DataTable, ColumnDef } from '../components/DataTable';
import {
  DatasetListItem,
  DatasetPreview,
  TargetValidationResponse,
  PreprocessingResponse,
} from '../types';
import { api } from '../services/api';

interface DatasetStudioPageProps {
  onNavigate: (tab: string) => void;
  selectedDatasetId: string | null;
  onSelectDataset: (id: string, name: string) => void;
  datasetPreview: DatasetPreview | null;
  setDatasetPreview: (preview: DatasetPreview | null) => void;
  prepData: PreprocessingResponse | null;
  setPrepData: (prep: PreprocessingResponse | null) => void;
  targetCol: string;
  setTargetCol: (col: string) => void;
  taskType: 'classification' | 'regression';
  setTaskType: (type: 'classification' | 'regression') => void;
}

export const DatasetStudioPage: React.FC<DatasetStudioPageProps> = ({
  onNavigate,
  selectedDatasetId,
  onSelectDataset,
  datasetPreview,
  setDatasetPreview,
  prepData,
  setPrepData,
  targetCol,
  setTargetCol,
  taskType,
  setTaskType,
}) => {
  const [datasets, setDatasets] = useState<DatasetListItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Target validation state
  const [validationResult, setValidationResult] = useState<TargetValidationResponse | null>(null);

  // Preprocessing configuration
  const [testSize, setTestSize] = useState<number>(0.2);
  const [randomSeed, setRandomSeed] = useState<number>(42);
  const [scaleNumerical, setScaleNumerical] = useState<boolean>(true);
  const [preprocessingLoading, setPreprocessingLoading] = useState<boolean>(false);
  const [prepError, setPrepError] = useState<string | null>(null);

  // Load dataset list on mount
  useEffect(() => {
    api
      .listDatasets()
      .then((items) => {
        setDatasets(items);
        // Default to demo if nothing selected
        if (!selectedDatasetId && items.length > 0) {
          handleDatasetChange(items[0].dataset_id, items[0].name);
        }
      })
      .catch((err) => {
        console.error('Failed to load datasets:', err);
      });
  }, []);

  const handleDatasetChange = async (datasetId: string, datasetName: string) => {
    onSelectDataset(datasetId, datasetName);
    setValidationResult(null);
    setPrepError(null);

    try {
      const preview = await api.getDatasetPreview(datasetId, 10);
      setDatasetPreview(preview);

      // Check dataset item for defaults
      const found = datasets.find((d) => d.dataset_id === datasetId);
      const defaultTgt = found?.default_target || preview.column_names[preview.column_names.length - 1];
      const defaultTask = found?.default_task || 'classification';

      setTargetCol(defaultTgt);
      setTaskType(defaultTask);

      // Auto validate
      if (defaultTgt) {
        validateTarget(datasetId, defaultTgt, defaultTask);
      }
    } catch (err: any) {
      console.error('Failed to fetch dataset preview:', err);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      setUploadError('Only CSV files (.csv) are currently supported.');
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setUploadError('File exceeds 50MB limit.');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const res = await api.uploadDataset(file);
      // Refresh list
      const updatedList = await api.listDatasets();
      setDatasets(updatedList);
      await handleDatasetChange(res.dataset_id, res.name);
    } catch (err: any) {
      setUploadError(err.message || 'Failed to upload CSV dataset.');
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const validateTarget = async (
    datasetId: string,
    target: string,
    task: 'classification' | 'regression'
  ) => {
    try {
      const res = await api.validateTarget(datasetId, target, task);
      setValidationResult(res);
    } catch (err: any) {
      setValidationResult({
        target_column: target,
        task_type: task,
        is_valid: false,
        error_message: err.message || 'Validation request failed',
        unique_classes: 0,
        null_count: 0,
      });
    }
  };

  const handleExecutePreprocessing = async () => {
    if (!selectedDatasetId || !targetCol) return;

    setPreprocessingLoading(true);
    setPrepError(null);

    try {
      const res = await api.preprocessDataset(selectedDatasetId, {
        target_column: targetCol,
        task_type: taskType,
        test_size: testSize,
        random_state: randomSeed,
        scale_numerical: scaleNumerical,
      });
      setPrepData(res);
    } catch (err: any) {
      setPrepError(err.message || 'Preprocessing execution failed');
    } finally {
      setPreprocessingLoading(false);
    }
  };

  // Preview table columns definition
  const previewColumns: ColumnDef<Record<string, any>>[] = datasetPreview
    ? datasetPreview.column_names.map((col) => ({
        key: col,
        header: col,
        render: (row) => {
          const val = row[col];
          if (val === null || val === undefined) {
            return <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>null</span>;
          }
          if (typeof val === 'number') {
            return Number.isInteger(val) ? val.toLocaleString() : val.toFixed(4);
          }
          return String(val);
        },
      }))
    : [];

  const selectedDataset = datasets.find((d) => d.dataset_id === selectedDatasetId);

  // Column summary table definition
  const columnDetailList = datasetPreview
    ? datasetPreview.column_names.map((col) => ({
        name: col,
        dtype: datasetPreview.summary.dtypes[col] || 'unknown',
        missing: datasetPreview.summary.missing_counts[col] || 0,
        missingPct: datasetPreview.summary.missing_percentages[col] || 0,
        isTarget: col === targetCol,
      }))
    : [];

  const columnTableCols: ColumnDef<(typeof columnDetailList)[0]>[] = [
    {
      key: 'name',
      header: 'Column Name',
      render: (r) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ fontWeight: 600, color: r.isTarget ? 'var(--eco-emerald-light)' : '#FFFFFF' }}>
            {r.name}
          </span>
          {r.isTarget && (
            <span className="pill pill-green" style={{ fontSize: '0.62rem', padding: '0px 5px' }}>
              Target
            </span>
          )}
        </div>
      ),
    },
    { key: 'dtype', header: 'Data Type' },
    { key: 'missing', header: 'Missing Rows', align: 'right' },
    {
      key: 'missingPct',
      header: 'Missing %',
      align: 'right',
      render: (r) => `${r.missingPct.toFixed(1)}%`,
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Dataset Selection & Upload Header */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem' }}>1. Select or Upload Dataset</h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Choose a curated benchmark dataset or upload your own tabular CSV file
            </p>
          </div>
          <label className="btn btn-secondary" style={{ cursor: 'pointer' }}>
            <Upload size={14} />
            <span>{uploading ? 'Uploading...' : 'Upload CSV (Max 50MB)'}</span>
            <input
              type="file"
              accept=".csv"
              style={{ display: 'none' }}
              onChange={handleFileUpload}
              disabled={uploading}
            />
          </label>
        </div>

        {uploadError && (
          <div
            style={{
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid var(--accent-rose)',
              borderRadius: 'var(--radius-sm)',
              color: '#FDA4AF',
              fontSize: '0.82rem',
              marginBottom: '1rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}
          >
            <span>{uploadError}</span>
            <button
              className="btn btn-ghost"
              onClick={() => setUploadError(null)}
              style={{ fontSize: '0.75rem', padding: '0.2rem 0.5rem', color: '#FDA4AF' }}
            >
              Dismiss
            </button>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '12px' }}>
          {datasets.map((d) => {
            const isSelected = selectedDatasetId === d.dataset_id;
            return (
              <div
                key={d.dataset_id}
                onClick={() => handleDatasetChange(d.dataset_id, d.name)}
                style={{
                  padding: '1rem',
                  borderRadius: 'var(--radius-md)',
                  border: isSelected ? '1px solid var(--eco-emerald)' : '1px solid var(--border-subtle)',
                  backgroundColor: isSelected ? 'rgba(16, 185, 129, 0.08)' : 'var(--bg-surface-elevated)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  position: 'relative',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 700, color: isSelected ? 'var(--eco-emerald-light)' : '#FFFFFF' }}>
                    {d.name}
                  </span>
                  <span className={`pill ${d.source_type === 'built_in' ? 'pill-sky' : 'pill-green'}`} style={{ fontSize: '0.65rem' }}>
                    {d.source_type === 'built_in' ? 'Built-in' : 'Uploaded'}
                  </span>
                </div>
                <p style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', marginBottom: '8px', lineHeight: '1.4' }}>
                  {d.description}
                </p>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  <span>{d.sample_count ? `${d.sample_count.toLocaleString()} samples` : '—'}</span>
                  <span>•</span>
                  <span>{d.feature_count ? `${d.feature_count} features` : '—'}</span>
                  <span>•</span>
                  <span>{d.default_task || 'classification'}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Dataset Statistics and Schema Inspection */}
      {datasetPreview && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
            <StatCard label="Total Samples" value={datasetPreview.rows.toLocaleString()} subtext="Instances" />
            <StatCard label="Total Columns" value={datasetPreview.columns} subtext="Raw attributes" />
            <StatCard
              label="Numerical Cols"
              value={datasetPreview.summary.numerical_columns.length}
              subtext="Float / Integer"
            />
            <StatCard
              label="Categorical Cols"
              value={datasetPreview.summary.categorical_columns.length}
              subtext="Text / Categorical"
            />
            <StatCard
              label="Memory Footprint"
              value={`${(datasetPreview.summary.memory_usage_kb / 1024).toFixed(2)} MB`}
              subtext="In-memory dataframe"
            />
          </div>

          {/* Preview Table */}
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <div>
                <h3 style={{ fontSize: '1.1rem' }}>2. Dataset Preview (First 10 Rows)</h3>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Verify column alignment, data representations, and missing values
                </p>
              </div>
              <span className="pill pill-muted">10 preview rows</span>
            </div>
            <DataTable
              columns={previewColumns}
              data={datasetPreview.preview_rows}
              pageSize={10}
              keyExtractor={(_, i) => i}
            />
          </div>

          {/* Configuration & Preprocessing Section */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            {/* Target and Task Configuration */}
            <div className="card">
              <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                3. Target Feature &amp; Task Type
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
                Specify the dependent prediction variable and the problem formulation
              </p>

              <div className="form-group">
                <label className="form-label">Target Column</label>
                <select
                  className="form-select"
                  value={targetCol}
                  onChange={(e) => {
                    const val = e.target.value;
                    setTargetCol(val);
                    if (selectedDatasetId) validateTarget(selectedDatasetId, val, taskType);
                  }}
                >
                  {selectedDataset?.default_target && datasetPreview.column_names.includes(selectedDataset.default_target) ? (
                    <>
                      <optgroup label="Recommended Target Column">
                        <option value={selectedDataset.default_target}>
                          {selectedDataset.default_target} ★ (Recommended Target)
                        </option>
                      </optgroup>
                      <optgroup label="Predictor Features / Other Columns">
                        {datasetPreview.column_names
                          .filter((c) => c !== selectedDataset.default_target)
                          .map((col) => (
                            <option key={col} value={col}>
                              {col}
                            </option>
                          ))}
                      </optgroup>
                    </>
                  ) : (
                    datasetPreview.column_names.map((col) => (
                      <option key={col} value={col}>
                        {col}
                      </option>
                    ))
                  )}
                </select>
                {selectedDataset?.default_target && (
                  <p style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '5px' }}>
                    Recommended target for this dataset: <strong style={{ color: 'var(--eco-emerald-light)' }}>{selectedDataset.default_target}</strong>
                    {selectedDatasetId === 'breast_cancer' && ' (Diagnosis: 0 = Malignant, 1 = Benign; other 30 columns are biological predictors)'}
                  </p>
                )}
              </div>

              <div className="form-group">
                <label className="form-label">Task Formulation</label>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <button
                    type="button"
                    className={`btn ${taskType === 'classification' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ flex: 1 }}
                    onClick={() => {
                      setTaskType('classification');
                      if (selectedDatasetId && targetCol)
                        validateTarget(selectedDatasetId, targetCol, 'classification');
                    }}
                  >
                    Classification
                  </button>
                  <button
                    type="button"
                    className={`btn ${taskType === 'regression' ? 'btn-primary' : 'btn-secondary'}`}
                    style={{ flex: 1 }}
                    onClick={() => {
                      setTaskType('regression');
                      if (selectedDatasetId && targetCol)
                        validateTarget(selectedDatasetId, targetCol, 'regression');
                    }}
                  >
                    Regression
                  </button>
                </div>
              </div>

              {/* Target Validation Feedback */}
              {validationResult && (
                <div
                  style={{
                    marginTop: '1rem',
                    padding: '0.85rem 1rem',
                    borderRadius: 'var(--radius-sm)',
                    backgroundColor: validationResult.is_valid
                      ? 'rgba(16, 185, 129, 0.08)'
                      : 'rgba(244, 63, 94, 0.08)',
                    border: `1px solid ${
                      validationResult.is_valid ? 'var(--border-emerald)' : 'var(--accent-rose)'
                    }`,
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {validationResult.is_valid ? (
                      <CheckCircle2 size={16} color="var(--eco-emerald)" />
                    ) : (
                      <AlertTriangle size={16} color="var(--accent-rose)" />
                    )}
                    <span
                      style={{
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        color: validationResult.is_valid ? 'var(--eco-emerald-light)' : '#FDA4AF',
                      }}
                    >
                      {validationResult.is_valid ? 'Valid Target Variable' : 'Target Validation Failed'}
                    </span>
                  </div>
                  {validationResult.error_message && (
                    <p style={{ fontSize: '0.76rem', color: '#FDA4AF', marginTop: '4px' }}>
                      {validationResult.error_message}
                    </p>
                  )}
                  <div style={{ display: 'flex', gap: '16px', marginTop: '6px', fontSize: '0.74rem', color: 'var(--text-secondary)' }}>
                    <span>Distinct Values: <strong style={{ color: '#FFFFFF' }}>{validationResult.unique_classes}</strong></span>
                    <span>Null Count: <strong style={{ color: '#FFFFFF' }}>{validationResult.null_count}</strong></span>
                  </div>
                </div>
              )}
            </div>

            {/* Partition & Preprocessing Pipeline Controls */}
            <div className="card">
              <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                4. Partition &amp; Pipeline Configuration
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
                Tune train/test split proportion, random state, and scaling transformations
              </p>

              <div className="form-group">
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <label className="form-label">Test Split Proportion</label>
                  <span style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--eco-emerald-light)' }}>
                    {Math.round(testSize * 100)}% ({Math.round((1 - testSize) * 100)}% Train)
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.05"
                  value={testSize}
                  onChange={(e) => setTestSize(parseFloat(e.target.value))}
                  style={{ width: '100%', accentColor: 'var(--eco-emerald)' }}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Random Seed</label>
                <input
                  type="number"
                  className="form-input"
                  value={randomSeed}
                  onChange={(e) => setRandomSeed(parseInt(e.target.value, 10) || 42)}
                />
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', margin: '1rem 0' }}>
                <input
                  type="checkbox"
                  id="scale_check"
                  checked={scaleNumerical}
                  onChange={(e) => setScaleNumerical(e.target.checked)}
                  style={{ accentColor: 'var(--eco-emerald)', width: '16px', height: '16px' }}
                />
                <label htmlFor="scale_check" style={{ fontSize: '0.84rem', cursor: 'pointer', color: 'var(--text-primary)' }}>
                  Standardize Numerical Features (Z-score StandardScaler)
                </label>
              </div>

              {prepError && (
                <div
                  style={{
                    padding: '0.75rem 1rem',
                    backgroundColor: 'rgba(244, 63, 94, 0.1)',
                    border: '1px solid var(--accent-rose)',
                    borderRadius: 'var(--radius-sm)',
                    color: '#FDA4AF',
                    fontSize: '0.78rem',
                    marginBottom: '1rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    gap: '10px',
                  }}
                >
                  <span>{prepError}</span>
                  <button
                    className="btn btn-secondary"
                    onClick={handleExecutePreprocessing}
                    style={{ fontSize: '0.72rem', padding: '0.25rem 0.6rem', flexShrink: 0 }}
                  >
                    Retry
                  </button>
                </div>
              )}

              <button
                className="btn btn-primary"
                style={{ width: '100%' }}
                disabled={preprocessingLoading || !validationResult?.is_valid}
                onClick={handleExecutePreprocessing}
              >
                {preprocessingLoading ? (
                  <span>Partitioning Dataset...</span>
                ) : (
                  <>
                    <span>Execute Preprocessing Pipeline</span>
                    <Sparkles size={16} />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Preprocessing Confirmation Banner */}
          {prepData && (
            <div
              className="card"
              style={{
                backgroundColor: 'rgba(16, 185, 129, 0.06)',
                border: '1px solid var(--border-emerald)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '1.25rem 1.75rem',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                  <CheckCircle2 size={18} color="var(--eco-emerald)" />
                  <span style={{ fontSize: '1rem', fontWeight: 700, color: '#FFFFFF' }}>
                    Preprocessing Partition Complete
                  </span>
                  <span className="pill pill-green">Ready for Benchmarking</span>
                </div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  {prepData.train_samples} Training samples • {prepData.test_samples} Test samples •{' '}
                  {prepData.feature_count} Transformed features • Hash:{' '}
                  <code style={{ color: 'var(--text-primary)' }}>{prepData.config_signature}</code>
                </div>
              </div>

              <button className="btn btn-primary" onClick={() => onNavigate('benchmarks')}>
                <span>Proceed to Benchmark Lab</span>
                <ArrowRight size={16} />
              </button>
            </div>
          )}

          {/* Column Summary Table */}
          <div className="card">
            <h3 style={{ fontSize: '1.05rem', marginBottom: '0.85rem' }}>
              Column Types &amp; Missingness Summary
            </h3>
            <DataTable columns={columnTableCols} data={columnDetailList} pageSize={8} />
          </div>
        </>
      )}
    </div>
  );
};
