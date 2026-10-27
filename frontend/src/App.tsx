import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { OverviewPage } from './pages/OverviewPage';
import { DatasetStudioPage } from './pages/DatasetStudioPage';
import { BenchmarkLabPage } from './pages/BenchmarkLabPage';
import { GreenIntelligencePage } from './pages/GreenIntelligencePage';
import { QuantumLabPage } from './pages/QuantumLabPage';
import { MethodologyPage } from './pages/MethodologyPage';
import {
  DatasetPreview,
  PreprocessingResponse,
  BenchmarkResults,
  RecommendationReport,
} from './types';
import { api } from './services/api';
import { storage } from './services/storage';

export const App: React.FC = () => {
  const savedState = storage.load();

  const [currentTab, setCurrentTab] = useState<string>(savedState?.currentTab || 'overview');

  // Core Cross-Tab & Persistent State
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(
    savedState?.selectedDatasetId ?? null
  );
  const [datasetName, setDatasetName] = useState<string | null>(savedState?.datasetName ?? null);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreview | null>(
    savedState?.datasetPreview ?? null
  );
  const [targetCol, setTargetCol] = useState<string>(savedState?.targetCol ?? '');
  const [taskType, setTaskType] = useState<'classification' | 'regression'>(
    savedState?.taskType ?? 'classification'
  );
  const [prepData, setPrepData] = useState<PreprocessingResponse | null>(
    savedState?.prepData ?? null
  );

  // Benchmarking & Recommendation State
  const [activeBenchmarkId, setActiveBenchmarkId] = useState<string | null>(
    savedState?.activeBenchmarkId ?? null
  );
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResults | null>(
    savedState?.benchmarkResults ?? null
  );
  const [recommendationReport, setRecommendationReport] = useState<RecommendationReport | null>(
    savedState?.recommendationReport ?? null
  );

  // Backend Health State
  const [backendOnline, setBackendOnline] = useState<boolean>(false);

  // Check health on mount and periodically
  const checkHealth = async () => {
    try {
      const health = await api.getHealth();
      setBackendOnline(health.status === 'healthy' || health.status === 'ok');
    } catch {
      setBackendOnline(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Sync state to sessionStorage whenever it changes
  useEffect(() => {
    storage.save({
      currentTab,
      selectedDatasetId,
      datasetName,
      datasetPreview,
      targetCol,
      taskType,
      prepData,
      activeBenchmarkId,
      benchmarkResults,
      recommendationReport,
    });
  }, [
    currentTab,
    selectedDatasetId,
    datasetName,
    datasetPreview,
    targetCol,
    taskType,
    prepData,
    activeBenchmarkId,
    benchmarkResults,
    recommendationReport,
  ]);

  // When a new dataset is selected, invalidate downstream stale partitions & benchmarks
  const handleSelectDataset = (id: string, name: string) => {
    if (id !== selectedDatasetId) {
      setSelectedDatasetId(id);
      setDatasetName(name);
      setPrepData(null);
      setActiveBenchmarkId(null);
      setBenchmarkResults(null);
      setRecommendationReport(null);
    }
  };

  // When target column or task type changes, invalidate downstream partitions & benchmarks
  const handleTargetChange = (newTarget: string) => {
    setTargetCol(newTarget);
    setPrepData(null);
    setActiveBenchmarkId(null);
    setBenchmarkResults(null);
    setRecommendationReport(null);
  };

  const handleTaskTypeChange = (newTask: 'classification' | 'regression') => {
    setTaskType(newTask);
    setPrepData(null);
    setActiveBenchmarkId(null);
    setBenchmarkResults(null);
    setRecommendationReport(null);
  };

  // When new preprocessing is executed, invalidate previous benchmarks & recommendations
  const handleSetPrepData = (newPrep: PreprocessingResponse | null) => {
    setPrepData(newPrep);
    setActiveBenchmarkId(null);
    setBenchmarkResults(null);
    setRecommendationReport(null);
  };

  // Reset entire session
  const handleResetSession = () => {
    storage.clear();
    setSelectedDatasetId(null);
    setDatasetName(null);
    setDatasetPreview(null);
    setTargetCol('');
    setTaskType('classification');
    setPrepData(null);
    setActiveBenchmarkId(null);
    setBenchmarkResults(null);
    setRecommendationReport(null);
    setCurrentTab('overview');
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: 'var(--bg-canvas)' }}>
      {/* Sidebar Navigation */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        datasetName={datasetName}
        prepData={prepData}
        benchmarkResults={benchmarkResults}
        backendOnline={backendOnline}
        onResetSession={handleResetSession}
      />

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <Header
          currentTab={currentTab}
          datasetName={datasetName}
          prepData={prepData}
          benchmarkResults={benchmarkResults}
          recommendationReport={recommendationReport}
          backendOnline={backendOnline}
          onRefresh={checkHealth}
          onNavigate={setCurrentTab}
          onResetSession={handleResetSession}
        />

        <main style={{ flex: 1, padding: '2rem', maxWidth: '1400px', width: '100%', margin: '0 auto' }}>
          {currentTab === 'overview' && (
            <OverviewPage
              onNavigate={setCurrentTab}
              datasetName={datasetName}
              datasetPreview={datasetPreview}
              prepData={prepData}
              benchmarkResults={benchmarkResults}
              recommendationReport={recommendationReport}
            />
          )}

          {currentTab === 'datasets' && (
            <DatasetStudioPage
              onNavigate={setCurrentTab}
              selectedDatasetId={selectedDatasetId}
              onSelectDataset={handleSelectDataset}
              datasetPreview={datasetPreview}
              setDatasetPreview={setDatasetPreview}
              prepData={prepData}
              setPrepData={handleSetPrepData}
              targetCol={targetCol}
              setTargetCol={handleTargetChange}
              taskType={taskType}
              setTaskType={handleTaskTypeChange}
            />
          )}

          {currentTab === 'benchmarks' && (
            <BenchmarkLabPage
              onNavigate={setCurrentTab}
              selectedDatasetId={selectedDatasetId}
              datasetName={datasetName}
              prepData={prepData}
              benchmarkResults={benchmarkResults}
              setBenchmarkResults={setBenchmarkResults}
              activeBenchmarkId={activeBenchmarkId}
              setActiveBenchmarkId={setActiveBenchmarkId}
              setRecommendationReport={setRecommendationReport}
            />
          )}

          {currentTab === 'intelligence' && (
            <GreenIntelligencePage
              onNavigate={setCurrentTab}
              activeBenchmarkId={activeBenchmarkId}
              benchmarkResults={benchmarkResults}
              recommendationReport={recommendationReport}
              setRecommendationReport={setRecommendationReport}
            />
          )}

          {currentTab === 'quantum' && (
            <QuantumLabPage
              onNavigate={setCurrentTab}
              activeBenchmarkId={activeBenchmarkId}
              benchmarkResults={benchmarkResults}
            />
          )}

          {currentTab === 'methodology' && <MethodologyPage />}
        </main>
      </div>
    </div>
  );
};

export default App;
