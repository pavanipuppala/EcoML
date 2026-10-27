# EcoML Frontend (React + Vite)

Target architecture for the modern EcoML Green AI Experimentation UI.

## Structure Planned for Phase 2:
- `src/components/` — UI components (stat cards, hero card, navigation tabs, metrics table, decision matrix)
- `src/features/` — Feature modules:
  - `overview/` — Executive pipeline overview
  - `datasets/` — Tabular data intake, schema inspection, preprocessing configuration
  - `benchmarks/` — Multi-model benchmark execution and telemetry monitoring
  - `intelligence/` — Green Model Recommender, Pareto frontier, radar spider chart
- `src/services/` — Typed API client connecting to FastAPI backend
- `src/types/` — TypeScript interfaces matching backend Pydantic schemas
- `src/styles/` — Deep midnight navy (`#080D1A`) and emerald green (`#10B981`) design tokens

