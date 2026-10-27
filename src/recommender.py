"""
EcoML — Multi-Criteria Green Model Recommendation Engine.

This module implements an objective, mathematically sound Multi-Criteria Decision
Making (MCDM) framework that evaluates candidate machine learning models across
predictive performance, execution duration, inference latency, memory footprint,
and environmental impact (energy/carbon).

Key Principles:
1. Dynamic Weight Redistribution: If energy/carbon measurements are unavailable,
   weights are dynamically redistributed without replacing missing values with zeroes.
2. Dual-Axis Pareto Analysis: Identifies Pareto-optimal models that achieve superior
   trade-offs between predictive quality and computational resource costs.
3. Context-Aware Synthesis: Explains why a model is recommended under specific weight
   profiles without claiming universal superiority.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.benchmark_engine import BenchmarkSuiteResult, SingleModelBenchmark


@dataclass
class WeightProfile:
    """User-configurable weights for multi-criteria model evaluation.

    Attributes:
        w_performance: Weight assigned to predictive accuracy / F1-score (default: 0.40).
        w_train_time: Weight assigned to training duration efficiency (default: 0.15).
        w_inference_latency: Weight assigned to inference latency efficiency (default: 0.15).
        w_ram: Weight assigned to peak resident RAM efficiency (default: 0.15).
        w_carbon: Weight assigned to energy/carbon emissions efficiency (default: 0.15).
        name: Profile display name (e.g. 'Balanced', 'Green-First', 'Custom').
    """
    w_performance: float = 0.40
    w_train_time: float = 0.15
    w_inference_latency: float = 0.15
    w_ram: float = 0.15
    w_carbon: float = 0.15
    name: str = "Balanced"

    def validate(self) -> None:
        """Validate that all weights are non-negative and sum to a positive value.

        Raises:
            ValueError: If any weight is negative or all weights sum to zero.
        """
        weights = [
            self.w_performance,
            self.w_train_time,
            self.w_inference_latency,
            self.w_ram,
            self.w_carbon,
        ]
        if any(w < 0 for w in weights):
            raise ValueError(f"Weights cannot be negative: {weights}")
        total = sum(weights)
        if total <= 0:
            raise ValueError("Total sum of criteria weights must be greater than zero.")

    def normalized_dict(self) -> Dict[str, float]:
        """Return criteria weights normalized to sum to 1.0."""
        self.validate()
        total = (
            self.w_performance
            + self.w_train_time
            + self.w_inference_latency
            + self.w_ram
            + self.w_carbon
        )
        return {
            "performance": self.w_performance / total,
            "train_time": self.w_train_time / total,
            "inference_latency": self.w_inference_latency / total,
            "ram": self.w_ram / total,
            "carbon": self.w_carbon / total,
        }

    @classmethod
    def balanced(cls) -> "WeightProfile":
        """Balanced profile: 40% Predictive Performance, 60% System Efficiency."""
        return cls(
            w_performance=0.40,
            w_train_time=0.15,
            w_inference_latency=0.15,
            w_ram=0.15,
            w_carbon=0.15,
            name="Balanced",
        )

    @classmethod
    def green_first(cls) -> "WeightProfile":
        """Eco-First profile: Prioritizes low emissions, minimal training compute, and RAM."""
        return cls(
            w_performance=0.25,
            w_train_time=0.25,
            w_inference_latency=0.15,
            w_ram=0.15,
            w_carbon=0.20,
            name="Green-First",
        )

    @classmethod
    def accuracy_first(cls) -> "WeightProfile":
        """Accuracy-First profile: Heavily weights task performance (70%)."""
        return cls(
            w_performance=0.70,
            w_train_time=0.10,
            w_inference_latency=0.10,
            w_ram=0.05,
            w_carbon=0.05,
            name="Accuracy-First",
        )

    @classmethod
    def edge_deployment(cls) -> "WeightProfile":
        """Edge deployment profile: Prioritizes low inference latency and compact RAM."""
        return cls(
            w_performance=0.30,
            w_train_time=0.10,
            w_inference_latency=0.35,
            w_ram=0.20,
            w_carbon=0.05,
            name="Edge-Deployment",
        )


@dataclass
class ModelScoreDetail:
    """Detailed multi-criteria evaluation score for an individual model.

    Attributes:
        model_id: Programmatic model identifier.
        model_name: Human-readable model display label.
        status: Execution status ('SUCCESS' or 'FAILED').
        rank: Overall recommendation rank (1 = recommended).
        composite_score: Normalized composite EcoML score out of 100.
        is_pareto_optimal: True if model lies on the Pareto efficiency frontier.
        raw_metrics: Unprocessed metric values (F1/R2, timing, memory, emissions).
        normalized_scores: Inverted and normalized scores for each active dimension [0, 1].
        effective_weights: Actual weights applied after dynamic redistribution.
        strengths: Key computational or predictive advantages identified.
        tradeoff_notes: Observed costs or compromises associated with this algorithm.
    """
    model_id: str
    model_name: str
    status: str
    rank: int
    composite_score: float
    is_pareto_optimal: bool
    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    effective_weights: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    tradeoff_notes: List[str] = field(default_factory=list)


@dataclass
class RecommendationReport:
    """Comprehensive recommendation report synthesizing multi-model benchmark results."""
    task_type: str
    dataset_name: str
    profile_name: str
    ranked_models: List[ModelScoreDetail]
    top_recommended_model: Optional[ModelScoreDetail]
    pareto_frontier_models: List[str]
    effective_weights: Dict[str, float]
    redistribution_note: Optional[str]
    tradeoff_synthesis: str
    scoring_methodology: str

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recommendation ranking into a structured comparison DataFrame."""
        rows = []
        for m in self.ranked_models:
            row = {
                "Rank": f"#{m.rank}" if m.status == "SUCCESS" else "N/A",
                "Model": m.model_name,
                "Status": m.status,
                "EcoML Score": round(m.composite_score, 1) if m.status == "SUCCESS" else None,
                "Pareto Optimal": "⭐ Yes" if m.is_pareto_optimal else "No",
                "Predictive Utility": round(m.normalized_scores.get("performance", 0.0) * 100, 1) if m.status == "SUCCESS" else None,
                "Train Time Eff.": round(m.normalized_scores.get("train_time", 0.0) * 100, 1) if m.status == "SUCCESS" else None,
                "Latency Eff.": round(m.normalized_scores.get("inference_latency", 0.0) * 100, 1) if m.status == "SUCCESS" else None,
                "RAM Eff.": round(m.normalized_scores.get("ram", 0.0) * 100, 1) if m.status == "SUCCESS" else None,
            }
            if "carbon" in m.normalized_scores:
                row["Carbon Eff."] = round(m.normalized_scores["carbon"] * 100, 1)
            rows.append(row)
        return pd.DataFrame(rows)


def _normalize_cost_series(values: List[float]) -> List[float]:
    """Invert and normalize cost metrics so lower cost maps to 1.0 (most efficient).

    Guarantees robust edge-case handling for missing, NaN, infinite, or equal values.

    Args:
        values: Raw cost measurements (e.g. seconds, megabytes, grams).

    Returns:
        List[float]: Normalized scores in [0.0, 1.0].
    """
    if not values:
        return []
    clean_vals = [
        float(v) if (v is not None and not np.isnan(v) and not np.isinf(v)) else 0.0
        for v in values
    ]
    min_val = min(clean_vals)
    max_val = max(clean_vals)
    if max_val == min_val:
        return [1.0 for _ in clean_vals]
    return [(max_val - v) / (max_val - min_val) for v in clean_vals]


def _extract_performance_metric(run: SingleModelBenchmark, task_type: str) -> float:
    """Extract primary predictive utility metric bounded in [0.0, 1.0]."""
    if task_type == "classification":
        val = run.f1 if run.f1 is not None else run.accuracy
        return float(val) if val is not None else 0.0
    else:
        # For regression, R2 can be negative; clamp at 0.0 for utility evaluation
        val = run.r2 if run.r2 is not None else 0.0
        return float(max(0.0, min(1.0, val)))


def compute_pareto_frontier(models: List[SingleModelBenchmark], task_type: str) -> List[str]:
    """Identify Pareto-optimal models along Predictive Quality vs. Efficiency Index.

    A candidate model is Pareto-optimal if no other model simultaneously achieves
    strictly superior predictive quality AND lower overall computational resource costs.

    Args:
        models: List of completed model benchmark runs.
        task_type: 'classification' or 'regression'.

    Returns:
        List[str]: Model IDs belonging to the Pareto frontier.
    """
    successful = [m for m in models if m.status == "SUCCESS"]
    if not successful:
        return []
    if len(successful) == 1:
        return [successful[0].model_id]

    # Compute resource cost index for each model across active dimensions
    train_times = [m.training_time_sec or 0.0 for m in successful]
    latencies = [m.latency_per_sample_ms or 0.0 for m in successful]
    rams = [m.ram_current_mb or 0.0 for m in successful]

    norm_tt = _normalize_cost_series(train_times)
    norm_lat = _normalize_cost_series(latencies)
    norm_ram = _normalize_cost_series(rams)

    has_valid_carbon = any(
        m.carbon_emissions_g is not None and m.carbon_emissions_g > 0
        for m in successful
    )
    if has_valid_carbon:
        carbons = [m.carbon_emissions_g or 0.0 for m in successful]
        norm_carb = _normalize_cost_series(carbons)
        efficiency_indices = [
            (norm_tt[i] + norm_lat[i] + norm_ram[i] + norm_carb[i]) / 4.0
            for i in range(len(successful))
        ]
    else:
        efficiency_indices = [
            (norm_tt[i] + norm_lat[i] + norm_ram[i]) / 3.0
            for i in range(len(successful))
        ]

    performances = [_extract_performance_metric(m, task_type) for m in successful]

    pareto_ids: List[str] = []
    for i, model_a in enumerate(successful):
        perf_a = performances[i]
        eff_a = efficiency_indices[i]
        dominated = False
        for j, model_b in enumerate(successful):
            if i == j:
                continue
            perf_b = performances[j]
            eff_b = efficiency_indices[j]
            # Model B dominates Model A if B is >= in both and strictly > in at least one
            if (perf_b >= perf_a and eff_b >= eff_a) and (perf_b > perf_a or eff_b > eff_a):
                dominated = True
                break
        if not dominated:
            pareto_ids.append(model_a.model_id)

    return pareto_ids


def calculate_model_recommendations(
    suite_result: BenchmarkSuiteResult,
    weight_profile: Optional[WeightProfile] = None,
) -> RecommendationReport:
    """Execute multi-criteria scoring, dynamic redistribution, and Pareto frontier ranking.

    Args:
        suite_result: Completed benchmark suite containing individual model runs.
        weight_profile: Custom or preset weighting profile (default: Balanced).

    Returns:
        RecommendationReport: Detailed rankings, scoring breakdowns, and trade-off synthesis.
    """
    profile = weight_profile or WeightProfile.balanced()
    profile.validate()
    initial_weights = profile.normalized_dict()

    successful_runs = [m for m in suite_result.models_evaluated if m.status == "SUCCESS"]
    failed_runs = [m for m in suite_result.models_evaluated if m.status != "SUCCESS"]

    if not successful_runs:
        return RecommendationReport(
            task_type=suite_result.task_type,
            dataset_name=suite_result.dataset_name,
            profile_name=profile.name,
            ranked_models=[],
            top_recommended_model=None,
            pareto_frontier_models=[],
            effective_weights=initial_weights,
            redistribution_note="All candidate models failed. No recommendations available.",
            tradeoff_synthesis="No successful model runs were recorded.",
            scoring_methodology="Weighted Sum Model (WSM) across predictive utility and normalized cost criteria.",
        )

    # 1. Determine active criteria availability across successful models
    has_valid_carbon = any(
        m.carbon_emissions_g is not None and m.carbon_emissions_g > 0
        for m in successful_runs
    )

    # Dynamic Weight Redistribution
    effective_weights = {}
    redistribution_note: Optional[str] = None

    if not has_valid_carbon:
        # Exclude carbon and redistribute its weight proportionally across the remaining 4 dimensions
        carbon_w = initial_weights["carbon"]
        remaining_sum = sum(w for k, w in initial_weights.items() if k != "carbon")
        if remaining_sum > 0:
            for k, w in initial_weights.items():
                if k != "carbon":
                    effective_weights[k] = w / remaining_sum
            redistribution_note = (
                f"Energy/Carbon telemetry was unavailable on this host; its {carbon_w * 100:.1f}% "
                f"weight was dynamically redistributed across predictive performance ({effective_weights['performance']*100:.1f}%), "
                f"training time ({effective_weights['train_time']*100:.1f}%), "
                f"latency ({effective_weights['inference_latency']*100:.1f}%), and "
                f"RAM footprint ({effective_weights['ram']*100:.1f}%)."
            )
        else:
            effective_weights = {"performance": 1.0}
    else:
        effective_weights = dict(initial_weights)

    # 2. Extract and normalize criteria series
    performances = [_extract_performance_metric(m, suite_result.task_type) for m in successful_runs]
    train_times = [m.training_time_sec or 0.0 for m in successful_runs]
    latencies = [m.latency_per_sample_ms or 0.0 for m in successful_runs]
    rams = [m.ram_current_mb or 0.0 for m in successful_runs]

    norm_perf = performances  # Naturally bounded [0, 1]
    norm_tt = _normalize_cost_series(train_times)
    norm_lat = _normalize_cost_series(latencies)
    norm_ram = _normalize_cost_series(rams)

    norm_carbon: Optional[List[float]] = None
    if has_valid_carbon:
        carbons = [m.carbon_emissions_g or 0.0 for m in successful_runs]
        norm_carbon = _normalize_cost_series(carbons)

    # 3. Compute Pareto Frontier
    pareto_ids = compute_pareto_frontier(suite_result.models_evaluated, suite_result.task_type)

    # 4. Calculate Composite Scores
    score_details: List[ModelScoreDetail] = []
    for idx, run in enumerate(successful_runs):
        norm_scores = {
            "performance": norm_perf[idx],
            "train_time": norm_tt[idx],
            "inference_latency": norm_lat[idx],
            "ram": norm_ram[idx],
        }
        if has_valid_carbon and norm_carbon is not None:
            norm_scores["carbon"] = norm_carbon[idx]

        composite = sum(
            norm_scores[dim] * effective_weights[dim]
            for dim in norm_scores
            if dim in effective_weights
        ) * 100.0

        # Formulate model strengths and trade-off observations
        strengths = []
        tradeoffs = []

        if norm_perf[idx] > 0 and norm_perf[idx] == max(norm_perf):
            strengths.append(f"Highest predictive utility ({norm_perf[idx]:.4f})")
        elif norm_perf[idx] <= 0:
            tradeoffs.append(f"Zero predictive utility ({norm_perf[idx]:.4f}) — model failed to classify/predict target correctly")
        if norm_tt[idx] == max(norm_tt):
            strengths.append(f"Fastest training duration ({run.training_time_sec:.4f}s)")
        if norm_lat[idx] == max(norm_lat):
            strengths.append(f"Lowest inference latency ({run.latency_per_sample_ms:.4f} ms/sample)")
        if norm_ram[idx] == max(norm_ram):
            strengths.append(f"Most compact memory footprint ({run.ram_current_mb} MB)")

        if norm_tt[idx] < 0.3:
            tradeoffs.append(f"High training compute ({run.training_time_sec:.4f}s)")
        if norm_lat[idx] < 0.3:
            tradeoffs.append(f"Elevated inference latency ({run.latency_per_sample_ms:.4f} ms/sample)")
        if norm_ram[idx] < 0.3:
            tradeoffs.append(f"Higher memory footprint ({run.ram_current_mb} MB)")

        score_details.append(
            ModelScoreDetail(
                model_id=run.model_id,
                model_name=run.model_name,
                status="SUCCESS",
                rank=0,
                composite_score=round(composite, 2),
                is_pareto_optimal=(run.model_id in pareto_ids),
                raw_metrics={
                    "performance": norm_perf[idx],
                    "accuracy": run.accuracy,
                    "f1": run.f1,
                    "precision": run.precision,
                    "recall": run.recall,
                    "r2": run.r2,
                    "mae": run.mae,
                    "rmse": run.rmse,
                    "training_time_sec": run.training_time_sec,
                    "latency_per_sample_ms": run.latency_per_sample_ms,
                    "ram_current_mb": run.ram_current_mb,
                    "carbon_emissions_g": run.carbon_emissions_g,
                },
                normalized_scores=norm_scores,
                effective_weights=effective_weights,
                strengths=strengths,
                tradeoff_notes=tradeoffs,
            )
        )

    # Sort descending by composite score to assign ranks
    score_details.sort(key=lambda x: x.composite_score, reverse=True)
    for rank_idx, detail in enumerate(score_details):
        detail.rank = rank_idx + 1

    # Include failed models at bottom of list
    for failed_run in failed_runs:
        score_details.append(
            ModelScoreDetail(
                model_id=failed_run.model_id,
                model_name=failed_run.model_name,
                status="FAILED",
                rank=len(score_details) + 1,
                composite_score=0.0,
                is_pareto_optimal=False,
                raw_metrics={
                    "performance": None,
                    "accuracy": None,
                    "f1": None,
                    "precision": None,
                    "recall": None,
                    "r2": None,
                    "mae": None,
                    "rmse": None,
                    "training_time_sec": None,
                    "latency_per_sample_ms": None,
                    "ram_current_mb": None,
                    "carbon_emissions_g": None,
                },
                normalized_scores={},
                effective_weights=effective_weights,
                strengths=[],
                tradeoff_notes=[failed_run.error_message or "Execution failed"],
            )
        )

    top_model = score_details[0] if score_details else None

    # Synthesize evidence-based recommendation explanation
    synthesis_parts = []
    if top_model is not None and top_model.status == "SUCCESS":
        max_perf = max((m.raw_metrics.get("performance") or 0.0) for m in score_details if m.status == "SUCCESS")
        if max_perf <= 0.0:
            synthesis_parts.append(
                "CRITICAL WARNING: All candidate models exhibited zero predictive utility (0.00% accuracy or R² ≤ 0.0). "
                "No candidate model is recommended for deployment. Verify that the target variable and task formulation "
                "(Classification vs Regression) are correctly configured."
            )
        else:
            synthesis_parts.append(
                f"Under the **{profile.name}** profile, **{top_model.model_name}** ranked #1 "
                f"with an EcoML Score of **{top_model.composite_score:.1f}/100**."
            )
            if top_model.strengths:
                synthesis_parts.append(f"Key strengths: {', '.join(top_model.strengths)}.")

            # Contrast with highest raw accuracy if top model was selected for efficiency
            successful_details = [m for m in score_details if m.status == "SUCCESS"]
            perf_champ = max(successful_details, key=lambda m: m.raw_metrics.get("performance") or 0.0) if successful_details else top_model
            if perf_champ and perf_champ.model_id != top_model.model_id:
                synthesis_parts.append(
                    f"Note: **{perf_champ.model_name}** achieved slightly higher raw predictive utility "
                    f"({perf_champ.raw_metrics['performance']:.4f} vs {top_model.raw_metrics['performance']:.4f}), "
                    f"but **{top_model.model_name}** provides superior overall computational efficiency under your weights."
                )

            pareto_names = [m.model_name for m in score_details if m.is_pareto_optimal]
            if pareto_names:
                synthesis_parts.append(
                    f"Pareto-optimal candidates representing defensible trade-offs: {', '.join(pareto_names)}."
                )

    tradeoff_synthesis = " ".join(synthesis_parts)

    return RecommendationReport(
        task_type=suite_result.task_type,
        dataset_name=suite_result.dataset_name,
        profile_name=profile.name,
        ranked_models=score_details,
        top_recommended_model=top_model,
        pareto_frontier_models=pareto_ids,
        effective_weights=effective_weights,
        redistribution_note=redistribution_note,
        tradeoff_synthesis=tradeoff_synthesis,
        scoring_methodology=(
            "Weighted Sum Model (WSM): Benefit metrics (Accuracy/F1/R2) are scaled directly, "
            "while cost metrics (Training Duration, Inference Latency, Peak RAM, Carbon) are "
            "inverted via min-max scaling so lower consumption yields higher utility."
        ),
    )

