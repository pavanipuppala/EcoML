"""
EcoML — Green Model Recommendation and Pareto Service.

Bridges the empirical benchmark outputs to the multi-criteria decision making (MCDM)
engine, evaluating models across predictive power and computational costs.
"""

from typing import Any, Dict, List, Optional

from src.benchmark_engine import BenchmarkSuiteResult
from src.recommender import (
    WeightProfile,
    calculate_model_recommendations,
    RecommendationReport,
    ModelScoreDetail,
)
from backend.services.benchmark_service import benchmark_service


class RecommendationService:
    """Service generating green model recommendations and Pareto frontier analysis."""

    def get_preset_profiles(self) -> List[Dict[str, Any]]:
        """Return predefined operational weight profiles."""
        presets = [
            (
                "balanced",
                "Balanced",
                "Equitable balance between predictive quality (40%) and system efficiency (60%).",
                WeightProfile.balanced(),
            ),
            (
                "green_first",
                "Green-First",
                "Maximizes ecological sustainability; penalizes carbon emissions, training compute, and RAM.",
                WeightProfile.green_first(),
            ),
            (
                "accuracy_first",
                "Accuracy-First",
                "Prioritizes high predictive quality (70%) with minor weighting on efficiency.",
                WeightProfile.accuracy_first(),
            ),
            (
                "edge_deployment",
                "Edge-Deployment",
                "Optimized for constrained microcontrollers and mobile edge; emphasizes low inference latency and RAM.",
                WeightProfile.edge_deployment(),
            ),
        ]
        return [
            {
                "profile_id": p_id,
                "name": name,
                "description": desc,
                "weights": {
                    "name": name,
                    "w_performance": prof.w_performance,
                    "w_train_time": prof.w_train_time,
                    "w_inference_latency": prof.w_inference_latency,
                    "w_ram": prof.w_ram,
                    "w_carbon": prof.w_carbon,
                },
            }
            for p_id, name, desc, prof in presets
        ]

    def generate_recommendations(
        self,
        benchmark_id: str,
        custom_weights: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate multi-criteria model ranking and Pareto efficiency analysis."""
        suite: BenchmarkSuiteResult = benchmark_service.get_suite_result_object(benchmark_id)

        # Build WeightProfile domain object
        if custom_weights:
            profile = WeightProfile(
                name=custom_weights.get("name", "Custom"),
                w_performance=float(custom_weights.get("w_performance", 0.40)),
                w_train_time=float(custom_weights.get("w_train_time", 0.15)),
                w_inference_latency=float(custom_weights.get("w_inference_latency", 0.15)),
                w_ram=float(custom_weights.get("w_ram", 0.15)),
                w_carbon=float(custom_weights.get("w_carbon", 0.15)),
            )
        else:
            profile = WeightProfile.balanced()

        report: RecommendationReport = calculate_model_recommendations(suite, weight_profile=profile)

        # Format ranked models
        def _format_model_detail(m: Optional[ModelScoreDetail]) -> Optional[Dict[str, Any]]:
            if m is None:
                return None
            return {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "status": m.status,
                "rank": m.rank,
                "composite_score": round(m.composite_score, 2),
                "is_pareto_optimal": m.is_pareto_optimal,
                "raw_metrics": m.raw_metrics,
                "normalized_scores": {k: round(v, 4) for k, v in m.normalized_scores.items()},
                "effective_weights": {k: round(v, 4) for k, v in m.effective_weights.items()},
                "strengths": m.strengths,
                "tradeoff_notes": m.tradeoff_notes,
            }

        formatted_ranked = [_format_model_detail(m) for m in report.ranked_models if m is not None]

        return {
            "benchmark_id": benchmark_id,
            "dataset_name": report.dataset_name,
            "task_type": report.task_type,
            "profile_name": report.profile_name,
            "ranked_models": formatted_ranked,
            "top_recommended_model": _format_model_detail(report.top_recommended_model),
            "pareto_frontier_models": report.pareto_frontier_models,
            "effective_weights": {k: round(v, 4) for k, v in report.effective_weights.items()},
            "redistribution_note": report.redistribution_note,
            "tradeoff_synthesis": report.tradeoff_synthesis,
            "scoring_methodology": report.scoring_methodology,
        }


# Global singleton instance for recommendation service
recommendation_service = RecommendationService()

