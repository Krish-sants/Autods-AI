import logging

from app.agents.base import safe_llm_call
from app.config import get_settings
from app.graph.state import PipelineState
from app.ml.explain import compute_shap
from app.storage.run_store import save_json

logger = logging.getLogger(__name__)


async def explainability_agent(state: PipelineState) -> PipelineState:
    settings = get_settings()
    problem_type = state["problem_type"]

    try:
        if problem_type == "clustering":
            shap_results = {
                "available": False,
                "reason": "SHAP is not applicable to unsupervised clustering; see cluster_profile in metrics instead.",
                "global_importance": [],
                "local_examples": [],
            }
        else:
            shap_results = compute_shap(
                state["best_pipeline"],
                state["best_model_id"],
                state["X_test"],
                max_samples=settings.SHAP_SAMPLE_SIZE,
            )

            if shap_results.get("available") and shap_results.get("global_importance"):
                top_features = ", ".join(f["feature"] for f in shap_results["global_importance"][:5])
                fallback = (
                    f"The model relies most heavily on: {top_features}. These features had the largest "
                    "average impact on individual predictions."
                )
                prompt = (
                    "You are a senior data scientist explaining a machine learning model to a non-technical "
                    f"stakeholder. The top features by SHAP importance are: {top_features}. In 2 sentences, "
                    "explain in plain English what this means for how the model makes decisions."
                )
                explanation, used_llm = await safe_llm_call(prompt, fallback)
                shap_results["plain_english_explanation"] = explanation
                state["narrative_llm_used"] = state.get("narrative_llm_used", False) or used_llm

        state["shap_results"] = shap_results
        save_json(state["run_id"], "shap_results.json", shap_results)

        state["current_step"] = "explainability"
        state.setdefault("steps_completed", []).append("explainability")
    except Exception as exc:
        logger.exception("explainability_agent failed")
        state.setdefault("errors", []).append(f"explainability_agent: {exc}")
        state["shap_results"] = {"available": False, "error": str(exc), "global_importance": [], "local_examples": []}
    return state
