from typing import Any

from app.ml.train import ModelResult


def rank_models(results: list[ModelResult]) -> tuple[list[dict[str, Any]], str | None]:
    valid = [r for r in results if r.error is None and r.estimator is not None]
    valid.sort(key=lambda r: r.cv_score, reverse=True)

    rows: list[dict[str, Any]] = []
    for rank, result in enumerate(valid, start=1):
        rows.append(
            {
                "rank": rank,
                "model_id": result.model_id,
                "display_name": result.display_name,
                "cv_score": round(result.cv_score, 4),
                "fit_time_s": round(result.fit_time_s, 3),
                "best_params": result.best_params,
            }
        )

    failed = [r for r in results if r.error is not None]
    for result in failed:
        rows.append(
            {
                "rank": None,
                "model_id": result.model_id,
                "display_name": result.display_name,
                "cv_score": None,
                "fit_time_s": round(result.fit_time_s, 3),
                "best_params": {},
                "error": result.error,
            }
        )

    best_model_id = rows[0]["model_id"] if valid else None
    return rows, best_model_id
