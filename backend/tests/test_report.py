from app.ml.report import build_template_executive_summary, render_report


def _full_context():
    return {
        "filename": "test.csv",
        "summary": {
            "n_rows": 100,
            "n_cols": 5,
            "numeric_columns": ["a", "b"],
            "categorical_columns": ["c"],
            "datetime_columns": [],
            "has_missing": True,
            "duplicate_count": 2,
        },
        "eda_results": {"notable_findings": ["Column 'a' has 10% missing values."]},
        "cleaning_report": {"shape_before": [100, 5], "shape_after": [98, 5], "actions": [{"action": "drop_duplicates", "rows_dropped": 2}]},
        "problem_type": "classification",
        "target_column": "churn",
        "leaderboard": [
            {"rank": 1, "model_id": "xgboost", "display_name": "XGBoost", "cv_score": 0.91, "fit_time_s": 1.2},
            {"rank": 2, "model_id": "random_forest", "display_name": "Random Forest", "cv_score": 0.89, "fit_time_s": 0.8},
        ],
        "best_model_name": "XGBoost",
        "metrics": {"accuracy": 0.91, "f1_weighted": 0.9},
        "shap_results": {"global_importance": [{"feature": "tenure", "mean_abs_shap": 0.31}]},
    }


def test_render_report_with_template_fallback_is_non_empty():
    context = _full_context()
    markdown_text, html_text = render_report(context, executive_summary=None)

    assert "XGBoost" in markdown_text
    assert len(markdown_text) > 0
    assert "<html" not in html_text  # markdown.markdown() returns a fragment, not a full doc
    assert "XGBoost" in html_text
    assert "<table>" in html_text  # leaderboard rendered as a markdown table


def test_template_executive_summary_mentions_best_model():
    context = _full_context()
    summary = build_template_executive_summary(context)
    assert "XGBoost" in summary
    assert "classification" in summary


def test_render_report_with_explicit_executive_summary():
    context = _full_context()
    markdown_text, html_text = render_report(context, executive_summary="Custom LLM-written summary.")
    assert "Custom LLM-written summary." in markdown_text
    assert "Custom LLM-written summary." in html_text
