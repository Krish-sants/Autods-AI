import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bleach
import markdown as markdown_lib
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# Dataset column names/values flow unescaped into the markdown template (e.g. EDA
# findings, cleaning actions), so the rendered HTML must be sanitized before the
# frontend renders it with dangerouslySetInnerHTML — a crafted CSV column name
# like "<script>..." should not survive into live HTML.
_ALLOWED_TAGS = [
    "p", "br", "hr", "h1", "h2", "h3", "h4", "ul", "ol", "li", "strong", "em",
    "code", "pre", "blockquote", "table", "thead", "tbody", "tr", "th", "td", "a",
]
_ALLOWED_ATTRS = {"a": ["href", "title"]}

_env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), trim_blocks=True, lstrip_blocks=True)

TEMPLATE_EXEC_SUMMARY = (
    "This dataset contains {n_rows} rows and {n_cols} columns. After automated cleaning and feature "
    "engineering, AutoDS-AI evaluated {n_models} models for a {problem_type} problem"
    "{target_clause}. The best-performing model was **{best_model_name}**, achieving a cross-validated "
    "score of {best_score}. {findings_clause}"
)


def build_template_executive_summary(context: dict[str, Any]) -> str:
    summary = context["summary"]
    leaderboard = context["leaderboard"]
    findings = context["eda_results"].get("notable_findings", [])
    target_clause = f" with target `{context['target_column']}`" if context.get("target_column") else ""
    findings_clause = f"Key observations: {findings[0]}" if findings else ""
    best_score = leaderboard[0]["cv_score"] if leaderboard else "N/A"
    return TEMPLATE_EXEC_SUMMARY.format(
        n_rows=summary["n_rows"],
        n_cols=summary["n_cols"],
        n_models=len(leaderboard),
        problem_type=context["problem_type"],
        target_clause=target_clause,
        best_model_name=context.get("best_model_name", "N/A"),
        best_score=best_score,
        findings_clause=findings_clause,
    )


def render_report(context: dict[str, Any], executive_summary: str | None = None) -> tuple[str, str]:
    template = _env.get_template("report.md.j2")

    exec_summary = executive_summary or build_template_executive_summary(context)

    render_context = {
        **context,
        "executive_summary": exec_summary,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics_json": json.dumps(context.get("metrics", {}), indent=2, default=str),
        "shap_global": context.get("shap_results", {}).get("global_importance", []),
    }

    markdown_text = template.render(**render_context)
    raw_html = markdown_lib.markdown(markdown_text, extensions=["tables"])
    html_text = bleach.clean(raw_html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, strip=True)
    return markdown_text, html_text
