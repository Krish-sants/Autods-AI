import logging
from pathlib import Path

from app.agents.base import safe_llm_call
from app.graph.state import PipelineState
from app.ml.report import build_template_executive_summary, render_docx, render_pdf, render_report
from app.storage.run_store import save_bytes, save_text

logger = logging.getLogger(__name__)


async def report_agent(state: PipelineState) -> PipelineState:
    try:
        leaderboard = state.get("leaderboard", [])
        best_model_name = next(
            (row["display_name"] for row in leaderboard if row["model_id"] == state.get("best_model_id")),
            state.get("best_model_id", "N/A"),
        )

        context = {
            "filename": Path(state["dataset_path"]).name,
            "summary": state["dataset_summary"],
            "eda_results": state["eda_results"],
            "cleaning_report": state["cleaning_report"],
            "problem_type": state["problem_type"],
            "target_column": state.get("target_column"),
            "leaderboard": leaderboard,
            "best_model_name": best_model_name,
            "metrics": state.get("metrics", {}),
            "shap_results": state.get("shap_results", {}),
        }

        fallback_summary = build_template_executive_summary(context)
        prompt = (
            "You are a senior data scientist writing the executive summary of an automated data-science "
            "report for a non-technical business stakeholder. Write 3-4 sentences covering: what the data "
            "shows, what model was chosen and how well it performs, and one practical recommendation.\n\n"
            f"Dataset: {context['summary']['n_rows']} rows, {context['summary']['n_cols']} columns.\n"
            f"Problem type: {context['problem_type']}, target: {context['target_column']}\n"
            f"Best model: {best_model_name}\n"
            f"Metrics: {context['metrics']}\n"
            f"EDA findings: {context['eda_results'].get('notable_findings')}\n"
        )
        executive_summary, used_llm = await safe_llm_call(prompt, fallback_summary)

        markdown_text, html_text = render_report(context, executive_summary=executive_summary)

        md_path = save_text(state["run_id"], "report.md", markdown_text)
        html_path = save_text(state["run_id"], "report.html", html_text)

        report_paths: dict = {"md": str(md_path), "html": str(html_path)}

        # PDF
        try:
            pdf_bytes = render_pdf(html_text)
            pdf_path = save_bytes(state["run_id"], "report.pdf", pdf_bytes)
            report_paths["pdf"] = str(pdf_path)
        except Exception as pdf_exc:
            logger.warning("PDF export failed (non-fatal): %s", pdf_exc)

        # DOCX
        try:
            docx_bytes = render_docx(markdown_text)
            docx_path = save_bytes(state["run_id"], "report.docx", docx_bytes)
            report_paths["docx"] = str(docx_path)
        except Exception as docx_exc:
            logger.warning("DOCX export failed (non-fatal): %s", docx_exc)

        state["report_paths"] = report_paths
        state["narrative_llm_used"] = state.get("narrative_llm_used", False) or used_llm
        state["executive_summary_source"] = "llm" if used_llm else "template"

        state["current_step"] = "complete"
        state.setdefault("steps_completed", []).append("report")
    except Exception as exc:
        logger.exception("report_agent failed")
        state.setdefault("errors", []).append(f"report_agent: {exc}")
        raise
    return state
