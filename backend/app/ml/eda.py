import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def _fig_to_json(fig: go.Figure) -> dict:
    # plotly's own encoder (not fig.to_plotly_json()) correctly converts numpy
    # arrays/scalars to plain lists/floats so the result is safe to return
    # directly from a FastAPI JSON response.
    return json.loads(pio.to_json(fig))


def _descriptive_stats(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": float(series.mode().iloc[0]) if not series.mode().empty else None,
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "variance": float(series.var()) if len(series) > 1 else 0.0,
            "skewness": float(series.skew()) if len(series) > 2 else 0.0,
            "kurtosis": float(series.kurtosis()) if len(series) > 3 else 0.0,
            "min": float(series.min()),
            "max": float(series.max()),
        }
    return stats


def _outlier_flags(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            flags[col] = {
                "method": "iqr",
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": outlier_count,
                "outlier_pct": round(outlier_count / len(series) * 100, 2),
            }
    return flags


def generate_eda(df: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    numeric_cols = [c for c in summary["numeric_columns"] if c in df.columns]
    categorical_cols = [c for c in summary["categorical_columns"] if c in df.columns]

    figures: dict[str, Any] = {}

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).round(3)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="RdBu",
                zmid=0,
            )
        )
        heatmap.update_layout(title="Correlation Matrix")
        figures["correlation_heatmap"] = _fig_to_json(heatmap)

    histograms = []
    for col in numeric_cols[:8]:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", marginal="box")
        histograms.append({"column": col, "figure": _fig_to_json(fig)})
    figures["histograms"] = histograms

    count_plots = []
    for col in categorical_cols[:8]:
        top_values = df[col].value_counts().head(15)
        fig = px.bar(x=top_values.index.astype(str), y=top_values.values, title=f"Counts of {col}")
        fig.update_layout(xaxis_title=col, yaxis_title="count")
        count_plots.append({"column": col, "figure": _fig_to_json(fig)})
    figures["count_plots"] = count_plots

    stats = _descriptive_stats(df, numeric_cols)
    outliers = _outlier_flags(df, numeric_cols)

    notable_findings: list[str] = []
    if "correlation_heatmap" in figures and len(numeric_cols) >= 2:
        corr_abs = df[numeric_cols].corr(numeric_only=True).abs()
        corr_array = corr_abs.to_numpy(dtype=float, copy=True)
        np.fill_diagonal(corr_array, 0)
        if corr_array.size:
            max_idx = np.unravel_index(np.argmax(corr_array), corr_array.shape)
            a, b = corr_abs.index[max_idx[0]], corr_abs.columns[max_idx[1]]
            val = corr_array[max_idx]
            if val > 0.5:
                notable_findings.append(f"Strong correlation ({val:.2f}) between '{a}' and '{b}'.")
    if summary.get("has_missing"):
        worst = max(summary["columns"], key=lambda c: c["missing_pct"])
        if worst["missing_pct"] > 0:
            notable_findings.append(f"Column '{worst['name']}' has {worst['missing_pct']}% missing values.")
    if outliers:
        worst_col = max(outliers, key=lambda k: outliers[k]["outlier_pct"])
        notable_findings.append(
            f"Column '{worst_col}' has {outliers[worst_col]['outlier_pct']}% potential outliers (IQR method)."
        )
    if summary.get("duplicate_count", 0) > 0:
        notable_findings.append(f"Found {summary['duplicate_count']} duplicate rows.")
    if not notable_findings:
        notable_findings.append("No major data quality issues detected in this dataset.")

    return {
        "figures": figures,
        "descriptive_stats": stats,
        "outlier_flags": outliers,
        "notable_findings": notable_findings,
    }
