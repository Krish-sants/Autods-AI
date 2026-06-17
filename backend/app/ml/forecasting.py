"""Prophet-based time-series forecasting module."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastOutput:
    model: Any                         # fitted Prophet model
    forecast_df: pd.DataFrame          # Prophet forecast DataFrame
    history_df: pd.DataFrame           # ds/y history used for fitting
    date_col: str                       # name of the datetime column in original df
    target_col: str
    freq: str                          # inferred pandas freq string
    metrics: dict[str, float]          # in-sample MAE/RMSE/MAPE on held-out tail
    plotly_figure: dict[str, Any]      # serialized Plotly figure


def _find_datetime_col(df: pd.DataFrame) -> str | None:
    """Return the name of the most suitable datetime column, or None.

    Checks dtype-typed datetime cols first, then falls back to string cols
    where ≥90% of non-null values parse as dates.
    """
    # Prefer already-typed datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # Fallback: datetime index
    if pd.api.types.is_datetime64_any_dtype(df.index):
        return df.index.name or "__index__"
    # Last resort: string columns that look like dates
    for col in df.columns:
        if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
            continue
        sample = df[col].dropna().head(30)
        if len(sample) < 2:
            continue
        try:
            if pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean() >= 0.9:
                return col
        except Exception:
            continue
    return None


def _infer_freq(series: pd.Series) -> str:
    """Infer pandas frequency string from a sorted datetime Series."""
    if len(series) < 2:
        return "D"
    diffs = series.sort_values().diff().dropna()
    median_days = diffs.dt.total_seconds().median() / 86400
    if median_days < 0.1:
        return "h"   # pandas 2.2+: "H" removed
    if median_days < 1.5:
        return "D"
    if median_days < 10:
        return "W"
    if median_days < 35:
        return "MS"
    if median_days < 100:
        return "QS"
    return "YS"      # pandas 2.2+: "AS" removed


def run_forecast(
    df: pd.DataFrame,
    target_column: str,
    periods: int = 30,
) -> ForecastOutput:
    """Fit a Prophet model and return forecast + diagnostics."""
    try:
        from prophet import Prophet  # lazy import — heavy library
    except ImportError as e:
        raise ImportError("prophet is required for forecasting: pip install prophet") from e

    import plotly.graph_objects as go
    import plotly.io as pio

    # Locate datetime column
    date_col = _find_datetime_col(df)
    if date_col is None:
        raise ValueError("No datetime column found for forecasting.")

    if date_col == "__index__":
        prophet_df = pd.DataFrame({"ds": df.index, "y": df[target_column].values})
    else:
        prophet_df = df[[date_col, target_column]].rename(columns={date_col: "ds", target_column: "y"})

    prophet_df = prophet_df.dropna()
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    freq = _infer_freq(prophet_df["ds"])

    # Hold out last 20% for validation metrics
    n_val = max(1, int(len(prophet_df) * 0.2))
    train_df = prophet_df.iloc[:-n_val]
    val_df = prophet_df.iloc[-n_val:]

    model = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        interval_width=0.95,
    )
    model.fit(train_df)

    # Validation predictions
    val_forecast = model.predict(val_df[["ds"]])
    y_true = val_df["y"].values
    y_pred = val_forecast["yhat"].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)))) * 100

    # Refit on full history and predict future
    full_model = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        interval_width=0.95,
    )
    full_model.fit(prophet_df)
    future = full_model.make_future_dataframe(periods=periods, freq=freq)
    forecast = full_model.predict(future)

    # Build Plotly figure
    history_len = len(prophet_df)
    hist_trace = go.Scatter(
        x=forecast["ds"].iloc[:history_len].tolist(),
        y=prophet_df["y"].tolist(),
        mode="markers",
        name="Actual",
        marker={"color": "#6366f1", "size": 4},
    )
    forecast_trace = go.Scatter(
        x=forecast["ds"].tolist(),
        y=forecast["yhat"].tolist(),
        mode="lines",
        name="Forecast",
        line={"color": "#f59e0b"},
    )
    upper_trace = go.Scatter(
        x=forecast["ds"].tolist(),
        y=forecast["yhat_upper"].tolist(),
        mode="lines",
        fill=None,
        line={"width": 0},
        showlegend=False,
    )
    lower_trace = go.Scatter(
        x=forecast["ds"].tolist(),
        y=forecast["yhat_lower"].tolist(),
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(245,158,11,0.15)",
        line={"width": 0},
        name="95% CI",
    )
    fig = go.Figure(data=[hist_trace, upper_trace, lower_trace, forecast_trace])
    fig.update_layout(
        title=f"Forecast: {target_column}",
        xaxis_title="Date",
        yaxis_title=target_column,
        hovermode="x unified",
    )
    plotly_figure = json.loads(pio.to_json(fig))

    # Serialize forecast dataframe as records (future portion only)
    future_forecast = forecast.iloc[history_len:][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_forecast["ds"] = future_forecast["ds"].dt.strftime("%Y-%m-%d")

    return ForecastOutput(
        model=full_model,
        forecast_df=future_forecast,
        history_df=prophet_df,
        date_col=date_col,
        target_col=target_column,
        freq=freq,
        metrics={"mae": mae, "rmse": rmse, "mape": mape},
        plotly_figure=plotly_figure,
    )


def serialize_forecast_results(output: ForecastOutput) -> dict[str, Any]:
    """Convert ForecastOutput to a JSON-safe dict for state.json."""
    future_records = output.forecast_df.to_dict(orient="records")
    history_records = [
        {"ds": str(row["ds"])[:10], "y": float(row["y"])}
        for _, row in output.history_df.iterrows()
    ]
    return {
        "target_col": output.target_col,
        "date_col": output.date_col,
        "freq": output.freq,
        "metrics": output.metrics,
        "future_forecast": future_records,
        "history": history_records,
        "plotly_figure": output.plotly_figure,
    }
