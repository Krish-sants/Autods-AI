"use client";

import type { ForecastResults } from "@/lib/types";
import PlotlyChart from "./PlotlyChart";

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 p-3">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}

export default function ForecastChart({ forecast }: { forecast: ForecastResults }) {
  const { metrics, future_forecast, freq, target_col, date_col, plotly_figure } = forecast;

  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-xl border border-zinc-200 p-4">
        <p className="text-sm font-medium mb-1">Forecast Configuration</p>
        <p className="text-sm text-zinc-500">
          Target: <span className="font-medium text-zinc-800">{target_col}</span> &nbsp;·&nbsp;
          Date column: <span className="font-medium text-zinc-800">{date_col}</span> &nbsp;·&nbsp;
          Frequency: <span className="font-medium text-zinc-800">{freq}</span> &nbsp;·&nbsp;
          Forecasting <span className="font-medium text-zinc-800">{future_forecast.length}</span> future periods
        </p>
      </div>

      <div className="grid grid-cols-3 gap-2 sm:gap-4">
        <MetricCard label="MAE (hold-out)" value={metrics.mae.toFixed(3)} />
        <MetricCard label="RMSE (hold-out)" value={metrics.rmse.toFixed(3)} />
        <MetricCard label="MAPE (hold-out)" value={`${metrics.mape.toFixed(1)}%`} />
      </div>

      {plotly_figure && (
        <div className="rounded-xl border border-zinc-200 p-4">
          <PlotlyChart figure={plotly_figure} />
        </div>
      )}

      {future_forecast.length > 0 && (
        <div>
          <p className="text-sm font-medium mb-2">Future Forecast ({future_forecast.length} periods)</p>
          <div className="overflow-x-auto rounded-xl border border-zinc-200">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-200 bg-zinc-50">
                  <th className="px-4 py-2 text-left font-medium text-zinc-600">Date</th>
                  <th className="px-4 py-2 text-right font-medium text-zinc-600">Forecast</th>
                  <th className="px-4 py-2 text-right font-medium text-zinc-600">Lower 95%</th>
                  <th className="px-4 py-2 text-right font-medium text-zinc-600">Upper 95%</th>
                </tr>
              </thead>
              <tbody>
                {future_forecast.slice(0, 30).map((row, i) => (
                  <tr key={i} className="border-b border-zinc-100 last:border-0 hover:bg-zinc-50">
                    <td className="px-4 py-2 font-mono text-xs">{row.ds}</td>
                    <td className="px-4 py-2 text-right font-semibold">{row.yhat.toFixed(3)}</td>
                    <td className="px-4 py-2 text-right text-zinc-500">{row.yhat_lower.toFixed(3)}</td>
                    <td className="px-4 py-2 text-right text-zinc-500">{row.yhat_upper.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
