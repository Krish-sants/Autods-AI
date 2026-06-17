import type { ClassificationMetrics, Metrics, RegressionMetrics } from "@/lib/types";

function isClassification(metrics: Metrics): metrics is ClassificationMetrics {
  return "confusion_matrix" in metrics;
}

function isRegression(metrics: Metrics): metrics is RegressionMetrics {
  return "rmse" in metrics;
}

export default function MetricsPanel({ metrics }: { metrics: Metrics }) {
  if (isClassification(metrics)) {
    return (
      <div className="flex flex-col gap-6">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Stat label="Accuracy" value={pct(metrics.accuracy)} />
          <Stat label="Precision" value={pct(metrics.precision_weighted)} />
          <Stat label="Recall" value={pct(metrics.recall_weighted)} />
          <Stat label="F1 Score" value={pct(metrics.f1_weighted)} />
          {metrics.roc_auc !== null && <Stat label="ROC AUC" value={metrics.roc_auc.toFixed(3)} />}
        </div>

        <div>
          <p className="text-sm font-medium mb-2">Confusion Matrix</p>
          <div className="overflow-x-auto">
          <table className="border border-zinc-200 text-sm">
            <thead>
              <tr>
                <th className="p-2" />
                {metrics.labels.map((l) => (
                  <th key={l} className="p-2 text-zinc-500">
                    pred {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {metrics.confusion_matrix.map((row, i) => (
                <tr key={i}>
                  <td className="p-2 font-medium text-zinc-500">actual {metrics.labels[i]}</td>
                  {row.map((cell, j) => (
                    <td key={j} className="p-2 border border-zinc-100 text-center">
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </div>
      </div>
    );
  }

  if (isRegression(metrics)) {
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="MAE" value={metrics.mae.toFixed(3)} />
        <Stat label="RMSE" value={metrics.rmse.toFixed(3)} />
        <Stat label="R²" value={metrics.r2.toFixed(3)} />
        {metrics.adjusted_r2 !== null && <Stat label="Adjusted R²" value={metrics.adjusted_r2.toFixed(3)} />}
        {metrics.mape !== null && <Stat label="MAPE" value={pct(metrics.mape)} />}
      </div>
    );
  }

  return (
    <pre className="rounded-xl bg-zinc-50 p-4 text-xs overflow-x-auto">{JSON.stringify(metrics, null, 2)}</pre>
  );
}

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 p-3">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}
