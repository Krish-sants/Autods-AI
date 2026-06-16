import type { EDAResults } from "@/lib/types";
import PlotlyChart from "./PlotlyChart";

export default function EDACharts({ eda }: { eda: EDAResults }) {
  return (
    <div className="flex flex-col gap-6">
      {eda.notable_findings.length > 0 && (
        <ul className="rounded-xl bg-amber-50 p-4 text-sm text-amber-900 list-disc pl-5 space-y-1">
          {eda.notable_findings.map((finding, i) => (
            <li key={i}>{finding}</li>
          ))}
        </ul>
      )}

      {eda.figures.correlation_heatmap && (
        <PlotlyChart figure={eda.figures.correlation_heatmap} title="Correlation Matrix" />
      )}

      {eda.figures.histograms.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {eda.figures.histograms.map((h) => (
            <PlotlyChart key={h.column} figure={h.figure} title={`Distribution: ${h.column}`} />
          ))}
        </div>
      )}

      {eda.figures.count_plots.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {eda.figures.count_plots.map((c) => (
            <PlotlyChart key={c.column} figure={c.figure} title={`Counts: ${c.column}`} />
          ))}
        </div>
      )}
    </div>
  );
}
