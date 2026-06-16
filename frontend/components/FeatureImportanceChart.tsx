import type { FeatureImportanceResponse } from "@/lib/types";
import PlotlyChart from "./PlotlyChart";

export default function FeatureImportanceChart({ data }: { data: FeatureImportanceResponse }) {
  if (!data.plotly_figure || data.importances.length === 0) {
    return <p className="text-sm text-zinc-500">Feature importance is not available for this model.</p>;
  }
  return <PlotlyChart figure={data.plotly_figure} />;
}
