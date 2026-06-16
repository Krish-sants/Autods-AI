"use client";

import dynamic from "next/dynamic";
import type { PlotlyFigureJSON } from "@/lib/types";

// react-plotly.js touches `window` at import time, so it must be loaded client-only.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export default function PlotlyChart({ figure, title }: { figure: PlotlyFigureJSON; title?: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 p-2">
      {title && <p className="px-2 pt-1 text-sm font-medium text-zinc-600">{title}</p>}
      <Plot
        data={figure.data as never}
        layout={{ ...figure.layout, autosize: true, margin: { t: 30, l: 50, r: 20, b: 40 } } as never}
        useResizeHandler
        style={{ width: "100%", height: "360px" }}
        config={{ displayModeBar: false, responsive: true }}
      />
    </div>
  );
}
