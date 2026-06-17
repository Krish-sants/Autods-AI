"use client";

import { useState } from "react";
import type { ShapResults } from "@/lib/types";

export default function SHAPExplorer({ shap }: { shap: ShapResults }) {
  const [selectedIdx, setSelectedIdx] = useState(0);

  if (!shap.available) {
    return <p className="text-sm text-zinc-500">{shap.reason || shap.error || "SHAP explanations are not available for this run."}</p>;
  }

  const maxAbs = Math.max(...shap.global_importance.map((f) => f.mean_abs_shap), 1e-9);
  const example = shap.local_examples[selectedIdx];

  return (
    <div className="flex flex-col gap-8">
      {shap.plain_english_explanation && (
        <p className="rounded-xl bg-blue-50 p-4 text-sm text-blue-900">{shap.plain_english_explanation}</p>
      )}

      <div>
        <p className="text-sm font-medium mb-3">Global Feature Importance (mean |SHAP value|)</p>
        <div className="flex flex-col gap-2">
          {shap.global_importance.slice(0, 15).map((f) => (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="w-24 sm:w-40 truncate text-sm text-zinc-600">{f.feature}</span>
              <div className="flex-1 h-3 rounded-full bg-zinc-100">
                <div
                  className="h-3 rounded-full bg-blue-500"
                  style={{ width: `${(f.mean_abs_shap / maxAbs) * 100}%` }}
                />
              </div>
              <span className="w-16 text-right text-xs text-zinc-500">{f.mean_abs_shap.toFixed(4)}</span>
            </div>
          ))}
        </div>
      </div>

      {shap.local_examples.length > 0 && example && (
        <div>
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between mb-3">
            <p className="text-sm font-medium">Individual Prediction Explanation</p>
            <select
              value={selectedIdx}
              onChange={(e) => setSelectedIdx(Number(e.target.value))}
              className="rounded-lg border border-zinc-300 px-2 py-1 text-sm"
            >
              {shap.local_examples.map((ex, i) => (
                <option key={ex.row_index} value={i}>
                  Sample row {ex.row_index}
                </option>
              ))}
            </select>
          </div>
          <div className="flex flex-col gap-2">
            {example.contributions.map((c) => (
              <div key={c.feature} className="flex items-center gap-3 text-sm">
                <span className="w-24 sm:w-40 truncate text-zinc-600">{c.feature}</span>
                <div className="flex-1 h-3 rounded-full bg-zinc-100 relative">
                  <div
                    className={`absolute h-3 rounded-full ${c.shap_value >= 0 ? "bg-green-500 left-1/2" : "bg-red-500 right-1/2"}`}
                    style={{ width: `${Math.min(Math.abs(c.shap_value) * 40, 50)}%` }}
                  />
                </div>
                <span className="w-16 text-right text-xs text-zinc-500">{c.shap_value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
