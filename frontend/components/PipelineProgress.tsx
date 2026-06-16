"use client";

import { Check, Loader2, XCircle } from "lucide-react";
import type { RunStatus } from "@/lib/types";

const STEPS = [
  { key: "understanding", label: "Understanding dataset" },
  { key: "eda", label: "Exploratory analysis" },
  { key: "cleaning", label: "Cleaning data" },
  { key: "target_detection", label: "Detecting target column" },
  { key: "feature_engineering", label: "Engineering features" },
  { key: "training", label: "Training models" },
  { key: "evaluation", label: "Evaluating models" },
  { key: "explainability", label: "Explaining predictions" },
  { key: "report", label: "Writing report" },
];

export default function PipelineProgress({ status }: { status: RunStatus }) {
  const currentIndex = STEPS.findIndex((s) => s.key === status.current_step);
  const failed = status.status === "failed";

  return (
    <div className="w-full max-w-md flex flex-col gap-3">
      {STEPS.map((step, i) => {
        const isDone =
          status.current_step === "complete" || i < currentIndex || (i === currentIndex && status.status === "awaiting_target_confirmation");
        const isCurrent = i === currentIndex && !isDone;
        const isFailedHere = failed && i === currentIndex;

        return (
          <div key={step.key} className="flex items-center gap-3">
            <div
              className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs ${
                isFailedHere
                  ? "bg-red-100 text-red-600"
                  : isDone
                  ? "bg-green-100 text-green-600"
                  : isCurrent
                  ? "bg-blue-100 text-blue-600"
                  : "bg-zinc-100 text-zinc-400"
              }`}
            >
              {isFailedHere ? (
                <XCircle size={14} />
              ) : isDone ? (
                <Check size={14} />
              ) : isCurrent ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                i + 1
              )}
            </div>
            <span className={isDone || isCurrent ? "text-zinc-900" : "text-zinc-400"}>{step.label}</span>
          </div>
        );
      })}
      {failed && status.error_message && (
        <p className="mt-2 rounded-lg bg-red-50 p-3 text-sm text-red-700">{status.error_message}</p>
      )}
    </div>
  );
}
