"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";

interface Props {
  candidateTarget: string | null;
  confidence: number | null;
  reasoning: string | null;
  columns: string[];
  onConfirm: (targetColumn: string | null) => Promise<void>;
}

export default function TargetConfirmation({ candidateTarget, confidence, reasoning, columns, onConfirm }: Props) {
  const [selected, setSelected] = useState<string>(candidateTarget ?? "");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    setSubmitting(true);
    try {
      await onConfirm(selected || null);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="w-full max-w-md rounded-xl border border-zinc-200 p-6 flex flex-col gap-4">
      <div>
        <h3 className="font-semibold">Confirm the target column</h3>
        <p className="text-sm text-zinc-500 mt-1">
          AutoDS-AI thinks the goal is to predict{" "}
          <span className="font-medium text-zinc-900">{candidateTarget ?? "(no clear target found)"}</span>
          {confidence !== null && <> ({Math.round(confidence * 100)}% confidence)</>}.
        </p>
        {reasoning && <p className="text-sm text-zinc-500 mt-1 italic">&ldquo;{reasoning}&rdquo;</p>}
      </div>

      <label className="text-sm font-medium">Target column</label>
      <select
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
        className="rounded-lg border border-zinc-300 px-3 py-2 text-sm"
      >
        <option value="">No target — find clusters/segments instead</option>
        {columns.map((col) => (
          <option key={col} value={col}>
            {col}
          </option>
        ))}
      </select>

      <button
        onClick={handleSubmit}
        disabled={submitting}
        className="flex items-center justify-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
      >
        {submitting && <Loader2 size={14} className="animate-spin" />}
        Confirm & Continue
      </button>
    </div>
  );
}
