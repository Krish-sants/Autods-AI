"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { confirmTarget, getStatus, getSummary } from "@/lib/api";
import type { RunStatus } from "@/lib/types";
import PipelineProgress from "@/components/PipelineProgress";
import TargetConfirmation from "@/components/TargetConfirmation";

export default function RunPage() {
  const { runId } = useParams<{ runId: string }>();
  const router = useRouter();
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const confirmedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const data = await getStatus(runId);
        if (cancelled) return;
        setStatus(data);

        if (data.status === "awaiting_target_confirmation" && columns.length === 0) {
          const summary = await getSummary(runId);
          if (summary && !cancelled) {
            setColumns([
              ...summary.numeric_columns,
              ...summary.categorical_columns,
              ...summary.datetime_columns,
              ...summary.boolean_columns,
            ]);
          }
        }

        if (data.status === "complete") {
          router.push(`/runs/${runId}/results`);
          return;
        }
        if (data.status === "failed") {
          toast.error(data.error_message || "Pipeline run failed");
          return;
        }
      } catch {
        // transient network error, keep polling
      }
      if (!cancelled) setTimeout(poll, 2000);
    }

    poll();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId]);

  async function handleConfirm(targetColumn: string | null) {
    if (confirmedRef.current) return;
    confirmedRef.current = true;
    try {
      await confirmTarget(runId, targetColumn ?? "");
      toast.success("Target confirmed — continuing pipeline");
      setStatus((prev) => (prev ? { ...prev, status: "running" } : prev));
    } catch {
      confirmedRef.current = false;
      toast.error("Failed to confirm target column");
    }
  }

  if (!status) {
    return <div className="flex flex-1 items-center justify-center text-zinc-400">Loading run...</div>;
  }

  return (
    <div className="flex flex-1 flex-col items-center gap-10 py-16 px-6">
      <h1 className="text-xl font-semibold">Run {runId.slice(0, 8)}</h1>
      <PipelineProgress status={status} />
      {status.status === "awaiting_target_confirmation" && (
        <TargetConfirmation
          candidateTarget={status.candidate_target}
          confidence={status.candidate_target_confidence}
          reasoning={status.candidate_target_reasoning}
          columns={columns}
          onConfirm={handleConfirm}
        />
      )}
    </div>
  );
}
