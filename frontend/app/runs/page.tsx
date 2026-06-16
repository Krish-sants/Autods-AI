"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listRuns } from "@/lib/api";
import type { RunListItem } from "@/lib/types";

export default function RunsHistoryPage() {
  const [runs, setRuns] = useState<RunListItem[]>([]);

  useEffect(() => {
    listRuns().then(setRuns);
  }, []);

  return (
    <div className="flex flex-1 flex-col gap-6 px-6 py-10 max-w-3xl mx-auto w-full">
      <h1 className="text-xl font-semibold">Run History</h1>
      {runs.length === 0 ? (
        <p className="text-sm text-zinc-400">No runs yet. Upload a dataset to get started.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {runs.map((run) => (
            <Link
              key={run.run_id}
              href={run.status === "complete" ? `/runs/${run.run_id}/results` : `/runs/${run.run_id}`}
              className="flex items-center justify-between rounded-lg border border-zinc-200 px-4 py-3 hover:bg-zinc-50"
            >
              <div>
                <p className="font-medium">{run.filename}</p>
                <p className="text-xs text-zinc-500">{new Date(run.created_at).toLocaleString()}</p>
              </div>
              <span className="text-sm capitalize text-zinc-600">{run.status.replace(/_/g, " ")}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
