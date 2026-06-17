"use client";

import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import toast from "react-hot-toast";
import { downloadArtifact } from "@/lib/api";

const ARTIFACTS: { key: string; label: string; filename: string; group: string }[] = [
  { key: "model", label: "Model (.pkl)", filename: "model.pkl", group: "model" },
  { key: "report-html", label: "Report (.html)", filename: "report.html", group: "report" },
  { key: "report-pdf", label: "Report (.pdf)", filename: "report.pdf", group: "report" },
  { key: "report-docx", label: "Report (.docx)", filename: "report.docx", group: "report" },
  { key: "report-md", label: "Report (.md)", filename: "report.md", group: "report" },
];

export default function DownloadButtons({ runId }: { runId: string }) {
  const [pending, setPending] = useState<string | null>(null);

  async function handleDownload(key: string, filename: string) {
    setPending(key);
    try {
      await downloadArtifact(runId, key, filename);
    } catch {
      toast.error(`Failed to download ${filename}`);
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      {ARTIFACTS.map((a) => (
        <button
          key={a.key}
          onClick={() => handleDownload(a.key, a.filename)}
          disabled={pending === a.key}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-300 px-3 py-1.5 text-xs font-medium hover:bg-zinc-50 disabled:opacity-50"
        >
          {pending === a.key ? <Loader2 size={12} className="animate-spin" /> : <Download size={12} />}
          {a.label}
        </button>
      ))}
    </div>
  );
}
