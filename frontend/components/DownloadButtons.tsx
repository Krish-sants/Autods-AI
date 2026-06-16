import { Download } from "lucide-react";
import { getDownloadUrl } from "@/lib/api";

const ARTIFACTS: { key: string; label: string }[] = [
  { key: "model", label: "Trained Model (.pkl)" },
  { key: "report-html", label: "Report (.html)" },
  { key: "report-md", label: "Report (.md)" },
];

export default function DownloadButtons({ runId }: { runId: string }) {
  return (
    <div className="flex flex-wrap gap-3">
      {ARTIFACTS.map((a) => (
        <a
          key={a.key}
          href={getDownloadUrl(runId, a.key)}
          className="flex items-center gap-2 rounded-lg border border-zinc-300 px-3 py-2 text-sm font-medium hover:bg-zinc-50"
        >
          <Download size={14} />
          {a.label}
        </a>
      ))}
    </div>
  );
}
