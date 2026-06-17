import type { ReportResponse } from "@/lib/types";

export default function ReportViewer({ report }: { report: ReportResponse }) {
  return (
    <div className="flex flex-col gap-4">
      <span
        className={`self-start rounded-full px-3 py-1 text-xs font-medium ${
          report.executive_summary_source === "llm" ? "bg-purple-100 text-purple-700" : "bg-zinc-100 text-zinc-600"
        }`}
      >
        {report.executive_summary_source === "llm" ? "AI-written summary" : "Template summary (LLM unavailable for this run)"}
      </span>
      <div className="overflow-x-auto rounded-xl border border-zinc-200">
        <div
          className="prose prose-zinc max-w-none p-4 sm:p-6 [&_table]:w-full [&_th]:text-left [&_th]:p-2 [&_td]:p-2 [&_table]:border [&_th]:border [&_td]:border"
          dangerouslySetInnerHTML={{ __html: report.html }}
        />
      </div>
    </div>
  );
}
