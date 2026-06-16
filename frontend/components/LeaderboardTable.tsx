import type { LeaderboardRow } from "@/lib/types";

export default function LeaderboardTable({
  rows,
  bestModelId,
  selectedModelId,
  onSelect,
}: {
  rows: LeaderboardRow[];
  bestModelId: string | null;
  selectedModelId: string | null;
  onSelect: (modelId: string) => void;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-200">
      <table className="w-full text-sm">
        <thead className="bg-zinc-50 text-left">
          <tr>
            <th className="p-2">Rank</th>
            <th className="p-2">Model</th>
            <th className="p-2">CV Score</th>
            <th className="p-2">Fit Time (s)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.model_id}
              onClick={() => row.cv_score !== null && onSelect(row.model_id)}
              className={`border-t border-zinc-100 cursor-pointer ${
                row.model_id === selectedModelId ? "bg-blue-50" : "hover:bg-zinc-50"
              }`}
            >
              <td className="p-2">{row.rank ?? "—"}</td>
              <td className="p-2 font-medium">
                {row.display_name}
                {row.model_id === bestModelId && (
                  <span className="ml-2 rounded-full bg-green-100 px-2 py-0.5 text-xs text-green-700">best</span>
                )}
              </td>
              <td className="p-2">{row.cv_score !== null ? row.cv_score.toFixed(4) : row.error ?? "failed"}</td>
              <td className="p-2">{row.fit_time_s ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
