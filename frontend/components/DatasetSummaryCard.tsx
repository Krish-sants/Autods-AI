import type { DatasetSummary } from "@/lib/types";

export default function DatasetSummaryCard({ summary }: { summary: DatasetSummary }) {
  return (
    <div className="flex flex-col gap-6">
      {summary.narrative && (
        <p className="rounded-xl bg-blue-50 p-4 text-sm text-blue-900">{summary.narrative}</p>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <Stat label="Rows" value={summary.n_rows.toLocaleString()} />
        <Stat label="Columns" value={summary.n_cols.toString()} />
        <Stat label="Numeric" value={summary.numeric_columns.length.toString()} />
        <Stat label="Categorical" value={summary.categorical_columns.length.toString()} />
        <Stat label="Datetime" value={summary.datetime_columns.length.toString()} />
        <Stat label="Duplicates" value={summary.duplicate_count.toString()} />
        <Stat label="Missing values" value={summary.missing_total.toLocaleString()} />
        <Stat label="Memory" value={`${summary.memory_mb} MB`} />
      </div>

      <div className="overflow-x-auto rounded-xl border border-zinc-200">
        <table className="w-full text-sm">
          <thead className="bg-zinc-50 text-left">
            <tr>
              <th className="p-2">Column</th>
              <th className="p-2">Type</th>
              <th className="p-2">Missing %</th>
              <th className="p-2">Cardinality</th>
            </tr>
          </thead>
          <tbody>
            {summary.columns.map((col) => (
              <tr key={col.name} className="border-t border-zinc-100">
                <td className="p-2 font-medium">{col.name}</td>
                <td className="p-2 capitalize text-zinc-500">{col.kind}</td>
                <td className="p-2">{col.missing_pct}%</td>
                <td className="p-2">{col.cardinality}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-zinc-200 p-3">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}
