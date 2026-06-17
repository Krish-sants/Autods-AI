"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import * as Tabs from "@radix-ui/react-tabs";
import {
  getEda,
  getFeatureImportance,
  getForecast,
  getLeaderboard,
  getMetrics,
  getReport,
  getShap,
  getSummary,
} from "@/lib/api";
import type {
  DatasetSummary,
  EDAResults,
  FeatureImportanceResponse,
  ForecastResults,
  LeaderboardResponse,
  Metrics,
  ReportResponse,
  ShapResults,
} from "@/lib/types";
import DatasetSummaryCard from "@/components/DatasetSummaryCard";
import EDACharts from "@/components/EDACharts";
import LeaderboardTable from "@/components/LeaderboardTable";
import MetricsPanel from "@/components/MetricsPanel";
import FeatureImportanceChart from "@/components/FeatureImportanceChart";
import SHAPExplorer from "@/components/SHAPExplorer";
import ReportViewer from "@/components/ReportViewer";
import ForecastChart from "@/components/ForecastChart";
import DownloadButtons from "@/components/DownloadButtons";

const BASE_TABS = [
  { key: "summary", label: "Summary" },
  { key: "eda", label: "EDA" },
  { key: "leaderboard", label: "Leaderboard" },
  { key: "metrics", label: "Metrics" },
  { key: "importance", label: "Feature Importance" },
  { key: "shap", label: "SHAP" },
  { key: "report", label: "Report" },
];

const FORECAST_TABS = [
  { key: "summary", label: "Summary" },
  { key: "eda", label: "EDA" },
  { key: "forecast", label: "Forecast" },
  { key: "report", label: "Report" },
];

export default function ResultsPage() {
  const { runId } = useParams<{ runId: string }>();

  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [eda, setEda] = useState<EDAResults | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardResponse | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportanceResponse | null>(null);
  const [shap, setShap] = useState<ShapResults | null>(null);
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResults | null>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [problemType, setProblemType] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const [s, e, l, fi, sh, r, fc] = await Promise.all([
        getSummary(runId),
        getEda(runId),
        getLeaderboard(runId),
        getFeatureImportance(runId),
        getShap(runId),
        getReport(runId),
        getForecast(runId),
      ]);
      setSummary(s);
      setEda(e);
      setLeaderboard(l);
      setFeatureImportance(fi);
      setShap(sh);
      setReport(r);
      setForecast(fc);
      if (l?.best_model_id) setSelectedModelId(l.best_model_id);
      if (fc) setProblemType("forecasting");
    })();
  }, [runId]);

  useEffect(() => {
    if (!selectedModelId) return;
    getMetrics(runId, selectedModelId).then(setMetrics);
  }, [runId, selectedModelId]);

  const isForecasting = problemType === "forecasting" || forecast !== null;
  const TABS = isForecasting ? FORECAST_TABS : BASE_TABS;

  return (
    <div className="flex flex-1 flex-col gap-6 px-6 py-10 max-w-5xl mx-auto w-full">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-xl font-semibold">Results</h1>
        <DownloadButtons runId={runId} />
      </div>

      <Tabs.Root defaultValue="summary" className="flex flex-col gap-6">
        <Tabs.List className="flex gap-1 border-b border-zinc-200 overflow-x-auto">
          {TABS.map((tab) => (
            <Tabs.Trigger
              key={tab.key}
              value={tab.key}
              className="px-3 py-2 text-sm font-medium text-zinc-500 border-b-2 border-transparent data-[state=active]:text-black data-[state=active]:border-black whitespace-nowrap"
            >
              {tab.label}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        <Tabs.Content value="summary">
          {summary ? <DatasetSummaryCard summary={summary} /> : <Loading />}
        </Tabs.Content>

        <Tabs.Content value="eda">
          {eda ? <EDACharts eda={eda} /> : <Loading />}
        </Tabs.Content>

        {/* Forecasting tab — shown only for time-series runs */}
        <Tabs.Content value="forecast">
          {forecast ? <ForecastChart forecast={forecast} /> : <Loading />}
        </Tabs.Content>

        {/* Supervised pipeline tabs */}
        <Tabs.Content value="leaderboard">
          {leaderboard ? (
            <LeaderboardTable
              rows={leaderboard.leaderboard}
              bestModelId={leaderboard.best_model_id}
              selectedModelId={selectedModelId}
              onSelect={setSelectedModelId}
            />
          ) : (
            <Loading />
          )}
        </Tabs.Content>

        <Tabs.Content value="metrics">
          {metrics ? (
            <div className="flex flex-col gap-4">
              {leaderboard && leaderboard.leaderboard.length > 1 && (
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-zinc-600">Model:</label>
                  <select
                    value={selectedModelId ?? ""}
                    onChange={(e) => setSelectedModelId(e.target.value)}
                    className="rounded-lg border border-zinc-300 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-black"
                  >
                    {leaderboard.leaderboard.map((row) => (
                      <option key={row.model_id} value={row.model_id}>
                        {row.display_name}
                        {row.model_id === leaderboard.best_model_id ? " ★" : ""}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <MetricsPanel metrics={metrics} />
            </div>
          ) : (
            <Loading />
          )}
        </Tabs.Content>

        <Tabs.Content value="importance">
          {featureImportance ? <FeatureImportanceChart data={featureImportance} /> : <Loading />}
        </Tabs.Content>

        <Tabs.Content value="shap">
          {shap ? <SHAPExplorer shap={shap} /> : <Loading />}
        </Tabs.Content>

        <Tabs.Content value="report">
          {report ? <ReportViewer report={report} /> : <Loading />}
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
}

function Loading() {
  return <p className="text-sm text-zinc-400">Loading...</p>;
}
