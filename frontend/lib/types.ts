export interface UploadResponse {
  run_id: string;
  filename: string;
  format: string;
  size_bytes: number;
  status: string;
}

export interface StartRunResponse {
  run_id: string;
  status: string;
  message: string;
}

export interface ConfirmTargetResponse {
  run_id: string;
  status: string;
  target_column: string;
  problem_type: string;
}

export interface RunStatus {
  run_id: string;
  status:
    | "uploaded"
    | "running"
    | "awaiting_target_confirmation"
    | "complete"
    | "failed"
    | "cancelled";
  current_step: string;
  candidate_target: string | null;
  candidate_target_confidence: number | null;
  candidate_target_reasoning: string | null;
  target_column: string | null;
  problem_type: string | null;
  best_model_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface RunListItem {
  run_id: string;
  filename: string;
  status: string;
  current_step: string;
  created_at: string;
}

export interface DatasetColumn {
  name: string;
  dtype: string;
  kind: string;
  missing_count: number;
  missing_pct: number;
  cardinality: number;
}

export interface DatasetSummary {
  n_rows: number;
  n_cols: number;
  numeric_columns: string[];
  categorical_columns: string[];
  datetime_columns: string[];
  boolean_columns: string[];
  duplicate_count: number;
  missing_total: number;
  has_missing: boolean;
  memory_mb: number;
  columns: DatasetColumn[];
  potential_targets: string[];
  sample_rows: Record<string, unknown>[];
  narrative?: string;
}

export interface PlotlyFigureJSON {
  data: unknown[];
  layout: Record<string, unknown>;
}

export interface EDAResults {
  figures: {
    correlation_heatmap?: PlotlyFigureJSON;
    histograms: { column: string; figure: PlotlyFigureJSON }[];
    count_plots: { column: string; figure: PlotlyFigureJSON }[];
  };
  descriptive_stats: Record<string, Record<string, number>>;
  outlier_flags: Record<string, { outlier_count: number; outlier_pct: number }>;
  notable_findings: string[];
}

export interface LeaderboardRow {
  rank: number | null;
  model_id: string;
  display_name: string;
  cv_score: number | null;
  fit_time_s: number | null;
  best_params: Record<string, unknown>;
  error?: string;
}

export interface LeaderboardResponse {
  leaderboard: LeaderboardRow[];
  best_model_id: string | null;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision_weighted: number;
  recall_weighted: number;
  f1_weighted: number;
  confusion_matrix: number[][];
  labels: string[];
  roc_auc: number | null;
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  rmse: number;
  r2: number;
  adjusted_r2: number | null;
  mape: number | null;
}

export type Metrics = ClassificationMetrics | RegressionMetrics | Record<string, unknown>;

export interface FeatureImportanceResponse {
  model_id: string | null;
  importances: { feature: string; importance: number }[];
  plotly_figure: PlotlyFigureJSON | null;
}

export interface ShapContribution {
  feature: string;
  shap_value: number;
}

export interface ShapLocalExample {
  row_index: number;
  base_value: number | null;
  contributions: ShapContribution[];
}

export interface ShapResults {
  available: boolean;
  reason?: string;
  error?: string;
  global_importance: { feature: string; mean_abs_shap: number }[];
  local_examples: ShapLocalExample[];
  plain_english_explanation?: string;
}

export interface ReportResponse {
  markdown: string;
  html: string;
  executive_summary_source: "llm" | "template";
}

export interface NotReadyResponse {
  detail: string;
  status: string;
}
