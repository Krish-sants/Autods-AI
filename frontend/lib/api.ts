import axios from "axios";
import type {
  ConfirmTargetResponse,
  DatasetSummary,
  EDAResults,
  FeatureImportanceResponse,
  ForecastResults,
  LeaderboardResponse,
  Metrics,
  ReportResponse,
  RunListItem,
  RunStatus,
  ShapResults,
  StartRunResponse,
  UploadResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api/v1";
const PASSWORD_STORAGE_KEY = "autods_app_password";
export const UNAUTHORIZED_EVENT = "autods:unauthorized";

export function getStoredPassword(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(PASSWORD_STORAGE_KEY);
}

export function setStoredPassword(password: string): void {
  window.localStorage.setItem(PASSWORD_STORAGE_KEY, password);
}

export function clearStoredPassword(): void {
  window.localStorage.removeItem(PASSWORD_STORAGE_KEY);
}

export const api = axios.create({ baseURL: API_BASE_URL });

api.interceptors.request.use((config) => {
  const password = getStoredPassword();
  if (password) {
    config.headers = config.headers ?? {};
    config.headers["X-App-Password"] = password;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401) {
      clearStoredPassword();
      if (typeof window !== "undefined") {
        window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
      }
    }
    return Promise.reject(error);
  }
);

// Verifies a candidate password directly (bypassing the stored-password interceptor)
// so the gate UI can check it before persisting it to localStorage.
export async function verifyPassword(password: string): Promise<boolean> {
  try {
    await axios.get(`${API_BASE_URL}/runs`, { headers: { "X-App-Password": password } });
    return true;
  } catch {
    return false;
  }
}

export async function downloadArtifact(runId: string, artifact: string, filename: string): Promise<void> {
  const response = await api.get(`/runs/${runId}/download/${artifact}`, { responseType: "blob" });
  const url = URL.createObjectURL(response.data as Blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export async function uploadDataset(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const { data } = await api.post<UploadResponse>("/datasets/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function uploadFromUrl(url: string): Promise<UploadResponse> {
  const { data } = await api.post<UploadResponse>("/datasets/from-url", { url });
  return data;
}

export async function startRun(runId: string): Promise<StartRunResponse> {
  const { data } = await api.post<StartRunResponse>(`/runs/${runId}/start`);
  return data;
}

export async function getStatus(runId: string): Promise<RunStatus> {
  const { data } = await api.get<RunStatus>(`/runs/${runId}/status`);
  return data;
}

export async function confirmTarget(runId: string, targetColumn: string): Promise<ConfirmTargetResponse> {
  const { data } = await api.post<ConfirmTargetResponse>(`/runs/${runId}/confirm-target`, {
    target_column: targetColumn,
  });
  return data;
}

export async function listRuns(): Promise<RunListItem[]> {
  const { data } = await api.get<RunListItem[]>("/runs");
  return data;
}

export async function getSummary(runId: string): Promise<DatasetSummary | null> {
  const { data, status } = await api.get(`/runs/${runId}/summary`, { validateStatus: () => true });
  return status === 200 ? (data as DatasetSummary) : null;
}

export async function getEda(runId: string): Promise<EDAResults | null> {
  const { data, status } = await api.get(`/runs/${runId}/eda`, { validateStatus: () => true });
  return status === 200 ? (data as EDAResults) : null;
}

export async function getLeaderboard(runId: string): Promise<LeaderboardResponse | null> {
  const { data, status } = await api.get(`/runs/${runId}/leaderboard`, { validateStatus: () => true });
  return status === 200 ? (data as LeaderboardResponse) : null;
}

export async function getMetrics(runId: string, modelId: string): Promise<Metrics | null> {
  const { data, status } = await api.get(`/runs/${runId}/metrics/${modelId}`, { validateStatus: () => true });
  return status === 200 ? (data as Metrics) : null;
}

export async function getFeatureImportance(runId: string): Promise<FeatureImportanceResponse | null> {
  const { data, status } = await api.get(`/runs/${runId}/feature-importance`, { validateStatus: () => true });
  return status === 200 ? (data as FeatureImportanceResponse) : null;
}

export async function getShap(runId: string): Promise<ShapResults | null> {
  const { data, status } = await api.get(`/runs/${runId}/shap`, { validateStatus: () => true });
  return status === 200 ? (data as ShapResults) : null;
}

export async function getReport(runId: string): Promise<ReportResponse | null> {
  const { data, status } = await api.get(`/runs/${runId}/report`, { validateStatus: () => true });
  return status === 200 ? (data as ReportResponse) : null;
}

export async function getForecast(runId: string): Promise<ForecastResults | null> {
  const { data, status } = await api.get(`/runs/${runId}/forecast`, { validateStatus: () => true });
  return status === 200 ? (data as ForecastResults) : null;
}

