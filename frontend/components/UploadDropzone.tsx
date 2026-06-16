"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import { UploadCloud, Link as LinkIcon, Loader2 } from "lucide-react";
import { startRun, uploadDataset, uploadFromUrl } from "@/lib/api";

export default function UploadDropzone() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [url, setUrl] = useState("");

  const goToRun = useCallback(
    async (runId: string) => {
      await startRun(runId);
      router.push(`/runs/${runId}`);
    },
    [router]
  );

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (!file) return;
      setBusy(true);
      try {
        const { run_id } = await uploadDataset(file);
        toast.success(`Uploaded ${file.name}`);
        await goToRun(run_id);
      } catch (err) {
        toast.error(axiosErrorMessage(err));
      } finally {
        setBusy(false);
      }
    },
    [goToRun]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      "text/csv": [".csv"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
      "application/json": [".json"],
      "application/octet-stream": [".parquet"],
    },
  });

  async function handleUrlSubmit() {
    if (!url.trim()) return;
    setBusy(true);
    try {
      const { run_id } = await uploadFromUrl(url.trim());
      toast.success("Dataset downloaded");
      await goToRun(run_id);
    } catch (err) {
      toast.error(axiosErrorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="w-full max-w-xl flex flex-col gap-6">
      <div
        {...getRootProps()}
        className={`flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-12 text-center transition-colors cursor-pointer ${
          isDragActive ? "border-blue-500 bg-blue-50" : "border-zinc-300 hover:border-zinc-400"
        }`}
      >
        <input {...getInputProps()} />
        {busy ? (
          <Loader2 className="animate-spin" size={32} />
        ) : (
          <UploadCloud size={32} className="text-zinc-500" />
        )}
        <p className="font-medium">Drag & drop a dataset here, or click to browse</p>
        <p className="text-sm text-zinc-500">CSV, Excel (.xlsx), JSON, or Parquet</p>
      </div>

      <div className="flex items-center gap-2">
        <div className="h-px flex-1 bg-zinc-200" />
        <span className="text-sm text-zinc-400">or</span>
        <div className="h-px flex-1 bg-zinc-200" />
      </div>

      <div className="flex gap-2">
        <div className="relative flex-1">
          <LinkIcon size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400" />
          <input
            type="url"
            placeholder="Paste a dataset URL"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={busy}
            className="w-full rounded-lg border border-zinc-300 py-2 pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
        </div>
        <button
          onClick={handleUrlSubmit}
          disabled={busy || !url.trim()}
          className="rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Fetch
        </button>
      </div>
    </div>
  );
}

function axiosErrorMessage(err: unknown): string {
  if (err && typeof err === "object" && "response" in err) {
    const response = (err as { response?: { data?: { detail?: string } } }).response;
    if (response?.data?.detail) return response.data.detail;
  }
  return "Something went wrong. Please try again.";
}
