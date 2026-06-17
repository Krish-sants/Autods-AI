"use client";

import { useEffect, useState } from "react";
import { Loader2, Lock } from "lucide-react";
import { UNAUTHORIZED_EVENT, getStoredPassword, setStoredPassword, verifyPassword } from "@/lib/api";

export default function PasswordGate({ children }: { children: React.ReactNode }) {
  // Start "locked" until we've checked localStorage on the client, to avoid an
  // SSR/client mismatch (server has no localStorage to read from).
  const [checked, setChecked] = useState(false);
  const [unlocked, setUnlocked] = useState(false);
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function init() {
      const stored = getStoredPassword();
      if (stored) {
        setUnlocked(true);
        setChecked(true);
        return;
      }
      // No stored password — probe the backend to see if auth is required.
      // If the backend has APP_ACCESS_PASSWORD unset, it returns 200 and we skip the gate.
      try {
        await verifyPassword("");
        setUnlocked(true);
      } catch {
        // backend requires a password — show the gate
      }
      setChecked(true);
    }
    init();

    function handleUnauthorized() {
      setUnlocked(false);
    }
    window.addEventListener(UNAUTHORIZED_EVENT, handleUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, handleUnauthorized);
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(false);
    const ok = await verifyPassword(password);
    if (ok) {
      setStoredPassword(password);
      setUnlocked(true);
    } else {
      setError(true);
    }
    setSubmitting(false);
  }

  if (!checked) return null;

  if (!unlocked) {
    return (
      <div className="flex flex-1 items-center justify-center px-6">
        <form onSubmit={handleSubmit} className="flex w-full max-w-sm flex-col gap-4 rounded-xl border border-zinc-200 p-6">
          <div className="flex items-center gap-2 text-zinc-700">
            <Lock size={18} />
            <h2 className="font-semibold">This AutoDS-AI instance is password-protected</h2>
          </div>
          <input
            type="password"
            autoFocus
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Access password"
            className="rounded-lg border border-zinc-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          {error && <p className="text-sm text-red-600">Incorrect password.</p>}
          <button
            type="submit"
            disabled={submitting || !password}
            className="flex items-center justify-center gap-2 rounded-lg bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {submitting && <Loader2 size={14} className="animate-spin" />}
            Unlock
          </button>
        </form>
      </div>
    );
  }

  return <>{children}</>;
}
