import type { AnalyzeResponse, AnalysisToggles, ChatResponse, DashboardUser } from "@/lib/types";
import { dummyAnalyzeResponse } from "@/lib/dummy-data";

const USE_DUMMY = process.env.NEXT_PUBLIC_USE_DUMMY_DATA === "true";
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").trim().replace(/\/$/, "");

async function parseJsonSafe<T>(res: Response): Promise<T> {
  const payload = (await res.json().catch(() => ({}))) as T;
  return payload;
}

function apiUrl(path: string): string {
  if (!API_BASE_URL) {
    return path;
  }
  return `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestWithRefresh(path: string, init: RequestInit): Promise<Response> {
  const initial = await fetch(apiUrl(path), {
    ...init,
    credentials: "include",
  });

  if (initial.status !== 401) {
    return initial;
  }

  const refreshRes = await fetch(apiUrl("/api/auth/refresh"), {
    method: "POST",
    credentials: "include",
    cache: "no-store",
  });
  if (!refreshRes.ok) {
    return initial;
  }

  return fetch(apiUrl(path), {
    ...init,
    credentials: "include",
  });
}

export async function fetchAuthSession(): Promise<DashboardUser> {
  const res = await fetch(apiUrl("/api/auth/session"), {
    method: "GET",
    credentials: "include",
    cache: "no-store",
  });

  if (!res.ok) {
    return {
      authState: "anonymous",
      accountType: "Free",
      apiUsed: 0,
      apiLimit: 250,
    };
  }

  const payload = await parseJsonSafe<Record<string, unknown>>(res);
  const authState = String(payload.auth_state || "anonymous") as DashboardUser["authState"];
  const apiLimit = Number(payload.guest_limit || 250);
  const apiUsed = Number(payload.guest_used || 0);

  return {
    authState,
    email: String(payload.user_email || "") || undefined,
    username: String(payload.user_username || "") || undefined,
    accountType: authState === "user" ? "Pro" : "Free",
    apiUsed,
    apiLimit,
  };
}

export async function analyzeMedia(file: File, toggles: AnalysisToggles): Promise<AnalyzeResponse> {
  if (USE_DUMMY) {
    await new Promise((resolve) => setTimeout(resolve, 950));
    return dummyAnalyzeResponse;
  }

  const formData = new FormData();
  formData.append("file", file);
  formData.append("enable_layer1", String(toggles.layer1));
  formData.append("enable_layer2", String(toggles.layer2));
  formData.append("enable_layer3", String(toggles.layer3));

  const res = await requestWithRefresh("/api/analyze", {
    method: "POST",
    body: formData,
  });
  const payload = await parseJsonSafe<Record<string, unknown>>(res);

  if (!res.ok) {
    const error = String(payload.error || "Analysis failed.");
    throw new Error(error);
  }

  return payload as unknown as AnalyzeResponse;
}

export async function sendChatMessage(input: {
  message: string;
  analysis_id?: string;
  layer1?: Record<string, unknown>;
  layer2?: Record<string, unknown>;
  layer3?: Record<string, unknown>;
}): Promise<ChatResponse> {
  const res = await requestWithRefresh("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(input),
  });
  const payload = await parseJsonSafe<Record<string, unknown>>(res);

  if (!res.ok) {
    const error = String(payload.error || "Chat request failed.");
    throw new Error(error);
  }

  return payload as unknown as ChatResponse;
}

export async function fetchAnalysisSummary(uploadId: string): Promise<{ summary: string; provider: string }> {
  if (USE_DUMMY) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    return {
      summary: "## Summary\nThis content appears to be highly suspicious...\n\n## Key Findings\n- AI artifacts detected in Layer 1\n- Multiple source matches in Layer 2",
      provider: "mock",
    };
  }

  const res = await requestWithRefresh("/api/analysis-summary", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ upload_id: uploadId }),
  });

  if (!res.ok) {
    const payload = await parseJsonSafe<Record<string, unknown>>(res);
    throw new Error(String(payload.error || "Failed to fetch AI summary."));
  }

  return res.json();
}
