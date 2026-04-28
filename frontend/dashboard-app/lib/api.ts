import type { AnalyzeResponse, AnalysisToggles, ChatResponse, DashboardUser } from "@/lib/types";
import { dummyAnalyzeResponse } from "@/lib/dummy-data";

const USE_DUMMY = process.env.NEXT_PUBLIC_USE_DUMMY_DATA === "true";
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").trim().replace(/\/$/, "");
const ACCESS_TOKEN_STORAGE_KEY = "tracelyt_access_token";
const ANALYSIS_POLL_INTERVAL_MS = 1500;
const ANALYSIS_POLL_TIMEOUT_MS = 180000;

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

function storedAccessToken(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return window.sessionStorage.getItem(ACCESS_TOKEN_STORAGE_KEY) || "";
}

function authHeaders(headers?: HeadersInit): HeadersInit {
  const token = storedAccessToken();
  if (!token) {
    return headers || {};
  }
  return {
    ...(headers || {}),
    Authorization: `Bearer ${token}`,
  };
}

export function captureAccessTokenFromUrl(): boolean {
  if (typeof window === "undefined" || !window.location.hash) {
    return false;
  }

  const params = new URLSearchParams(window.location.hash.slice(1));
  const token = params.get("access_token") || "";
  if (!token) {
    return false;
  }

  window.sessionStorage.setItem(ACCESS_TOKEN_STORAGE_KEY, token);
  window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
  return true;
}

async function requestWithRefresh(path: string, init: RequestInit): Promise<Response> {
  const initial = await fetch(apiUrl(path), {
    ...init,
    headers: authHeaders(init.headers),
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
    headers: authHeaders(init.headers),
    credentials: "include",
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}

function normalizeAnalysis(payload: unknown): AnalyzeResponse {
  const raw = asRecord(payload);
  const layer1 = asRecord(raw.layer1);
  const layer2 = asRecord(raw.layer2);
  const layer3 = asRecord(raw.layer3);
  const risk = asRecord(layer3.risk);
  const growth = asRecord(layer3.growth);
  const meta = asRecord(raw.meta);
  const guestUsage = asRecord(raw.guest_usage);

  return {
    ...(raw as unknown as AnalyzeResponse),
    analysis_id: String(raw.analysis_id || raw.upload_id || ""),
    auth_state: String(raw.auth_state || "anonymous") as AnalyzeResponse["auth_state"],
    guest_usage: {
      guest_limit: Number(guestUsage.guest_limit || 5),
      guest_used: Number(guestUsage.guest_used || 0),
      guest_remaining: Number(guestUsage.guest_remaining || 5),
    },
    layer1: {
      result: String(layer1.result || "PENDING") as AnalyzeResponse["layer1"]["result"],
      confidence: Number(layer1.confidence || 0),
      heatmap: typeof layer1.heatmap === "string" ? layer1.heatmap : null,
    },
    layer2: {
      ...(layer2 as AnalyzeResponse["layer2"]),
      matches: asArray(layer2.matches),
      count: Number(layer2.count || asArray(layer2.matches).length || 0),
    },
    layer3: {
      ...(layer3 as AnalyzeResponse["layer3"]),
      timeline: asArray(layer3.timeline),
      growth: {
        rate_percent: Number(growth.rate_percent || 0),
        spike_detected: Boolean(growth.spike_detected),
        window: String(growth.window || "1h"),
      },
      growth_rate: Number(layer3.growth_rate || growth.rate_percent || 0),
      alerts: asArray(layer3.alerts),
      risk_score: Number(layer3.risk_score || risk.risk_score || 0),
      risk: {
        risk_score: Number(risk.risk_score || layer3.risk_score || 0),
        fake_probability: Number(risk.fake_probability || 0),
        spread_velocity: Number(risk.spread_velocity || 0),
        source_credibility: Number(risk.source_credibility || 0),
      },
    },
    meta: {
      filename: String(meta.filename || ""),
      created_at: String(meta.created_at || new Date().toISOString()),
      model_version: String(meta.model_version || "fusion-v1"),
    },
  };
}

async function pollAnalysisJob(jobId: string): Promise<AnalyzeResponse> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < ANALYSIS_POLL_TIMEOUT_MS) {
    await sleep(ANALYSIS_POLL_INTERVAL_MS);
    const statusRes = await requestWithRefresh(`/api/status/${encodeURIComponent(jobId)}`, {
      method: "GET",
      cache: "no-store",
    });
    const statusPayload = await parseJsonSafe<Record<string, unknown>>(statusRes);

    if (!statusRes.ok || statusPayload.status === "error") {
      throw new Error(String(statusPayload.message || statusPayload.error || "Analysis status check failed."));
    }

    if (statusPayload.status === "completed" && statusPayload.analysis) {
      return normalizeAnalysis(statusPayload.analysis);
    }
  }

  throw new Error("Analysis is still processing. Please retry in a moment.");
}

export async function fetchAuthSession(): Promise<DashboardUser> {
  const res = await fetch(apiUrl("/api/auth/session"), {
    method: "GET",
    headers: authHeaders(),
    credentials: "include",
    cache: "no-store",
  });

  if (!res.ok) {
    if (typeof window !== "undefined") {
      window.sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
    }
    return {
      authState: "anonymous",
      accountType: "Free",
      apiUsed: 0,
      apiLimit: 250,
    };
  }

  const payload = await parseJsonSafe<Record<string, unknown>>(res);
  const authState = String(payload.auth_state || "anonymous") as DashboardUser["authState"];
  if (authState === "anonymous" && typeof window !== "undefined") {
    window.sessionStorage.removeItem(ACCESS_TOKEN_STORAGE_KEY);
  }
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

  if (payload.analysis && payload.status === "completed") {
    return normalizeAnalysis(payload.analysis);
  }

  if (payload.status === "processing" && payload.job_id) {
    return pollAnalysisJob(String(payload.job_id));
  }

  if (payload.analysis) {
    return normalizeAnalysis(payload.analysis);
  }

  return normalizeAnalysis(payload);
}

export async function sendChatMessage(input: {
  message: string;
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
