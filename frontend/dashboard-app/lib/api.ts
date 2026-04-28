import type { AnalyzeResponse, AnalysisToggles, ChatResponse, DashboardUser } from "@/lib/types";
import { dummyAnalyzeResponse } from "@/lib/dummy-data";

const USE_DUMMY = process.env.NEXT_PUBLIC_USE_DUMMY_DATA === "true";
const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL || "").trim().replace(/\/$/, "");
const ACCESS_TOKEN_STORAGE_KEY = "tracelyt_access_token";

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

  return payload as unknown as AnalyzeResponse;
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
