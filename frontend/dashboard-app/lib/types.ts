export type AuthState = "anonymous" | "user" | "guest";

export interface DashboardUser {
  authState: AuthState;
  email?: string;
  username?: string;
  accountType: "Free" | "Pro";
  apiUsed: number;
  apiLimit: number;
}

export interface Layer1Result {
  result: "REAL" | "FAKE";
  confidence: number;
  heatmap: string | null;
}

export interface Layer2Match {
  id: string;
  preview_url: string;
  source_url: string;
  similarity: number;
  first_seen: string;
  platform: string;
  title: string;
}

export interface TimelinePoint {
  timestamp: string;
  mentions: number;
}

export interface AlertItem {
  id: string;
  severity: "low" | "warning" | "high";
  title: string;
  message: string;
  created_at: string;
}

export interface RiskMetrics {
  risk_score: number;
  fake_probability: number;
  spread_velocity: number;
  source_credibility: number;
}

export interface AnalyzeResponse {
  analysis_id: string;
  auth_state: AuthState;
  guest_usage: {
    guest_limit: number;
    guest_used: number;
    guest_remaining: number;
  };
  layer1: Layer1Result;
  layer2: {
    matches: Layer2Match[];
    count: number;
  };
  layer3: {
    timeline: TimelinePoint[];
    growth: {
      rate_percent: number;
      spike_detected: boolean;
      window: string;
    };
    growth_rate: number;
    alerts: AlertItem[];
    risk_score: number;
    risk: RiskMetrics;
  };
  meta: {
    filename: string;
    created_at: string;
    model_version: string;
  };
}

export interface ChatResponse {
  reply: string;
  context_used: boolean;
  analysis_summary: {
    layer1_result: string;
    layer1_confidence: number;
    match_count: number;
    timeline_points: number;
    growth_rate_percent: number;
    risk_score_percent: number;
    alert_count: number;
  };
}

export interface AnalysisToggles {
  layer1: boolean;
  layer2: boolean;
  layer3: boolean;
}
