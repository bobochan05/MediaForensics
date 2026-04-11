import type { AnalyzeResponse } from "@/lib/types";

export const dummyAnalyzeResponse: AnalyzeResponse = {
  analysis_id: "demo-analysis-001",
  auth_state: "user",
  guest_usage: {
    guest_limit: 5,
    guest_used: 1,
    guest_remaining: 4,
  },
  layer1: {
    result: "FAKE",
    confidence: 93.4,
    heatmap: null,
  },
  layer2: {
    count: 3,
    matches: [
      {
        id: "m1",
        preview_url: "https://images.unsplash.com/photo-1522075469751-3a6694fb2f61?w=800&q=80",
        source_url: "https://example-news.net/deepfake-sample",
        similarity: 0.91,
        first_seen: "2026-04-09T06:00:00Z",
        platform: "example-news.net",
        title: "Repost cluster candidate",
      },
      {
        id: "m2",
        preview_url: "https://images.unsplash.com/photo-1519389950473-47ba0277781c?w=800&q=80",
        source_url: "https://media-observer.org/visual-archive/201",
        similarity: 0.86,
        first_seen: "2026-04-09T07:30:00Z",
        platform: "media-observer.org",
        title: "Archive overlap",
      },
      {
        id: "m3",
        preview_url: "https://images.unsplash.com/photo-1507146153580-69a1fe6d8aa1?w=800&q=80",
        source_url: "https://social-stream.example/post/8821",
        similarity: 0.79,
        first_seen: "2026-04-09T08:10:00Z",
        platform: "social-stream.example",
        title: "Viral repost chain",
      },
    ],
  },
  layer3: {
    growth: {
      rate_percent: 168.5,
      spike_detected: true,
      window: "1h",
    },
    growth_rate: 168.5,
    risk_score: 0.86,
    alerts: [
      {
        id: "a1",
        severity: "high",
        title: "Propagation spike detected",
        message: "Mentions accelerated across multiple sources within 2 hours.",
        created_at: "2026-04-09T09:10:00Z",
      },
    ],
    risk: {
      risk_score: 0.86,
      fake_probability: 0.93,
      spread_velocity: 0.74,
      source_credibility: 0.28,
    },
    timeline: [
      { timestamp: "2026-04-09T00:00:00Z", mentions: 8 },
      { timestamp: "2026-04-09T01:00:00Z", mentions: 10 },
      { timestamp: "2026-04-09T02:00:00Z", mentions: 14 },
      { timestamp: "2026-04-09T03:00:00Z", mentions: 18 },
      { timestamp: "2026-04-09T04:00:00Z", mentions: 24 },
      { timestamp: "2026-04-09T05:00:00Z", mentions: 31 },
      { timestamp: "2026-04-09T06:00:00Z", mentions: 42 },
      { timestamp: "2026-04-09T07:00:00Z", mentions: 58 },
      { timestamp: "2026-04-09T08:00:00Z", mentions: 71 },
      { timestamp: "2026-04-09T09:00:00Z", mentions: 86 },
    ],
  },
  meta: {
    filename: "demo-file.jpg",
    created_at: "2026-04-09T09:15:00Z",
    model_version: "fusion-v1",
  },
};
