"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDashboardStore } from "@/store/dashboard-store";
import { formatPercent, shortDateTime } from "@/lib/utils";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function VisualizationPanel() {
  const result = useDashboardStore((state) => state.result);

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Visualization Intelligence</CardTitle>
          <CardDescription>Timeline and risk storytelling appears after analysis.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const timeline = result.layer3.timeline.map((point) => ({
    ...point,
    timeLabel: shortDateTime(point.timestamp),
    risk: Math.round((result.layer3.risk?.risk_score ?? result.layer3.risk_score ?? 0) * 100),
  }));
  const risk = result.layer3.risk ?? {
    risk_score: result.layer3.risk_score ?? 0,
    fake_probability: 0,
    spread_velocity: 0,
    source_credibility: 0,
  };

  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Spread Timeline</CardTitle>
          <CardDescription>X: time · Y: mentions</CardDescription>
        </CardHeader>
        <CardContent className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timeline}>
              <defs>
                <linearGradient id="mentionsGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4f8cff" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#4f8cff" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148,163,184,.14)" strokeDasharray="4 6" />
              <XAxis dataKey="timeLabel" tick={{ fill: "#8ea7cd", fontSize: 11 }} minTickGap={24} />
              <YAxis tick={{ fill: "#8ea7cd", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#111b2e", border: "1px solid rgba(148,163,184,.2)", borderRadius: 10 }} />
              <Area type="monotone" dataKey="mentions" stroke="#67a8ff" fill="url(#mentionsGradient)" strokeWidth={2.2} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Growth Curve & Risk</CardTitle>
          <CardDescription>Spike-aware trendline with risk signal overlay.</CardDescription>
        </CardHeader>
        <CardContent className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timeline}>
              <CartesianGrid stroke="rgba(148,163,184,.14)" strokeDasharray="4 6" />
              <XAxis dataKey="timeLabel" tick={{ fill: "#8ea7cd", fontSize: 11 }} minTickGap={24} />
              <YAxis yAxisId="left" tick={{ fill: "#8ea7cd", fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: "#8ea7cd", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#111b2e", border: "1px solid rgba(148,163,184,.2)", borderRadius: 10 }} />
              <Line yAxisId="left" type="monotone" dataKey="mentions" stroke="#7db4ff" strokeWidth={2.2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="risk" stroke="#f59e0b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Risk Score Meter</CardTitle>
          <CardDescription>Composite risk from authenticity, velocity, and source credibility.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 sm:grid-cols-3">
            <RiskMetric label="Fake Probability" value={formatPercent(risk.fake_probability)} />
            <RiskMetric label="Spread Velocity" value={formatPercent(risk.spread_velocity)} />
            <RiskMetric label="Source Credibility" value={formatPercent(risk.source_credibility)} />
          </div>
          <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
            <p className="text-xs uppercase tracking-wide text-[#8ea7cd]">Composite Risk</p>
            <p className="mt-1 text-3xl font-bold text-white">{formatPercent(risk.risk_score)}</p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Network Graph</CardTitle>
          <CardDescription>Propagation topology placeholder (platform nodes + diffusion edges).</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="rounded-xl border border-dashed border-white/20 bg-white/[0.02] p-5 text-sm text-[#8ea7cd]">
            Node-edge propagation visualization placeholder ready for graph engine integration (D3 / Sigma).
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function RiskMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
      <p className="text-xs uppercase tracking-wide text-[#8ea7cd]">{label}</p>
      <p className="mt-1 text-base font-semibold text-[#e7f0ff]">{value}</p>
    </div>
  );
}
