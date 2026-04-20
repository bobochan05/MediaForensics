"use client";

import { motion } from "framer-motion";
import { ExternalLink } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { ReportGenerator } from "@/components/dashboard/report-generator";
import { useDashboardStore } from "@/store/dashboard-store";
import { shortDateTime } from "@/lib/utils";

export function ResultsPanel() {
  const result = useDashboardStore((state) => state.result);
  const error = useDashboardStore((state) => state.error);

  if (error) {
    return (
      <Card className="border-rose-400/30 bg-rose-500/10">
        <CardHeader>
          <CardTitle>Analysis Error</CardTitle>
          <CardDescription>{error}</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Results Panel</CardTitle>
          <CardDescription>Run an analysis to populate layer outputs and intelligence insights.</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const isFake = result.layer1.result === "FAKE";
  const growthRate = result.layer3.growth?.rate_percent ?? result.layer3.growth_rate ?? 0;
  const riskScore = result.layer3.risk?.risk_score ?? result.layer3.risk_score ?? 0;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div>
            <CardTitle>Layer 1 · Deepfake Detection</CardTitle>
            <CardDescription>Authenticity verdict with confidence and explainability context.</CardDescription>
          </div>
          <ReportGenerator />
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-[200px,1fr]">
            <div className="rounded-[28px] border border-[var(--app-line)] bg-[var(--app-panel-soft)] p-4">
              <p className="text-xs uppercase tracking-wide text-[var(--app-text-muted)]">Verdict</p>
              <p className="mt-2 text-3xl font-bold text-[var(--app-text-strong)]">{result.layer1.result}</p>
              <Badge variant={isFake ? "danger" : "success"} className="mt-2">
                {result.layer1.confidence.toFixed(2)}% confidence
              </Badge>
            </div>
            <div className="rounded-[28px] border border-[var(--app-line)] bg-[var(--app-panel-soft)] p-4">
              <p className="mb-2 text-sm text-[var(--app-text-muted)]">Confidence meter</p>
              <Progress value={result.layer1.confidence} className="mb-3" />
              <p className="text-xs text-[var(--app-text-muted)]">Heatmap overlay: {result.layer1.heatmap ? "available" : "not available"}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Layer 2 · Source Matching</CardTitle>
          <CardDescription>CLIP + FAISS + reverse-search references with similarity confidence.</CardDescription>
        </CardHeader>
        <CardContent>
          {result.layer2.matches.length === 0 ? (
            <p className="text-sm text-[var(--app-text-muted)]">No external source matches returned for this run.</p>
          ) : (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {result.layer2.matches.map((match, index) => (
                <motion.article
                  key={match.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="overflow-hidden rounded-[28px] border border-[var(--app-line)] bg-[var(--app-panel-soft)]"
                >
                  <img src={match.preview_url} alt={match.title} className="h-28 w-full object-cover" />
                  <div className="space-y-2 p-3">
                    <p className="line-clamp-1 text-sm font-medium text-[var(--app-text-strong)]">{match.title}</p>
                    <p className="text-xs text-[var(--app-text-muted)]">Similarity {(match.similarity * 100).toFixed(1)}%</p>
                    <p className="text-xs text-[var(--app-text-muted)]">First seen {shortDateTime(match.first_seen)}</p>
                    <a
                      className="inline-flex items-center gap-1 text-xs text-[var(--app-text-strong)] hover:text-white"
                      href={match.source_url}
                      target="_blank"
                      rel="noreferrer"
                    >
                      View Source
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  </div>
                </motion.article>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Layer 3 · Tracking & Alerts</CardTitle>
          <CardDescription>Propagation velocity, growth trend, and spike monitoring.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-3">
            <Metric label="Growth Rate" value={`${growthRate.toFixed(2)}%`} />
            <Metric label="Risk Score" value={`${(riskScore * 100).toFixed(1)}%`} />
            <Metric label="Active Alerts" value={String(result.layer3.alerts.length)} />
          </div>

          {result.layer3.alerts.length > 0 && (
            <div className="mt-4 space-y-2">
              {result.layer3.alerts.map((alert) => (
                <div key={alert.id} className="rounded-[24px] border border-amber-400/30 bg-amber-500/10 p-3">
                  <p className="text-sm font-medium text-[#ffe7b5]">{alert.title}</p>
                  <p className="text-xs text-[#d8c7a3]">{alert.message}</p>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[24px] border border-[var(--app-line)] bg-[var(--app-panel-soft)] p-3">
      <p className="text-xs uppercase tracking-wide text-[var(--app-text-muted)]">{label}</p>
      <p className="mt-1 text-xl font-semibold text-[var(--app-text-strong)]">{value}</p>
    </div>
  );
}
