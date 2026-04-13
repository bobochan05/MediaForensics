"use client";

import { useEffect, useState } from "react";
import { Sparkles, Loader2, AlertCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchAnalysisSummary } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";

interface InsightSummaryProps {
  uploadId: string;
}

export function InsightSummary({ uploadId }: InsightSummaryProps) {
  const [summary, setSummary] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!uploadId) return;

    async function loadSummary() {
      setIsLoading(true);
      setError(null);
      try {
        const data = await fetchAnalysisSummary(uploadId);
        setSummary(data.summary);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load AI summary.");
      } finally {
        setIsLoading(false);
      }
    }

    loadSummary();
  }, [uploadId]);

  if (!uploadId && !isLoading) return null;

  return (
    <Card className="mb-6 overflow-hidden border-none bg-gradient-to-r from-blue-600/10 via-indigo-600/5 to-transparent backdrop-blur-sm">
      <CardContent className="p-0">
        <div className="flex items-start gap-4 p-5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-[#4facfe] to-[#00f2fe] text-white shadow-lg shadow-blue-500/20">
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <Sparkles className="h-5 w-5" />
            )}
          </div>
          <div className="flex-1 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold uppercase tracking-wider text-blue-400">Forensic AI Insight</h3>
              {isLoading && <span className="text-[10px] font-medium text-blue-300/40 animate-pulse">GENERATING SUMMARY...</span>}
            </div>
            
            {error ? (
              <div className="flex items-center gap-2 text-sm text-red-400">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            ) : isLoading ? (
              <div className="space-y-2">
                <div className="h-4 w-[90%] animate-pulse rounded bg-white/5" />
                <div className="h-4 w-[75%] animate-pulse rounded bg-white/5" />
              </div>
            ) : (
              <div className="prose prose-invert prose-sm max-w-none prose-headings:text-white prose-p:text-blue-50/80 prose-strong:text-blue-300">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {summary || ""}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
