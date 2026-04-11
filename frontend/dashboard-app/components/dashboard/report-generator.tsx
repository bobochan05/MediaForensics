"use client";

import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useDashboardStore } from "@/store/dashboard-store";

export function ReportGenerator() {
  const result = useDashboardStore((state) => state.result);

  const generateReport = () => {
    if (!result) {
      return;
    }

    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `tracelyt-report-${result.analysis_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Button variant="outline" onClick={generateReport} disabled={!result} className="gap-2">
      <Download className="h-4 w-4" />
      Generate Report
    </Button>
  );
}
