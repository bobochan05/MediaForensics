"use client";

import { useState } from "react";
import { HeaderBar } from "@/components/dashboard/header-bar";
import { LeftSidebar } from "@/components/dashboard/left-sidebar";
import { ResultsPanel } from "@/components/dashboard/results-panel";
import { RightSidebar } from "@/components/dashboard/right-sidebar";
import { UploadPanel } from "@/components/dashboard/upload-panel";
import { VisualizationPanel } from "@/components/dashboard/visualization-panel";
import { cn } from "@/lib/utils";
import { useDashboardStore } from "@/store/dashboard-store";

export function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const showRawData = useDashboardStore((state) => state.showRawData);
  const result = useDashboardStore((state) => state.result);

  return (
    <div className="min-h-screen bg-transparent text-[var(--app-text)]">
      <div className="mx-auto max-w-[1680px] px-4 pb-8 pt-4 lg:px-6">
        <HeaderBar sidebarOpen={sidebarOpen} onToggleSidebar={() => setSidebarOpen((value) => !value)} />

        <div
          className={cn(
            "mt-5 grid gap-5 transition-[grid-template-columns] duration-300",
            sidebarOpen ? "xl:grid-cols-[240px,minmax(0,1fr),320px]" : "xl:grid-cols-[minmax(0,1fr),320px]"
          )}
        >
          {sidebarOpen && <LeftSidebar active="dashboard" />}

          <main className="space-y-5">
            <UploadPanel />
            <ResultsPanel />
            <VisualizationPanel />

            {showRawData && result && (
              <section className="rounded-[32px] border border-[var(--app-line)] bg-[var(--app-panel)] p-4 backdrop-blur-sm">
                <h3 className="mb-2 text-sm font-semibold text-[var(--app-text-strong)]">Raw Analysis Payload</h3>
                <pre className="max-h-[360px] overflow-auto rounded-2xl bg-black/20 p-3 text-xs text-[var(--app-text-muted)]">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </section>
            )}
          </main>

          <RightSidebar />
        </div>
      </div>
    </div>
  );
}
