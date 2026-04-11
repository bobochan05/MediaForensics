"use client";

import { HeaderBar } from "@/components/dashboard/header-bar";
import { LeftSidebar } from "@/components/dashboard/left-sidebar";
import { OnboardingPanel } from "@/components/dashboard/onboarding-panel";
import { ResultsPanel } from "@/components/dashboard/results-panel";
import { RightSidebar } from "@/components/dashboard/right-sidebar";
import { UploadPanel } from "@/components/dashboard/upload-panel";
import { VisualizationPanel } from "@/components/dashboard/visualization-panel";
import { useDashboardStore } from "@/store/dashboard-store";

export function DashboardLayout() {
  const showRawData = useDashboardStore((state) => state.showRawData);
  const result = useDashboardStore((state) => state.result);

  return (
    <div className="min-h-screen bg-[#070b14] text-[#e5edff]">
      <div className="mx-auto max-w-[1600px] px-4 pb-8 pt-4">
        <HeaderBar />

        <div className="mt-4 grid gap-4 xl:grid-cols-[240px,minmax(0,1fr),320px]">
          <LeftSidebar active="dashboard" />

          <main className="space-y-4">
            <OnboardingPanel />
            <UploadPanel />
            <ResultsPanel />
            <VisualizationPanel />

            {showRawData && result && (
              <section className="rounded-2xl border border-white/10 bg-[#0f1729]/80 p-4">
                <h3 className="mb-2 text-sm font-semibold text-[#dce8ff]">Raw Analysis Payload</h3>
                <pre className="max-h-[360px] overflow-auto rounded-lg bg-black/30 p-3 text-xs text-[#99b4dc]">
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
