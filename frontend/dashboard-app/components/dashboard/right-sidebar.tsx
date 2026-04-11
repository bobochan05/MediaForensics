"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { ChatPanel } from "@/components/dashboard/chat-panel";
import { useDashboardStore } from "@/store/dashboard-store";

export function RightSidebar() {
  const user = useDashboardStore((state) => state.user);
  const modelVersion = useDashboardStore((state) => state.modelVersion);
  const setModelVersion = useDashboardStore((state) => state.setModelVersion);
  const showRawData = useDashboardStore((state) => state.showRawData);
  const setShowRawData = useDashboardStore((state) => state.setShowRawData);
  const resetRun = useDashboardStore((state) => state.resetRun);

  const apiLimit = user?.apiLimit ?? 250;
  const apiUsed = user?.apiUsed ?? 0;
  const usagePercentage = Math.min(100, (apiUsed / Math.max(1, apiLimit)) * 100);

  return (
    <aside className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Advanced Controls</CardTitle>
          <CardDescription>Operator controls for repeatable intelligence workflows.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button variant="outline" className="w-full gap-2" onClick={resetRun}>
            <RefreshCw className="h-4 w-4" />
            Re-run Analysis
          </Button>

          <div className="rounded-xl border border-white/10 bg-white/[0.02] p-3">
            <p className="mb-2 text-xs uppercase tracking-wide text-[#8ea7cd]">Model Version</p>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant={modelVersion === "fusion-v1" ? "default" : "outline"}
                className="w-full"
                onClick={() => setModelVersion("fusion-v1")}
              >
                v1
              </Button>
              <Button
                variant={modelVersion === "fusion-v2" ? "default" : "outline"}
                className="w-full"
                onClick={() => setModelVersion("fusion-v2")}
              >
                v2
              </Button>
            </div>
          </div>

          <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.02] px-3 py-2">
            <p className="text-sm text-[#dce8ff]">Show Raw Data</p>
            <Switch checked={showRawData} onCheckedChange={setShowRawData} />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API Usage</CardTitle>
          <CardDescription>Current consumption for this account window.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-2 flex justify-between text-sm text-[#dce8ff]">
            <span>{apiUsed} used</span>
            <span>{apiLimit} quota</span>
          </div>
          <Progress value={usagePercentage} />
        </CardContent>
      </Card>

      <ChatPanel />
    </aside>
  );
}
