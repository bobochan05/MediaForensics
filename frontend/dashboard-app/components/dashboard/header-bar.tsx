"use client";

import { Bell, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useDashboardStore } from "@/store/dashboard-store";

export function HeaderBar() {
  const user = useDashboardStore((state) => state.user);
  const result = useDashboardStore((state) => state.result);

  const alertCount = result?.layer3.alerts.length ?? 0;
  const apiUsed = user?.apiUsed ?? 0;
  const apiLimit = user?.apiLimit ?? 250;
  const displayUser = user?.username || user?.email || "Guest Analyst";
  const accountType = user?.accountType ?? "Free";

  return (
    <Card className="sticky top-4 z-20 border-white/10 bg-[#0d1528]/95 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-[#7e97c1]">Intelligence Console</p>
          <h1 className="text-lg font-semibold text-white">Welcome, {displayUser}</h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={accountType === "Pro" ? "success" : "default"}>{accountType}</Badge>
          <Badge variant="default">API {apiUsed}/{apiLimit}</Badge>
          <Badge variant="success" className="gap-1">
            <Activity className="h-3.5 w-3.5" /> System Active
          </Badge>
          <button
            type="button"
            className="relative inline-flex h-9 w-9 items-center justify-center rounded-xl border border-white/15 bg-white/5 text-[#d8e6ff] transition hover:bg-white/10"
            aria-label="Notifications"
          >
            <Bell className="h-4 w-4" />
            {alertCount > 0 && (
              <span className="absolute -right-1 -top-1 inline-flex min-h-5 min-w-5 items-center justify-center rounded-full bg-rose-500 px-1 text-[10px] font-semibold text-white">
                {alertCount}
              </span>
            )}
          </button>
        </div>
      </div>
    </Card>
  );
}
