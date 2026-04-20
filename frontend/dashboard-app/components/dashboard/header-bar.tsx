"use client";

import { Bell, Menu, PanelLeftClose, Activity } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useDashboardStore } from "@/store/dashboard-store";

export function HeaderBar({
  sidebarOpen,
  onToggleSidebar,
}: {
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
}) {
  const user = useDashboardStore((state) => state.user);
  const result = useDashboardStore((state) => state.result);

  const alertCount = result?.layer3.alerts.length ?? 0;
  const apiUsed = user?.apiUsed ?? 0;
  const apiLimit = user?.apiLimit ?? 250;
  const displayUser = user?.username || user?.email || "Guest";
  const accountType = user?.accountType ?? "Free";

  return (
    <header className="sticky top-4 z-20 rounded-[22px] border border-black/5 bg-white/95 px-5 py-4 text-[#111827] shadow-[0_3px_16px_rgba(15,23,42,.05)] backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleSidebar}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-[#e5e7eb] bg-transparent text-[#111827] transition hover:bg-[#f3f4f6]"
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>

          <div className="hidden sm:block">
            <p className="text-[11px] font-extrabold uppercase tracking-[0.18em] text-[#6b7280]">Tracelyt</p>
            <h1 className="text-lg font-semibold text-[#111827]">Investigation workspace</h1>
          </div>
        </div>

        <div className="pointer-events-none absolute left-1/2 top-1/2 hidden -translate-x-1/2 -translate-y-1/2 md:block">
          <div className="rounded-full border border-[#e5e7eb] px-4 py-2 text-sm font-extrabold uppercase tracking-[0.18em] text-[#374151]">
            Tracelyt
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={accountType === "Pro" ? "success" : "default"}>{accountType}</Badge>
          <Badge variant="default">User: {displayUser}</Badge>
          <Badge variant="default">Attempts: {apiUsed}/{apiLimit}</Badge>
          <Badge variant="success" className="gap-1">
            <Activity className="h-3.5 w-3.5" /> Active
          </Badge>
          <button
            type="button"
            className="relative inline-flex h-10 w-10 items-center justify-center rounded-full border border-[#e5e7eb] bg-transparent text-[#111827] transition hover:bg-[#f3f4f6]"
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
    </header>
  );
}
