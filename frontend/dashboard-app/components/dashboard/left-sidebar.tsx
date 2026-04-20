"use client";

import { BarChart3, FileText, History, LayoutDashboard, Radar, Settings, UploadCloud } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "upload", label: "Upload & Analyze", icon: UploadCloud },
  { id: "history", label: "History", icon: History },
  { id: "reports", label: "Reports", icon: FileText },
  { id: "tracking", label: "Tracking", icon: Radar },
  { id: "settings", label: "Settings", icon: Settings },
];

export function LeftSidebar({ active = "dashboard" }: { active?: string }) {
  return (
    <aside className="rounded-[32px] border border-[var(--app-line)] bg-[var(--app-panel)] p-4 backdrop-blur-sm">
      <div className="mb-4 border-b border-[var(--app-line)] pb-4">
        <p className="text-[11px] font-extrabold uppercase tracking-[0.18em] text-[var(--app-text-muted)]">Workspace</p>
        <p className="mt-2 text-sm leading-relaxed text-[var(--app-text-muted)]">
          Detection, explanation, and propagation tracking in one place.
        </p>
      </div>

      <nav className="space-y-1.5">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              className={cn(
                "group flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left text-sm transition",
                isActive
                  ? "bg-white text-[#111827]"
                  : "text-[var(--app-text-muted)] hover:bg-[var(--app-accent-soft)] hover:text-[var(--app-text-strong)]"
              )}
              type="button"
            >
              <Icon className={cn("h-4 w-4", isActive ? "text-[#111827]" : "text-[var(--app-text-muted)] group-hover:text-[var(--app-text-strong)]")} />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-5 rounded-[28px] border border-[var(--app-line)] bg-[var(--app-panel-soft)] p-4">
        <div className="flex items-center gap-2 text-sm text-[var(--app-text-strong)]">
          <BarChart3 className="h-4 w-4" />
          Insight mode enabled
        </div>
        <p className="mt-2 text-xs leading-relaxed text-[var(--app-text-muted)]">
          Surfaces anomalies, spread spikes, and confidence drifts with decision-ready context.
        </p>
      </div>
    </aside>
  );
}
