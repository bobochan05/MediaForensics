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
    <aside className="rounded-2xl border border-white/10 bg-[#0f1729]/80 p-3">
      <nav className="space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = item.id === active;
          return (
            <button
              key={item.id}
              className={cn(
                "group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left text-sm transition",
                isActive
                  ? "bg-gradient-to-r from-[#4f8cff]/25 to-[#17b6ff]/20 text-white"
                  : "text-[#9bb0d4] hover:bg-white/5 hover:text-white"
              )}
              type="button"
            >
              <Icon className={cn("h-4 w-4", isActive ? "text-[#72b3ff]" : "text-[#7e96bb] group-hover:text-[#aad1ff]")} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="mt-5 rounded-xl border border-white/10 bg-white/[0.03] p-3">
        <div className="flex items-center gap-2 text-sm text-[#b8cae8]">
          <BarChart3 className="h-4 w-4 text-[#5ba7ff]" />
          Insight Mode Enabled
        </div>
        <p className="mt-2 text-xs leading-relaxed text-[#8ea6cb]">
          Surfaces anomalies, spread spikes, and confidence drifts with decision-ready context.
        </p>
      </div>
    </aside>
  );
}
