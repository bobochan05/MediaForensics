"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost" | "outline";
}

export function Button({ className, variant = "default", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-full px-4 py-2 text-sm font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-white/30 disabled:cursor-not-allowed disabled:opacity-60",
        variant === "default" && "bg-[var(--app-accent)] text-[#111827] hover:bg-white",
        variant === "ghost" && "bg-transparent text-[var(--app-text-strong)] hover:bg-white/10",
        variant === "outline" && "border border-[var(--app-line-strong)] bg-transparent text-[var(--app-text-strong)] hover:bg-white/5",
        className
      )}
      {...props}
    />
  );
}
