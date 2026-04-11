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
        "inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-[#4f8cff] disabled:cursor-not-allowed disabled:opacity-60",
        variant === "default" && "bg-gradient-to-r from-[#4f8cff] to-[#17b6ff] text-white hover:brightness-110",
        variant === "ghost" && "bg-transparent text-[#d8e6ff] hover:bg-white/10",
        variant === "outline" && "border border-white/20 bg-transparent text-[#d8e6ff] hover:bg-white/5",
        className
      )}
      {...props}
    />
  );
}
