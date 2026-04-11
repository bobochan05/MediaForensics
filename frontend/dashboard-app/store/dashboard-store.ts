"use client";

import { create } from "zustand";
import type { AnalyzeResponse, AnalysisToggles, DashboardUser } from "@/lib/types";

interface DashboardState {
  user: DashboardUser | null;
  selectedFile: File | null;
  toggles: AnalysisToggles;
  isProcessing: boolean;
  processMessage: string;
  result: AnalyzeResponse | null;
  modelVersion: "fusion-v1" | "fusion-v2";
  showRawData: boolean;
  isFirstRun: boolean;
  error: string | null;
  setUser: (user: DashboardUser) => void;
  setSelectedFile: (file: File | null) => void;
  setToggle: (key: keyof AnalysisToggles, value: boolean) => void;
  setProcessing: (processing: boolean, message?: string) => void;
  setResult: (result: AnalyzeResponse | null) => void;
  setModelVersion: (version: "fusion-v1" | "fusion-v2") => void;
  setShowRawData: (value: boolean) => void;
  completeOnboarding: () => void;
  setError: (message: string | null) => void;
  resetRun: () => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  user: null,
  selectedFile: null,
  toggles: {
    layer1: true,
    layer2: true,
    layer3: true,
  },
  isProcessing: false,
  processMessage: "Ready for analysis",
  result: null,
  modelVersion: "fusion-v1",
  showRawData: false,
  isFirstRun: true,
  error: null,
  setUser: (user) => set({ user }),
  setSelectedFile: (file) => set({ selectedFile: file }),
  setToggle: (key, value) =>
    set((state) => ({
      toggles: {
        ...state.toggles,
        [key]: value,
      },
    })),
  setProcessing: (processing, message) =>
    set({
      isProcessing: processing,
      processMessage: message ?? (processing ? "Processing..." : "Ready for analysis"),
    }),
  setResult: (result) => set({ result, isFirstRun: false }),
  setModelVersion: (version) => set({ modelVersion: version }),
  setShowRawData: (value) => set({ showRawData: value }),
  completeOnboarding: () => set({ isFirstRun: false }),
  setError: (message) => set({ error: message }),
  resetRun: () => set({ result: null, error: null, processMessage: "Ready for analysis", isProcessing: false }),
}));
