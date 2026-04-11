"use client";

import { motion } from "framer-motion";
import { UploadCloud, Video, Image as ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { useDashboardStore } from "@/store/dashboard-store";
import type { AnalysisToggles } from "@/lib/types";
import { analyzeMedia } from "@/lib/api";

const processStages = [
  "Analyzing authenticity...",
  "Searching global sources...",
  "Tracking propagation...",
];

export function UploadPanel() {
  const selectedFile = useDashboardStore((state) => state.selectedFile);
  const setSelectedFile = useDashboardStore((state) => state.setSelectedFile);
  const toggles = useDashboardStore((state) => state.toggles);
  const setToggle = useDashboardStore((state) => state.setToggle);
  const setProcessing = useDashboardStore((state) => state.setProcessing);
  const setResult = useDashboardStore((state) => state.setResult);
  const setError = useDashboardStore((state) => state.setError);
  const resetRun = useDashboardStore((state) => state.resetRun);
  const isProcessing = useDashboardStore((state) => state.isProcessing);
  const processMessage = useDashboardStore((state) => state.processMessage);

  const updateToggle = (key: keyof AnalysisToggles, value: boolean) => {
    setToggle(key, value);
  };

  const runAnalysis = async () => {
    if (!selectedFile || isProcessing) {
      return;
    }

    try {
      resetRun();
      setError(null);
      setProcessing(true, processStages[0]);

      for (let i = 0; i < processStages.length - 1; i += 1) {
        await new Promise((resolve) => setTimeout(resolve, 550));
        setProcessing(true, processStages[i + 1]);
      }

      const response = await analyzeMedia(selectedFile, toggles);
      setResult(response);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Analysis failed.";
      setError(message);
    } finally {
      setProcessing(false, "Ready for analysis");
    }
  };

  return (
    <Card className="border-white/10">
      <CardHeader>
        <CardTitle>Upload & Analyze</CardTitle>
        <CardDescription>
          Upload an image or video and run layered intelligence analysis with selective pipeline controls.
        </CardDescription>
      </CardHeader>

      <CardContent>
        <label
          htmlFor="media-input"
          className="group relative flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-white/20 bg-white/[0.03] p-6 text-center transition hover:border-[#5ca7ff]/70 hover:bg-white/[0.05]"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file) {
              if (!file.type.startsWith("image/") && !file.type.startsWith("video/")) {
                setError("Unsupported format. Upload an image or video file.");
                return;
              }
              setSelectedFile(file);
            }
          }}
        >
          <UploadCloud className="mb-3 h-8 w-8 text-[#7eb7ff]" />
          <p className="text-sm font-medium text-[#e5edff]">
            {selectedFile ? selectedFile.name : "Drag & drop image/video or click to browse"}
          </p>
          <p className="mt-1 text-xs text-[#8ca5c9]">Supported: JPG, PNG, MP4, MOV, WEBM</p>
          <input
            id="media-input"
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0] ?? null;
              if (file && !file.type.startsWith("image/") && !file.type.startsWith("video/")) {
                setError("Unsupported format. Upload an image or video file.");
                return;
              }
              setSelectedFile(file);
            }}
          />
        </label>

        <div className="mt-4 grid gap-3 rounded-xl border border-white/10 bg-white/[0.02] p-4 md:grid-cols-3">
          <ToggleRow label="Enable Layer 1" enabled={toggles.layer1} onChange={(v) => updateToggle("layer1", v)} icon={<ImageIcon className="h-4 w-4" />} />
          <ToggleRow label="Enable Layer 2" enabled={toggles.layer2} onChange={(v) => updateToggle("layer2", v)} icon={<UploadCloud className="h-4 w-4" />} />
          <ToggleRow label="Enable Layer 3" enabled={toggles.layer3} onChange={(v) => updateToggle("layer3", v)} icon={<Video className="h-4 w-4" />} />
        </div>

        <div className="mt-4 flex items-center justify-between gap-4">
          <motion.p
            key={processMessage}
            initial={{ opacity: 0.3, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-[#9cb0cf]"
          >
            {isProcessing ? processMessage : "Pipeline standing by"}
          </motion.p>
          <Button onClick={runAnalysis} disabled={!selectedFile || isProcessing}>
            {isProcessing ? "Running..." : "Run Intelligence Analysis"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function ToggleRow({
  label,
  enabled,
  onChange,
  icon,
}: {
  label: string;
  enabled: boolean;
  onChange: (checked: boolean) => void;
  icon: JSX.Element;
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-white/10 px-3 py-2">
      <div className="flex items-center gap-2 text-sm text-[#dbe7ff]">
        <span className="text-[#6ea7ff]">{icon}</span>
        {label}
      </div>
      <Switch checked={enabled} onCheckedChange={onChange} />
    </div>
  );
}
