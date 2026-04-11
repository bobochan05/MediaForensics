"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useDashboardStore } from "@/store/dashboard-store";

const steps = [
  "Upload media (image/video) into the analysis panel.",
  "AI detects authenticity risk and confidence patterns.",
  "Layer 2 surfaces globally similar sources.",
  "Layer 3 tracks spread over time and spike alerts.",
  "Generate a report for sharing or downstream systems.",
];

export function OnboardingPanel() {
  const isFirstRun = useDashboardStore((state) => state.isFirstRun);
  const completeOnboarding = useDashboardStore((state) => state.completeOnboarding);

  if (!isFirstRun) {
    return null;
  }

  return (
    <Card className="border-[#2f4f7b] bg-[#10203a]/70">
      <CardHeader>
        <CardTitle>How To Use Tracelyt Intelligence</CardTitle>
        <CardDescription>Quick onboarding for first-time analysts.</CardDescription>
      </CardHeader>
      <CardContent>
        <ol className="space-y-2 text-sm text-[#c1d4f3]">
          {steps.map((step, idx) => (
            <li key={step} className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2">
              <span className="mr-2 font-semibold text-[#7bb5ff]">{idx + 1}.</span>
              {step}
            </li>
          ))}
        </ol>

        <p className="mt-3 text-xs text-[#8ea7cd]">
          Tip: keep all layers enabled for complete risk context when investigating high-impact content.
        </p>

        <div className="mt-4">
          <Button variant="outline" onClick={completeOnboarding}>Got it</Button>
        </div>
      </CardContent>
    </Card>
  );
}
