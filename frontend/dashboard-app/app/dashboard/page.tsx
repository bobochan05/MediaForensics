"use client";

import { useAuthGuard } from "@/hooks/use-auth-guard";
import { DashboardLayout } from "@/components/dashboard/dashboard-layout";
import { Skeleton } from "@/components/ui/skeleton";

export default function DashboardPage() {
  const { isLoading } = useAuthGuard();

  if (isLoading) {
    return (
      <div className="mx-auto max-w-[1200px] space-y-4 px-4 py-6">
        <Skeleton className="h-20 w-full" />
        <Skeleton className="h-[420px] w-full" />
      </div>
    );
  }

  return <DashboardLayout />;
}
