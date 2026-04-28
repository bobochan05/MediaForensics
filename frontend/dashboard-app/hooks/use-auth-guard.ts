"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { captureAccessTokenFromUrl, fetchAuthSession } from "@/lib/api";
import { useDashboardStore } from "@/store/dashboard-store";

const USE_DUMMY = process.env.NEXT_PUBLIC_USE_DUMMY_DATA === "true";

export function useAuthGuard() {
  const router = useRouter();
  const setUser = useDashboardStore((state) => state.setUser);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function run() {
      if (USE_DUMMY) {
        setUser({
          authState: "guest",
          username: "Demo User",
          accountType: "Free",
          apiUsed: 0,
          apiLimit: 250,
        });
        setIsLoading(false);
        return;
      }

      captureAccessTokenFromUrl();
      const session = await fetchAuthSession().catch(() => ({
        authState: "anonymous" as const,
        accountType: "Free" as const,
        apiUsed: 0,
        apiLimit: 250,
      }));
      if (!mounted) {
        return;
      }

      if (session.authState === "anonymous") {
        router.replace("/login");
        return;
      }

      setUser(session);
      setIsLoading(false);
    }

    run();
    return () => {
      mounted = false;
    };
  }, [router, setUser]);

  return { isLoading };
}
