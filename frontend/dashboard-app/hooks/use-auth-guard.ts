"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchAuthSession } from "@/lib/api";
import { useDashboardStore } from "@/store/dashboard-store";

export function useAuthGuard() {
  const router = useRouter();
  const setUser = useDashboardStore((state) => state.setUser);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let mounted = true;

    async function run() {
      const session = await fetchAuthSession();
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
