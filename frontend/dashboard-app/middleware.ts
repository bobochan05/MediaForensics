import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const protectedPaths = ["/dashboard"];
const USE_DUMMY = process.env.NEXT_PUBLIC_USE_DUMMY_DATA === "true";

export function middleware(request: NextRequest) {
  if (USE_DUMMY) {
    return NextResponse.next();
  }

  const { pathname } = request.nextUrl;
  const isProtected = protectedPaths.some((path) => pathname.startsWith(path));

  if (!isProtected) {
    return NextResponse.next();
  }

  const accessToken = request.cookies.get("tracelyt_access_token")?.value;
  if (!accessToken) {
    const loginUrl = new URL("/login", request.url);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard/:path*"],
};
