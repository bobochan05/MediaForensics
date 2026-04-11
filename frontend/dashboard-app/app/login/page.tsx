import Link from "next/link";

export default function LoginPage() {
  return (
    <main className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-md rounded-2xl border border-white/15 bg-[#0e1628]/85 p-6 text-center">
        <h1 className="text-2xl font-semibold text-white">Login Required</h1>
        <p className="mt-2 text-sm text-[#9db0d1]">
          Use the Flask auth entry page to sign in, then return to the dashboard.
        </p>
        <Link
          href="http://127.0.0.1:5000/"
          className="mt-4 inline-flex rounded-xl bg-gradient-to-r from-[#4f8cff] to-[#17b6ff] px-4 py-2 text-sm font-medium text-white"
        >
          Go To Login
        </Link>
      </div>
    </main>
  );
}
