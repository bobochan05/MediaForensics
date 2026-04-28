import Link from "next/link";
import { ArrowRight, Fingerprint, Radar, ShieldCheck } from "lucide-react";

const signals = [
  { label: "Deepfake Detection", icon: ShieldCheck },
  { label: "Source Tracing", icon: Fingerprint },
  { label: "Spread Intelligence", icon: Radar },
];

export default function HomePage() {
  return (
    <main className="min-h-screen overflow-hidden bg-[#f5f5f3] text-[#151515]">
      <section className="relative flex min-h-screen flex-col px-5 py-5 sm:px-8 lg:px-10">
        <div className="flex items-center justify-between rounded-full border border-black/10 bg-white/75 px-5 py-4 shadow-sm backdrop-blur">
          <span className="text-sm font-black uppercase tracking-[0.28em] text-[#242424]">Tracelyt</span>
          <Link
            href="/dashboard"
            className="inline-flex items-center gap-2 rounded-full bg-[#151515] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[#2d2d2d]"
          >
            Get Started
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </Link>
        </div>

        <div className="grid flex-1 content-center gap-10 py-12 lg:grid-cols-[minmax(0,1fr)_420px] lg:items-center">
          <div className="max-w-6xl">
            <p className="mb-5 max-w-2xl text-base font-semibold uppercase tracking-[0.26em] text-[#5c5f66]">
              AI media forensics workspace
            </p>
            <h1 className="text-[4.4rem] font-black uppercase leading-[0.78] tracking-normal text-[#151515] sm:text-[7rem] md:text-[9rem] lg:text-[11rem]">
              Tracelyt
            </h1>
            <p className="mt-8 max-w-3xl text-lg leading-8 text-[#4b4f58] sm:text-xl">
              Detect synthetic media, trace public source signals, and review investigation-ready risk context from one focused dashboard.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link
                href="/dashboard"
                className="inline-flex min-h-12 items-center gap-2 rounded-full bg-[#151515] px-6 py-3 text-base font-bold text-white transition hover:bg-[#303030]"
              >
                Get Started
                <ArrowRight className="h-5 w-5" aria-hidden="true" />
              </Link>
              <span className="rounded-full border border-black/10 bg-white px-5 py-3 text-sm font-semibold text-[#50545d]">
                Demo mode active
              </span>
            </div>
          </div>

          <div className="relative min-h-[430px] overflow-hidden rounded-[32px] border border-black/10 bg-[#191a1d] p-6 text-white shadow-2xl">
            <div className="absolute inset-x-0 top-0 h-24 bg-white/10" />
            <div className="relative flex items-center justify-between">
              <span className="text-xs font-black uppercase tracking-[0.24em] text-white/55">Live scan</span>
              <span className="rounded-full border border-emerald-300/30 bg-emerald-300/15 px-3 py-1 text-xs font-bold text-emerald-100">
                Active
              </span>
            </div>

            <div className="relative mt-10 rounded-2xl border border-white/12 bg-white/[0.06] p-5">
              <div className="h-44 rounded-xl border border-white/10 bg-[linear-gradient(135deg,#2d3037,#121316)]">
                <div className="grid h-full grid-cols-6 gap-px p-3 opacity-70">
                  {Array.from({ length: 36 }).map((_, index) => (
                    <span
                      key={index}
                      className={`rounded-sm ${
                        index % 7 === 0
                          ? "bg-sky-300/70"
                          : index % 5 === 0
                            ? "bg-rose-300/60"
                            : "bg-white/10"
                      }`}
                    />
                  ))}
                </div>
              </div>
              <div className="mt-5 flex items-end justify-between">
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.2em] text-white/45">Verdict</p>
                  <p className="mt-1 text-4xl font-black">FAKE</p>
                </div>
                <p className="rounded-full bg-rose-400/20 px-3 py-1 text-sm font-bold text-rose-100">93.4%</p>
              </div>
            </div>

            <div className="relative mt-5 grid gap-3">
              {signals.map(({ label, icon: Icon }) => (
                <div key={label} className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3">
                  <span className="font-semibold text-white/82">{label}</span>
                  <Icon className="h-5 w-5 text-sky-200" aria-hidden="true" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
