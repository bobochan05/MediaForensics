import { cn } from "@/lib/utils";

const variants = {
  default: "bg-white/10 text-[#d6e4ff] border-white/15",
  success: "bg-emerald-500/20 text-emerald-200 border-emerald-400/40",
  warning: "bg-amber-500/20 text-amber-100 border-amber-400/40",
  danger: "bg-rose-500/20 text-rose-100 border-rose-400/40",
};

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: keyof typeof variants }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
        variants[variant],
        className
      )}
      {...props}
    />
  );
}
