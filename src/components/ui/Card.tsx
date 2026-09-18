import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cn("rounded-2xl border border-zinc-200 bg-white p-6 dark:border-zinc-800 dark:bg-zinc-950", className)}>
      {children}
    </div>
  );
}

export function CardTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-base font-semibold text-zinc-900 dark:text-zinc-50">{children}</h3>;
}

export function CardDesc({ children }: { children: ReactNode }) {
  return <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">{children}</p>;
}
