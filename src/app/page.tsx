import { DashboardExplorer } from "@/components/dashboard/DashboardExplorer";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col lg:h-screen lg:overflow-hidden">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-200 px-4 dark:border-zinc-800">
        <span className="font-semibold">Opick Analysis</span>
      </header>
      <main className="flex min-h-0 min-w-0 flex-1 flex-col gap-6 overflow-visible p-4 lg:overflow-hidden lg:p-6">
        <DashboardExplorer />
      </main>
    </div>
  );
}
