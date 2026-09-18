import Link from "next/link";
import { siteConfig } from "@/config/site";
import { Container } from "./Container";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-black/60">
      <Container className="flex h-14 items-center justify-between">
        <Link href="/" className="font-semibold tracking-tight">
          {siteConfig.name}
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          {siteConfig.nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="rounded-full px-3 py-1.5 text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 dark:text-zinc-400 dark:hover:bg-zinc-900 dark:hover:text-zinc-100"
            >
              {item.title}
            </Link>
          ))}
          <Link
            href="/"
            className="ml-2 rounded-full bg-zinc-900 px-4 py-1.5 font-medium text-white hover:bg-zinc-700 dark:bg-white dark:text-black"
          >
            시작하기
          </Link>
        </nav>
      </Container>
    </header>
  );
}
