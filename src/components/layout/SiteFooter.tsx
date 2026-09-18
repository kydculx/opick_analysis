import { siteConfig } from "@/config/site";
import { Container } from "./Container";

export function SiteFooter() {
  return (
    <footer className="border-t border-zinc-200 py-8 dark:border-zinc-800">
      <Container className="flex flex-col gap-2 text-sm text-zinc-500 sm:flex-row sm:items-center sm:justify-between">
        <p>
          © {new Date().getFullYear()} {siteConfig.name}
        </p>
        <p>{siteConfig.description}</p>
      </Container>
    </footer>
  );
}
