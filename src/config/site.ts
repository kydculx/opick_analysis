export const siteConfig = {
  name: "Opick Analysis",
  description: "Next.js + Supabase(Read-only) + Tailwind SaaS 스타터",
  nav: [{ title: "대시보드", href: "/" }],
  links: {
    docs: "https://nextjs.org/docs",
    supabase: "https://supabase.com/docs",
  },
} as const;

export type SiteConfig = typeof siteConfig;
