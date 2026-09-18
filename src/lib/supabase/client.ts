"use client";

import { createClient } from "@supabase/supabase-js";
import { env } from "@/lib/env";

let browserClient: ReturnType<typeof createClient> | null = null;

export function createBrowserSupabaseClient() {
  if (browserClient) return browserClient;
  if (!env.supabaseAnonKey) {
    throw new Error("NEXT_PUBLIC_SUPABASE_ANON_KEY 없음. 브라우저 조회가 필요하면 anon key를 추가하세요. 현재는 서버 조회만 사용.");
  }
  browserClient = createClient(env.supabaseUrl, env.supabaseAnonKey);
  return browserClient;
}
