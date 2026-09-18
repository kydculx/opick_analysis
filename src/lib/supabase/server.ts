import { createClient } from "@supabase/supabase-js";
import { env } from "@/lib/env";

export function createServerSupabaseClient() {
  return createClient(env.supabaseUrl, env.supabaseServiceKey, {
    auth: { persistSession: false },
  });
}
