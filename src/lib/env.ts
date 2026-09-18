// 서버 전용 secret 사용. NEXT_PUBLIC_에 secret 절대 금지.
// 공개: NEXT_PUBLIC_SUPABASE_URL
// 서버전용: SUPABASE_SECRET_KEY (.env.local, git 커밋 금지)

function required(key: string, value: string | undefined): string {
  if (!value) {
    // 빌드 시점에는 빈 문자열로 통과시키고, 런타임에 명확한 에러를 던진다.
    // 이렇게 해야 ENV 없이도 `npm run build`가 깨지지 않는다.
    if (process.env.NODE_ENV === "production" && typeof window === "undefined") {
      console.warn(`[env] Missing ${key}. Supabase 조회는 런타임에 실패합니다.`);
      return "";
    }
    throw new Error(`Missing env: ${key}. .env.example을 참고해 .env.local을 만드세요.`);
  }
  return value;
}

export const env = {
  get supabaseUrl() {
    return required(
      "NEXT_PUBLIC_SUPABASE_URL",
      process.env.NEXT_PUBLIC_SUPABASE_URL
    );
  },
  get supabaseServiceKey() {
    return required(
      "SUPABASE_SECRET_KEY",
      process.env.SUPABASE_SECRET_KEY
    );
  },
  get supabaseAnonKey() {
    return process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || "";
  },
};
