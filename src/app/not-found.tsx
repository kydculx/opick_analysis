import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 py-24 text-center">
      <p className="text-sm text-zinc-500">404</p>
      <h1 className="text-2xl font-bold">페이지를 찾을 수 없습니다</h1>
      <Link href="/" className="rounded-full bg-zinc-900 px-5 py-2.5 text-sm text-white dark:bg-white dark:text-black">
        홈으로
      </Link>
    </div>
  );
}
