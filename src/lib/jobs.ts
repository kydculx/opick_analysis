import { spawn } from "node:child_process";
import { mkdirSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export const JOB_DIR = join(tmpdir(), "opick-ml-jobs");

export type JobStatus = {
  id: string;
  kind: "train" | "predict";
  status: "running" | "done" | "error";
  league: string;
  seasons: string[];
  ver: string;
  log?: string;
  startedAt: string;
  finishedAt?: string;
};

export function readJob(id: string): JobStatus | null {
  try {
    return JSON.parse(readFileSync(join(JOB_DIR, `${id}.json`), "utf-8"));
  } catch {
    return null;
  }
}

function writeJob(job: JobStatus) {
  mkdirSync(JOB_DIR, { recursive: true });
  writeFileSync(join(JOB_DIR, `${job.id}.json`), JSON.stringify(job));
}

export function startJob(opts: {
  kind: "train" | "predict";
  league: string;
  seasons: string[];
  ver: string;
  argv: string[];
}): JobStatus {
  const id = `${opts.league}-${opts.kind}-${Date.now()}`;
  const job: JobStatus = {
    id,
    kind: opts.kind,
    status: "running",
    league: opts.league,
    seasons: opts.seasons,
    ver: opts.ver,
    startedAt: new Date().toISOString(),
  };
  mkdirSync(JOB_DIR, { recursive: true });
  const logPath = join(JOB_DIR, `${id}.log`);
  writeJob(job);

  const python = process.env.ML_PYTHON || "python3";
  const logFd = openSync(logPath, "w");
  const child = spawn(/*turbopackIgnore: true*/ python, ["ml/train.py", ...opts.argv], {
    cwd: process.cwd(),
    stdio: ["ignore", logFd, logFd],
  });
  child.on("error", (e) => {
    writeJob({ ...job, status: "error", log: e.message, finishedAt: new Date().toISOString() });
  });
  child.on("close", (code) => {
    let log: string | undefined;
    try {
      const tail = readFileSync(logPath, "utf-8").split("\n").filter(Boolean).slice(-5).join(" | ");
      if (code !== 0 && tail) log = tail;
    } catch {
      log = code === 0 ? undefined : `exit code ${code}`;
    }
    writeJob({ ...job, status: code === 0 ? "done" : "error", log, finishedAt: new Date().toISOString() });
  });
  return job;
}

export function isJobId(id: string) {
  return /^[A-Za-z0-9_-]+$/.test(id);
}
