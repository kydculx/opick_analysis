export type XgbSlimTree = {
  l: number[];
  r: number[];
  f: number[];
  v: number[];
  w: number[];
};

export type XgbSlim = {
  trees: XgbSlimTree[];
  numClass: number;
  base: number[];
};

function parseBaseScore(raw: unknown, numClass: number): number[] {
  if (typeof raw === "string") {
    const t = raw.trim();
    const inner = t.startsWith("[") && t.endsWith("]") ? t.slice(1, -1) : t;
    const parts = inner
      .split(",")
      .map((s) => Number.parseFloat(s.trim()))
      .filter((v) => Number.isFinite(v));
    if (parts.length === numClass) return parts;
    if (parts.length === 1) return new Array(numClass).fill(parts[0]);
  }
  if (typeof raw === "number" && Number.isFinite(raw)) {
    return new Array(numClass).fill(raw);
  }
  return new Array(numClass).fill(0);
}

export function slimBooster(full: unknown): XgbSlim | null {
  try {
    const o = full as {
      learner?: {
        gradient_booster?: { model?: { trees?: unknown[] } };
        learner_model_param?: { num_class?: unknown; base_score?: unknown };
      };
    };
    const trees = o?.learner?.gradient_booster?.model?.trees;
    if (!Array.isArray(trees) || trees.length === 0) return null;
    const ncRaw = o?.learner?.learner_model_param?.num_class;
    const numClass = typeof ncRaw === "string" ? parseInt(ncRaw, 10) : 3;
    if (!Number.isFinite(numClass) || numClass <= 0) return null;
    const base = parseBaseScore(o?.learner?.learner_model_param?.base_score, numClass);
    const slim: XgbSlimTree[] = [];
    for (const t of trees) {
      const tr = t as Record<string, unknown>;
      if (
        !Array.isArray(tr.l) && !Array.isArray(tr.left_children)
      ) {
        return null;
      }
      slim.push({
        l: (tr.left_children ?? tr.l) as number[],
        r: (tr.right_children ?? tr.r) as number[],
        f: (tr.split_indices ?? tr.f) as number[],
        v: (tr.split_conditions ?? tr.v) as number[],
        w: (tr.base_weights ?? tr.w) as number[],
      });
    }
    return { trees: slim, numClass, base };
  } catch {
    return null;
  }
}

export function xgbMargins(slim: XgbSlim, x: number[]): number[] {
  const margins = [...slim.base];
  while (margins.length < slim.numClass) margins.push(0);
  for (let t = 0; t < slim.trees.length; t++) {
    const tree = slim.trees[t];
    const c = t % slim.numClass;
    let node = 0;
    for (let guard = 0; guard < 64; guard++) {
      const left = tree.l[node] ?? -1;
      if (left < 0) {
        margins[c] += tree.w[node] ?? 0;
        break;
      }
      const f = tree.f[node] ?? 0;
      const xv = Math.fround(x[f] ?? 0);
      const cv = Math.fround(tree.v[node] ?? 0);
      node = xv < cv ? left : (tree.r[node] ?? 0);
      if (node < 0) break;
    }
  }
  return margins;
}

export function xgbProba(slim: XgbSlim, x: number[]): number[] {
  const m = xgbMargins(slim, x);
  const mx = Math.max(...m);
  const ex = m.map((v) => Math.exp(Math.max(-30, Math.min(30, v - mx))));
  const s = ex.reduce((a, b) => a + b, 0) || 1;
  return ex.map((v) => v / s);
}
