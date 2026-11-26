// Pure helpers used by the views. Tested in lib.test.ts.
import type { Contribution, Matrix } from "./types";

/** Linear map from [d0, d1] to [r0, r1]. A zero-width domain maps to the middle of the range. */
export function linear(d0: number, d1: number, r0: number, r1: number): (x: number) => number {
  if (d1 === d0) return () => (r0 + r1) / 2;
  return (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
}

/** Symmetric extent for diverging bars: [-m, m] with m the largest absolute value (at least `min`). */
export function symmetricExtent(values: number[], min = 1e-9): [number, number] {
  const m = Math.max(min, ...values.map((v) => Math.abs(v)));
  return [-m, m];
}

export interface Bar { feature: string; value: string; contribution: number; x: number; width: number; positive: boolean }

/** Horizontal diverging bars for contributions around a centre line, in a box `width` pixels wide. */
export function contributionBars(items: Contribution[], width: number): Bar[] {
  const [lo, hi] = symmetricExtent(items.map((i) => i.contribution));
  const s = linear(lo, hi, 0, width);
  const mid = s(0);
  return items.map((i) => {
    const end = s(i.contribution);
    return { feature: i.feature, value: i.value, contribution: i.contribution, x: Math.min(mid, end),
             width: Math.abs(end - mid), positive: i.contribution >= 0 };
  });
}

export interface MatrixGrid {
  tactics: string[];
  models: string[];
  get: (tactic: string, model: string) => { objects: number; flagged: number } | undefined;
  max: number;
  rowTotals: Record<string, number>;
}

/** Index the API's sparse cell list into a grid, dropping hidden models and tactics with no objects left. */
export function buildMatrix(m: Matrix, hiddenModels: Set<string> = new Set()): MatrixGrid {
  const models = m.models.filter((x) => !hiddenModels.has(x));
  const key = (t: string, mo: string) => `${t}\u0000${mo}`;
  const idx = new Map(m.cells.filter((c) => !hiddenModels.has(c.model)).map((c) => [key(c.tactic, c.model), c]));
  const rowTotals: Record<string, number> = {};
  for (const c of idx.values()) rowTotals[c.tactic] = (rowTotals[c.tactic] ?? 0) + c.objects;
  const tactics = m.tactics.filter((t) => (rowTotals[t] ?? 0) > 0);
  const max = Math.max(0, ...[...idx.values()].map((c) => c.objects));
  return { tactics, models, get: (t, mo) => idx.get(key(t, mo)), max, rowTotals };
}

export type Piece = { text: string } | { cite: string };

/** Split text on citation-like ids that appear in `ids`, so the view can turn them into links. */
export function linkCitations(text: string, ids: string[]): Piece[] {
  const known = [...new Set(ids)].filter(Boolean).sort((a, b) => b.length - a.length);
  if (!known.length) return [{ text }];
  const esc = known.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${esc.join("|")})`, "g");
  const out: Piece[] = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index! > last) out.push({ text: text.slice(last, m.index) });
    out.push({ cite: m[0] });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}

export function fmt(x: number | null | undefined, digits = 3): string {
  return x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits);
}

export function pct(x: number | null | undefined, digits = 1): string {
  return x === null || x === undefined ? "—" : `${(x * 100).toFixed(digits)}%`;
}

export function shortModel(name: string): string {
  if (name === "stage2") return "GRU";
  if (name === "stage3") return "AE";
  return name.replace(/^stage\d_/, "").replace("random_forest", "RF").replace("xgboost", "XGB")
    .replace("autoencoder", "AE").replace("gru_w32", "GRU").replace("_temporal", "").replace("_stratified", " strat");
}

export function timeOf(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  return new Date(ms).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

/** Colour ramp from transparent to the accent, for heat-map cells (t in [0, 1]). */
export function heat(t: number): string {
  const a = Math.max(0, Math.min(1, t));
  return `rgba(56, 189, 248, ${(0.08 + 0.82 * a).toFixed(3)})`;
}
