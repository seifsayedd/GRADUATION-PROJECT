import type {
  AlertRow, AnswerOutput, Evaluation, Explanation, Generated, Matrix, Rejection, ReportOutput, SourceText,
  Summary, Timeline,
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, public detail: unknown) {
    super(typeof detail === "string" ? detail : `HTTP ${status}`);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, (body as { detail?: unknown }).detail ?? body);
  return body as T;
}

export const api = {
  health: () => call<{ status: string; provider: string; embedder: string; store: string; auth: string }>("/api/health"),
  summary: () => call<Summary>("/api/summary"),
  evaluation: () => call<Evaluation>("/api/evaluation"),
  alerts: (q: Record<string, string> = {}) =>
    call<{ count: number; rows: AlertRow[] }>("/api/alerts?" + new URLSearchParams(q).toString()),
  alert: (id: string) => call<Explanation>(`/api/alerts/${encodeURIComponent(id)}`),
  report: (id: string) => call<Generated<ReportOutput>>(`/api/alerts/${encodeURIComponent(id)}/report`, { method: "POST" }),
  ask: (question: string) => call<Generated<AnswerOutput>>("/api/ask", { method: "POST", body: JSON.stringify({ question }) }),
  source: (id: string) => call<SourceText>(`/api/sources/${id.split("/").map(encodeURIComponent).join("/")}`),
  timeline: () => call<Timeline>("/api/timeline"),
  matrix: () => call<Matrix>("/api/attack-matrix"),
};

export function asRejection(e: unknown): Rejection | null {
  if (e instanceof ApiError && e.status === 422 && typeof e.detail === "object" && e.detail !== null
      && (e.detail as Rejection).rejected) return e.detail as Rejection;
  return null;
}

export function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    if (typeof e.detail === "string") return `${e.status}: ${e.detail}`;
    return `${e.status}: ${JSON.stringify(e.detail)}`;
  }
  return e instanceof Error ? e.message : String(e);
}
