// Shapes returned by the Phase 6 API (src/sentinelai/api).

export interface Card {
  id: string;
  label: string;
  value: number | null;
  range: [number | null, number | null] | null;
  note: string;
  source: string;
  path: string;
}

export interface Summary {
  cards: Card[];
  lab_flags: { source: string; rows: { model: string; flagged: number; false_alarms: number; attack_caught: boolean }[] };
  holdout: { source: string; note: string; rows: { model: string; class: string; detected: number }[] };
}

export interface MetricRow {
  f1: number; f1_lo: number | null; f1_hi: number | null;
  precision: number; recall: number; fpr: number; n: number;
}

export interface Evaluation {
  operating_point: string;
  models: { source: string; rows: (MetricRow & { model: string; threshold: number })[] };
  loso: { source: string; rows: (MetricRow & { feature_set: string; test_source: string; roc_auc: number })[] };
}

export interface AlertRow {
  alert_id: string; model: string; stage: number; dataset: string | null;
  src: string | null; dst: string | null; protocol: string | null; dst_port: number | null;
  start_ms: number | null; score: number; threshold: number; score_kind: string; flagged: boolean;
  true_tactic: string | null; predicted_tactic: string | null;
}

export interface Contribution { feature: string; value: string; contribution: number; direction: string }

export interface AttentionItem { offset: number; weight: number; occlusion_delta: number; flow: string }

export interface Explanation {
  schema_version: string;
  alert_id: string;
  flow: Record<string, string | number | null>;
  model: { stage: number; name: string; feature_set: string; git_commit: string };
  ground_truth?: { label: number; attack: string; tactic: string };
  decision: { score: number; score_kind: string; threshold: number; operating_point: string; flagged: boolean };
  tactic?: { predicted: string; confidence: number };
  contributions?: { space: string; base_value: number; output: number; items: Contribution[]; rest: number; additivity_error: number };
  attention?: AttentionItem[];
  caveats: string[];
  summary: string;
}

export interface ReportOutput {
  title: string; summary: string;
  evidence: { statement: string; citations: string[] }[];
  attack_mapping: { tactic: string; note: string; citations: string[] };
  caveats: string[];
  recommended_checks: string[];
}

export interface Generated<T> {
  output: T; model: string; fallback_used: boolean; sources: string[]; provider: string; report_id?: string;
}

export interface AnswerOutput { answer: string; citations: string[]; in_evidence: boolean }

export interface Rejection {
  rejected: true; reason: string;
  unsupported_numbers?: string[]; unsupported_ips?: string[]; unknown_citations?: string[];
}

export interface TimelineRow {
  row: number; start_ms: number | null; src: string | null; dst: string | null; protocol: string | null;
  dst_port: number | null; truth: string | null; true_tactic: string | null;
  verdicts: Record<string, { alert_id: string; flagged: boolean; score: number }>;
  session?: string; gru_p_attack?: number;
}

export interface Timeline { source: string; note: string; models: string[]; rows: TimelineRow[] }

export interface Matrix {
  source: string; note: string; tactics: string[]; models: string[];
  cells: { tactic: string; model: string; objects: number; flagged: number }[];
}

export interface SourceText { id: string; origin: string; section?: string; text: string }
