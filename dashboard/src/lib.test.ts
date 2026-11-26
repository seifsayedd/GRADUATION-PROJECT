import { describe, expect, it } from "vitest";
import { buildMatrix, contributionBars, fmt, linear, linkCitations, pct, shortModel, symmetricExtent } from "./lib";
import type { Matrix } from "./types";

describe("scales", () => {
  it("maps linearly and handles a zero-width domain", () => {
    const s = linear(0, 10, 0, 100);
    expect(s(5)).toBe(50);
    expect(linear(3, 3, 0, 100)(3)).toBe(50);
  });
  it("symmetric extent uses the largest absolute value", () => {
    expect(symmetricExtent([-3, 1, 2])).toEqual([-3, 3]);
    expect(symmetricExtent([])[1]).toBeGreaterThan(0);
  });
  it("diverging bars start at the centre and keep their sign", () => {
    const bars = contributionBars([
      { feature: "A", value: "0", contribution: 2, direction: "towards attack" },
      { feature: "B", value: "1", contribution: -1, direction: "towards benign" },
    ], 200);
    expect(bars[0]).toMatchObject({ x: 100, width: 100, positive: true });
    expect(bars[1]).toMatchObject({ x: 50, width: 50, positive: false });
  });
});

describe("buildMatrix", () => {
  const m: Matrix = {
    source: "", note: "", tactics: ["Benign", "Impact", "Exfiltration"], models: ["a", "b"],
    cells: [
      { tactic: "Benign", model: "a", objects: 4, flagged: 1 },
      { tactic: "Impact", model: "b", objects: 2, flagged: 2 },
    ],
  };
  it("indexes cells and drops empty tactics", () => {
    const g = buildMatrix(m);
    expect(g.tactics).toEqual(["Benign", "Impact"]);
    expect(g.get("Impact", "b")).toMatchObject({ objects: 2, flagged: 2 });
    expect(g.get("Impact", "a")).toBeUndefined();
    expect(g.max).toBe(4);
  });
  it("hides models and the tactics that only they covered", () => {
    const g = buildMatrix(m, new Set(["b"]));
    expect(g.models).toEqual(["a"]);
    expect(g.tactics).toEqual(["Benign"]);
  });
});

describe("linkCitations", () => {
  it("splits text around known ids, longest first", () => {
    const p = linkCitations("see docs/a.md#1 and docs/a.md#12.", ["docs/a.md#1", "docs/a.md#12"]);
    expect(p).toEqual([{ text: "see " }, { cite: "docs/a.md#1" }, { text: " and " }, { cite: "docs/a.md#12" }, { text: "." }]);
  });
  it("leaves text alone without ids", () => {
    expect(linkCitations("plain", [])).toEqual([{ text: "plain" }]);
  });
});

describe("formatting", () => {
  it("formats numbers and missing values", () => {
    expect(fmt(0.12345)).toBe("0.123");
    expect(fmt(null)).toBe("—");
    expect(pct(0.0625)).toBe("6.3%");
    expect(shortModel("stage1_xgboost_temporal_lab")).toBe("XGB_lab");
    expect(shortModel("autoencoder_temporal_full")).toBe("AE_full");
  });
});
