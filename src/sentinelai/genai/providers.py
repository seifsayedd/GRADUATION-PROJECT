"""LLM providers behind one interface.

* ``AnthropicProvider``: the official ``anthropic`` SDK, model from settings
  (default ``claude-opus-5-5``), structured JSON output, and server-side refusal
  fallbacks enabled by default (``fallbacks="default"``: if Claude declines on
  safety grounds, the API retries on another Claude model in the same call).
* ``TemplateProvider``: no LLM at all. Builds the report from the sources with
  fixed rules, so the pipeline runs offline and the output is grounded by
  construction.
* ``FakeProvider``: scripted answers for tests. Never touches the network.

Adding another provider means writing one class with ``generate``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import Settings, require_api_key

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class RefusalError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str
    data: dict | None
    model: str
    stop_reason: str | None = None
    fallback_used: bool = False


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, user: str, schema: dict | None = None, *, sources: list[dict] | None = None,
                 task: str = "report") -> Generation: ...


# ------------------------------------------------------------------ Anthropic
class AnthropicProvider:
    name = "anthropic"

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=require_api_key())
        self.client = client

    def generate(self, system, user, schema=None, *, sources=None, task="report") -> Generation:
        output_config = {"effort": self.settings.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        resp = self.client.beta.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=self.settings.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise RefusalError(f"the model declined this request (category: {cat})")
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        iters = getattr(getattr(resp, "usage", None), "iterations", None) or []
        used = any(getattr(i, "type", None) == "fallback_message" for i in iters)
        return Generation(text, json.loads(text) if schema is not None else None, resp.model, resp.stop_reason, used)


# ------------------------------------------------------------------ Template (offline)
_SENT = re.compile(r"(?<=[.!?])\s+")


class TemplateProvider:
    """Rule-based stand-in for an LLM: copies facts from the sources, so it cannot invent anything."""
    name = "template"

    def generate(self, system, user, schema=None, *, sources=None, task="report") -> Generation:
        sources = sources or []
        alert = next((s for s in sources if s["id"].startswith("alert:")), None)
        others = [s for s in sources if s is not alert]
        if task == "report" and alert is not None:
            e = alert["object"]
            items = e.get("contributions", {}).get("items", [])[:4]
            evidence = [{"statement": f"{i['feature']} = {i['value']} pushed the score {i['direction']} "
                                      f"({i['contribution']:+.3f} {e['contributions']['space']}).", "citations": [alert["id"]]}
                        for i in items]
            for a in e.get("attention", [])[:2]:
                evidence.append({"statement": f"Attention weight {a['weight']:.3f} on {a.get('flow', 'an earlier flow')}.",
                                 "citations": [alert["id"]]})
            tac = e.get("tactic", {}).get("predicted", "none (not flagged)")
            data = {"title": f"{'Alert' if e['decision']['flagged'] else 'No alert'}: {e['flow']['protocol']} flow "
                             f"{e['flow']['src']} -> {e['flow']['dst']}",
                    "summary": e["summary"],
                    "evidence": evidence,
                    "attack_mapping": {"tactic": tac, "note": e.get("tactic", {}).get("taxonomy_note", ""),
                                       "citations": [alert["id"]]},
                    "caveats": list(e.get("caveats", [])),
                    "recommended_checks": ([f"Review the related project context in {s['origin']}." for s in others[:2]]
                                           or ["No related project documents were retrieved."])}
        else:
            first = [(_SENT.split(s["text"].strip())[0], s["id"]) for s in others[:3] if s["text"].strip()]
            data = {"answer": " ".join(t for t, _ in first) or "Not in the evidence.",
                    "citations": [i for _, i in first], "in_evidence": bool(first)}
        return Generation(json.dumps(data), data, "template", "end_turn")


# ------------------------------------------------------------------ Fake (tests)
@dataclass
class FakeProvider:
    """Returns scripted outputs. ``script`` is a list of dicts/strings or a function of the call."""
    script: list | Callable = field(default_factory=list)
    calls: list = field(default_factory=list)
    name: str = "fake"

    def generate(self, system, user, schema=None, *, sources=None, task="report") -> Generation:
        self.calls.append({"system": system, "user": user, "schema": schema, "sources": sources, "task": task})
        out = self.script(self.calls[-1]) if callable(self.script) else self.script.pop(0)
        text = out if isinstance(out, str) else json.dumps(out)
        return Generation(text, json.loads(text) if schema is not None else None, "fake", "end_turn")


def get_provider(settings: Settings) -> LLMProvider:
    if settings.provider == "anthropic":
        return AnthropicProvider(settings)
    if settings.provider == "template":
        return TemplateProvider()
    raise ValueError(f"unknown provider {settings.provider!r} (anthropic | template)")
