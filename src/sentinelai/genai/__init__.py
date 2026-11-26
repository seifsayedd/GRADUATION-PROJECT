"""Phase 5: GenAI layer (skeleton).

Incident reports and analyst Q&A grounded only in the project's own evidence
(Phase 4 explanation objects, findings docs, the ATT&CK taxonomy). The LLM is
behind ``providers.LLMProvider``; settings and API keys come only from
environment variables (see ``.env.example``). Every generated report passes a
groundedness check before it is stored: numbers and IP addresses must appear
in the cited sources.
"""
