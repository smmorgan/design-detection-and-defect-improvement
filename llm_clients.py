



#!/usr/bin/env python3
"""
Provider-agnostic LLM client for design/non-design ticket classification.

Supports Anthropic, OpenAI, and local (Ollama) models via tool-use /
function-calling so responses are returned as structured
{label, confidence, rationale} rather than parsed free text.

Requires the relevant SDK + API key only at call time (not import time), so this
module can be imported and the rest of the pipeline built/tested before keys exist:

    export ANTHROPIC_API_KEY=...   # for provider="anthropic"
    export OPENAI_API_KEY=...      # for provider="openai"
    pip install anthropic openai

For provider="local", no API key is needed -- it talks to an Ollama server's
OpenAI-compatible endpoint (default http://localhost:11434/v1, override with
OLLAMA_BASE_URL). Start the server and pull a model first:

    ollama serve &
    ollama pull llama3.1:8b
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

DESIGN_DEFINITION = """You are classifying JIRA issues for a research study on architectural \
("design") work in software projects.

Design issues involve one of:
- documentation or creation of architecture
- code or process refactoring for project-level improvement
- design for new features

Non-design issues include:
- bug fixes
- code cleanup
- implementation tasks with clear, already-decided scope

Classify using ONLY the issue's own text below. Do not assume additional context."""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["design", "non-design"],
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in the label, from 0.0 (guess) to 1.0 (certain).",
        },
        "rationale": {
            "type": "string",
            "description": "One sentence explaining the classification.",
        },
    },
    "required": ["label", "confidence", "rationale"],
}


@dataclass
class ClassificationResult:
    label: str
    confidence: float
    rationale: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    provider: str


def _clean(value):
    """Normalize None/NaN (e.g. missing pandas fields) to '' -- NaN is truthy in
    Python so `value or ''` lets it through, and slicing a float then blows up."""
    if value is None:
        return ''
    if isinstance(value, float) and value != value:  # NaN
        return ''
    return str(value)


def build_user_message(issue_type, priority, summary, description, few_shot_examples=None):
    parts = []
    if few_shot_examples:
        parts.append("Labeled examples from OTHER projects (for calibration only):\n")
        for ex in few_shot_examples:
            parts.append(
                f"- [{ex.get('issue_type', 'unknown')}] {_clean(ex.get('summary', ''))}\n"
                f"  {_clean(ex.get('description', ''))[:400]}\n"
                f"  -> {ex['label']}\n"
            )
        parts.append("\nNow classify this issue:\n")
    parts.append(f"Issue type: {issue_type or 'unknown'}")
    parts.append(f"Priority: {priority or 'unknown'}")
    parts.append(f"Summary: {_clean(summary)}")
    parts.append(f"Description: {_clean(description)[:4000]}")
    return "\n".join(parts)


def _classify_anthropic(model, user_message, api_key):
    import anthropic

    # Identity-linked API keys (as opposed to workspace-scoped keys) require the
    # target workspace id to be sent explicitly as a header.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    client = anthropic.Anthropic(api_key=api_key, default_headers=default_headers)
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=300,
        system=DESIGN_DEFINITION,
        messages=[{"role": "user", "content": user_message}],
        tools=[{
            "name": "classify_issue",
            "description": "Record the design/non-design classification.",
            "input_schema": CLASSIFY_SCHEMA,
        }],
        tool_choice={"type": "tool", "name": "classify_issue"},
    )
    latency = time.monotonic() - t0

    tool_use = next(b for b in resp.content if b.type == "tool_use")
    parsed = tool_use.input
    return ClassificationResult(
        label=parsed["label"],
        confidence=float(parsed["confidence"]),
        rationale=parsed["rationale"],
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        latency_s=latency,
        model=model,
        provider="anthropic",
    )


def _classify_openai_compatible(model, user_message, api_key, base_url, provider_name):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DESIGN_DEFINITION},
            {"role": "user", "content": user_message},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "classify_issue",
                "description": "Record the design/non-design classification.",
                "parameters": CLASSIFY_SCHEMA,
            },
        }],
        tool_choice={"type": "function", "function": {"name": "classify_issue"}},
    )
    latency = time.monotonic() - t0

    import json
    call = resp.choices[0].message.tool_calls[0]
    parsed = json.loads(call.function.arguments)
    return ClassificationResult(
        label=parsed["label"],
        confidence=float(parsed["confidence"]),
        rationale=parsed["rationale"],
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
        latency_s=latency,
        model=model,
        provider=provider_name,
    )


def _classify_openai(model, user_message, api_key):
    return _classify_openai_compatible(model, user_message, api_key, base_url=None, provider_name="openai")


def _classify_local(model, user_message, api_key):
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    return _classify_openai_compatible(model, user_message, api_key, base_url=base_url, provider_name="local")


PROVIDERS = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "classify_fn": _classify_anthropic,
        "default_model": "claude-sonnet-5",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "classify_fn": _classify_openai,
        "default_model": "gpt-4o-2024-08-06",
    },
    "local": {
        "env_var": None,
        "classify_fn": _classify_local,
        "default_model": "llama3.1:8b",
    },
}

# Approximate USD per 1M tokens (input, output). Update to match actual pricing
# at run time -- this is only used for the cost-overhead estimate in the paper,
# not for anything that affects results. Local models cost $0 in API fees.
PRICE_PER_M_TOKENS = {
    "claude-sonnet-5": (3.00, 15.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "llama3.1:8b": (0.0, 0.0),
}


def classify_issue(
    provider: str,
    issue_type: str,
    priority: str,
    summary: str,
    description: str,
    model: Optional[str] = None,
    few_shot_examples: Optional[list] = None,
    api_key: Optional[str] = None,
) -> ClassificationResult:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(PROVIDERS)}")

    cfg = PROVIDERS[provider]
    if cfg["env_var"] is None:
        key = api_key or "not-needed"
    else:
        key = api_key or os.environ.get(cfg["env_var"])
        if not key:
            raise RuntimeError(
                f"No API key found for provider '{provider}'. "
                f"Set the {cfg['env_var']} environment variable."
            )

    model = model or cfg["default_model"]
    user_message = build_user_message(issue_type, priority, summary, description, few_shot_examples)
    return cfg["classify_fn"](model, user_message, key)


def classify_issue_with_retry(max_attempts: int = 6, base_delay: float = 2.0, **kwargs) -> ClassificationResult:
    """Wraps classify_issue with exponential backoff so transient rate limits
    (429s from a shared account-wide TPM budget, momentary 503s, etc.) are
    retried rather than silently dropping the ticket -- a dropped ticket biases
    per-project metrics rather than just costing time.
    """
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return classify_issue(**kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
    raise last_exc


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICE_PER_M_TOKENS:
        return float("nan")
    in_price, out_price = PRICE_PER_M_TOKENS[model]
    return (input_tokens / 1e6) * in_price + (output_tokens / 1e6) * out_price
