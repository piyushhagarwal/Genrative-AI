"""
classifier.py
-------------
Everything the classifier needs lives here:
  - Enums       → the allowed values for category and priority
  - Pydantic    → validates the LLM's JSON output
  - Cost table  → estimates USD per API call
  - System prompt → tells the LLM what to return
  - classify_with_openai()    → calls OpenAI API
  - classify_with_anthropic() → calls Anthropic API
  - classify_ticket()         → the one function you call from anywhere
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# load_dotenv() reads your .env file and puts the keys into os.environ
# so os.environ.get("ANTHROPIC_API_KEY") works without manually exporting
load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
# basicConfig sets up a simple console logger
# %(asctime)s   → timestamp
# %(levelname)s → INFO / WARNING / ERROR
# %(message)s   → the actual message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Enums ──────────────────────────────────────────────────────────────────────
# (str, Enum) means the value serialises as a plain string ("billing")
# instead of Category.BILLING — cleaner in JSON output

class Category(str, Enum):
    BILLING   = "billing"
    TECHNICAL = "technical"
    GENERAL   = "general"
    REFUND    = "refund"


class Priority(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class Provider(str, Enum):
    OPENAI    = "openai"
    ANTHROPIC = "anthropic"


# ── Pydantic models ────────────────────────────────────────────────────────────
# Pydantic is a validation library.
# When you do TicketClassification(**data), it checks:
#   - every required field is present
#   - every field has the right type
#   - every field is within the allowed range (ge/le)
# If anything is wrong it raises a ValidationError with a clear message.
# This protects you from bad LLM output before it reaches the rest of your code.

class TicketClassification(BaseModel):
    category: Category = Field(
        description="Type of support request"
    )
    priority: Priority = Field(
        description="Urgency level of the ticket"
    )
    sentiment_score: float = Field(
        description="Customer sentiment: -1.0=furious, 0.0=neutral, 1.0=delighted",
        ge=-1.0,   # ge = greater than or equal
        le=1.0,    # le = less than or equal
    )
    suggested_reply: str = Field(
        description="Draft reply to send to the customer, 2-4 sentences, professional tone"
    )
    estimated_resolution_hours: int = Field(
        description="Hours to resolve: 1=minutes, 4=same day, 24=next day, 72=complex",
        ge=1,
    )
    confidence: float = Field(
        description="How confident you are in this classification, 0.0 to 1.0",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(
        description="One sentence explaining why you chose this category and priority"
    )

    @field_validator("sentiment_score", "confidence", mode="before")
    @classmethod
    def round_floats(cls, v: object) -> float:
        # LLMs sometimes return 0.7499999 or the string "0.75"
        # mode="before" means this runs before Pydantic's type check
        # so it handles both float and string inputs
        return round(float(v), 2)


class UsageLog(BaseModel):
    """Token usage and cost — written to logs/usage.jsonl after every call."""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    latency_seconds: float
    timestamp: str


# ── Cost table ─────────────────────────────────────────────────────────────────
# Prices are USD per 1,000 tokens (input and output billed separately)
# Update this dict when providers change their prices

COST_TABLE = {
    "gpt-4o-mini":               {"input": 0.00015, "output": 0.00060},
    "gpt-4o":                    {"input": 0.00500, "output": 0.01500},
    "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
    "claude-sonnet-4-20250514":  {"input": 0.00300, "output": 0.01500},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = COST_TABLE.get(model, {"input": 0.001, "output": 0.002}) # default rates if model not in table
    cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates["output"]
    return round(cost, 6)


# ── System prompt ──────────────────────────────────────────────────────────────
# This is the instruction set sent to the LLM before every email.
# It defines: the role, the exact output format, and the decision rules.
# Changing this is "prompt engineering" — the most impactful thing you can do
# when the classifier gets something wrong.

SYSTEM_PROMPT = """\
You are an expert customer support ticket classifier for a SaaS company.

Analyze the support email and return a JSON object with EXACTLY these fields:

{
  "category": "billing" | "technical" | "general" | "refund",
  "priority": "low" | "medium" | "high" | "critical",
  "sentiment_score": <float -1.0 to 1.0>,
  "suggested_reply": "<2-4 sentence professional draft reply>",
  "estimated_resolution_hours": <integer, minimum 1>,
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence explaining category and priority choice>"
}

CATEGORY RULES:
  billing   → charges, invoices, plans, pricing questions
  technical → bugs, errors, broken features, API issues, integrations
  refund    → requests to reverse a charge or get money back
  general   → everything else: how-to questions, feature requests, compliments

PRIORITY RULES:
  critical → production down, data loss, security breach, legal threat
  high     → core feature broken, payment failed, significant revenue impact
  medium   → partial issue, billing question, refund request
  low      → general question, feature request, minor cosmetic bug

Return ONLY valid JSON. No markdown fences. No explanation before or after.\
"""


# ── Helper: strip markdown fences ─────────────────────────────────────────────
def _strip_fences(text: str) -> str:
    """
    Some models wrap their JSON in ```json ... ``` despite being told not to.
    This removes those fences so json.loads() can parse the response.

    Input:  ```json\n{"category": "billing"}\n```
    Output: {"category": "billing"}
    """
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1]
        if inner.startswith("json"):
            inner = inner[4:]
        return inner.strip()
    return text


# ── OpenAI provider ────────────────────────────────────────────────────────────
def classify_with_openai(
    email_text: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> tuple[TicketClassification, UsageLog]:

    from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    for attempt in range(max_retries):
        try:
            start = time.monotonic()

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Classify this support email:\n\n{email_text}"},
                ],
                temperature=0.1,                           # low = more consistent outputs
                timeout=30,                              # seconds before we consider the request timed out    
                response_format={"type": "json_object"},   # forces JSON-only output
            )

            latency = round(time.monotonic() - start, 2)
            raw = response.choices[0].message.content
            data = json.loads(_strip_fences(raw))
            classification = TicketClassification(**data)

            usage = response.usage
            log = UsageLog(
                provider="openai",
                model=model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                estimated_cost_usd=estimate_cost(model, usage.prompt_tokens, usage.completion_tokens),
                latency_seconds=latency,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return classification, log

        except RateLimitError as e:
            # Exponential backoff: attempt 0 → wait 1s, attempt 1 → 2s, attempt 2 → 4s
            wait = 2 ** attempt
            logger.warning(f"[OpenAI] Rate limit, attempt {attempt + 1}. Waiting {wait}s. {e}")
            time.sleep(wait)

        except APITimeoutError as e:
            logger.warning(f"[OpenAI] Timeout, attempt {attempt + 1}. {e}")
            if attempt == max_retries - 1:
                raise

        except APIConnectionError as e:
            logger.error(f"[OpenAI] Connection error — check your API key and network. {e}")
            raise

        except json.JSONDecodeError as e:
            raise ValueError(f"[OpenAI] Malformed JSON response.\nRaw: {raw}\nError: {e}")

    raise RuntimeError(f"[OpenAI] Failed after {max_retries} retries.")


# ── Anthropic provider ─────────────────────────────────────────────────────────
def classify_with_anthropic(
    email_text: str,
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
) -> tuple[TicketClassification, UsageLog]:

    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for attempt in range(max_retries):
        try:
            start = time.monotonic()

            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Classify this support email:\n\n{email_text}"},
                ],
            )

            latency = round(time.monotonic() - start, 2)
            raw = response.content[0].text
            data = json.loads(_strip_fences(raw))
            classification = TicketClassification(**data)

            usage = response.usage
            log = UsageLog(
                provider="anthropic",
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                estimated_cost_usd=estimate_cost(model, usage.input_tokens, usage.output_tokens),
                latency_seconds=latency,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            return classification, log

        except anthropic.RateLimitError as e:
            wait = 2 ** attempt
            logger.warning(f"[Anthropic] Rate limit, attempt {attempt + 1}. Waiting {wait}s. {e}")
            time.sleep(wait)

        except anthropic.APITimeoutError as e:
            logger.warning(f"[Anthropic] Timeout, attempt {attempt + 1}. {e}")
            if attempt == max_retries - 1:
                raise

        except anthropic.APIConnectionError as e:
            logger.error(f"[Anthropic] Connection error — check your API key and network. {e}")
            raise

        except json.JSONDecodeError as e:
            raise ValueError(f"[Anthropic] Malformed JSON response.\nRaw: {raw}\nError: {e}")

    raise RuntimeError(f"[Anthropic] Failed after {max_retries} retries.")


# ── Public function ────────────────────────────────────────────────────────────
def classify_ticket(
    email_text: str,
    provider: Provider = Provider.ANTHROPIC,
    model: Optional[str] = None,
    log_to_file: bool = True,
) -> dict:
    """
    Classify a raw support email. This is the only function you need to call.

    Args:
        email_text:  The full body of the support email.
        provider:    Provider.ANTHROPIC (default) or Provider.OPENAI
        model:       Override the default model. None = use provider default.
        log_to_file: Write usage stats to logs/usage.jsonl (default True)

    Returns:
        dict with two keys:
          "classification" → category, priority, sentiment, reply, etc.
          "usage"          → tokens, cost, latency, timestamp

    Raises:
        ValueError:   Empty email, or LLM returned malformed JSON.
        RuntimeError: All retries exhausted.
    """
    if not email_text or not email_text.strip():
        raise ValueError("email_text cannot be empty.")

    logger.info(f"Classifying ticket via {provider.value}...")

    if provider == Provider.OPENAI:
        chosen_model = model or "gpt-4o-mini"
        classification, usage_log = classify_with_openai(email_text, model=chosen_model)
    else:
        chosen_model = model or "claude-haiku-4-5-20251001"
        classification, usage_log = classify_with_anthropic(email_text, model=chosen_model)

    logger.info(
        f"✓  category={classification.category.value}  "
        f"priority={classification.priority.value}  "
        f"sentiment={classification.sentiment_score:+.2f}  "
        f"resolve={classification.estimated_resolution_hours}h  "
        f"tokens={usage_log.total_tokens}  "
        f"cost=${usage_log.estimated_cost_usd:.5f}  "
        f"latency={usage_log.latency_seconds}s"
    )

    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        with open("logs/usage.jsonl", "a", encoding="utf-8") as f:
            f.write(usage_log.model_dump_json() + "\n")

    return {
        "classification": classification.model_dump(),
        "usage": usage_log.model_dump(),
    }