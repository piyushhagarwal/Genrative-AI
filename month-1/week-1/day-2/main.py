from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Literal, List
import json 
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ── Test ticket ───────────────────────────────────────────────────────────────

ticket = """
Ticket ID: TKT-001
I've been waiting 3 weeks for my refund. Nobody is responding to my emails.
This is completely unacceptable. I want this resolved TODAY.
"""

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH A — JSON Mode
# You tell the model "return JSON". The model decides the shape.
# Problem: today it returns "priority", tomorrow it might return "urgency_level"
# ══════════════════════════════════════════════════════════════════════════════

def approach_a_json_mode():
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="You are a support assistant.",
        input=ticket + "Give me a JSON object with the ticket's category, priority, sentiment, and a one-line summary.",
        text={
            "format": {"type": "json_object"}   # tells model: output must be valid JSON
        }
    )

    # output_text is a raw string — you parse it yourself
    data = json.loads(response.output_text)

    print("── Approach A: JSON Mode ──────────────────────")
    print(f"Raw response : {data}")
    print(f"Priority     : {data.get('priority', 'KEY MISSING — model used a different name')}")

# ══════════════════════════════════════════════════════════════════════════════
# APPROACH B — Pydantic Structured Outputs
# You define exactly what you want. The SDK enforces it.
# "priority" will ALWAYS be one of: low, medium, high, critical. No surprises.
# ══════════════════════════════════════════════════════════════════════════════

# ── Nested model ──────────────────────────────────────────────────────────────
# A nested model is just a Pydantic class used as a field type inside another class.
# Here, each action item is its own validated object — not a plain string or dict.

class ActionItem(BaseModel):
    action         : str                                                        # what needs to be done
    owner          : Literal["support_agent", "billing_team", "engineering"]   # who does it
    deadline_hours : int                                                        # by when


# ── Main schema ───────────────────────────────────────────────────────────────

class TicketAnalysis(BaseModel):
    category         : Literal["billing", "technical", "shipping", "account", "other"]
    priority         : Literal["low", "medium", "high", "critical"]
    sentiment        : Literal["positive", "neutral", "frustrated", "angry"]
    summary          : str = Field(description="One sentence summary of the issue")
    reply            : str = Field(description="Professional, empathetic reply under 80 words")
    resolution_hours : int
    needs_escalation : bool
    action_items     : List[ActionItem]   # ← nested model — a list of ActionItem objects


def approach_b_pydantic():
    response = client.responses.parse(
        model="gpt-4o-mini",
        instructions="You are an expert support ticket analyzer. Analyze the ticket and return structured data.",
        input=ticket,
        text_format=TicketAnalysis,     # SDK converts this class (+ nested ActionItem) to JSON schema
    )

    result = response.output_parsed

    print("── Approach B: Pydantic ───────────────────────")
    print(f"Category  : {result.category}")
    print(f"Priority  : {result.priority}")
    print(f"Sentiment : {result.sentiment}")
    print(f"Escalate  : {result.needs_escalation}")
    print(f"ETA       : {result.resolution_hours}h")
    print(f"Summary   : {result.summary}")
    print(f"Reply     : {result.reply}")

    # Each action_item is a fully typed ActionItem object — not a dict
    print(f"\nAction Items:")
    for item in result.action_items:
        print(f"  → [{item.owner}] {item.action} (within {item.deadline_hours}h)")

    # .model_dump() converts everything — including nested objects — to a plain dict
    print(f"\nAs dict : {json.dumps(result.model_dump(), indent=2)}")


# ── Run both ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    approach_a_json_mode()
    print()
    approach_b_pydantic()
