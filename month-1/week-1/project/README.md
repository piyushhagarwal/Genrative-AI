# Support Ticket Classifier

Classifies raw customer support emails into structured JSON using OpenAI or Anthropic. A single flag switches between providers.

---

## Output

```json
{
  "classification": {
    "category": "billing",
    "priority": "high",
    "sentiment_score": -0.72,
    "suggested_reply": "Hi Sarah, I've confirmed the duplicate charge and initiated a refund of $49.99. It will appear within 3–5 business days. Apologies for the inconvenience.",
    "estimated_resolution_hours": 4,
    "confidence": 0.95,
    "reasoning": "Billing category due to duplicate charge; high priority because it involves a financial error on a paying account."
  },
  "usage": {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "input_tokens": 312,
    "output_tokens": 178,
    "total_tokens": 490,
    "estimated_cost_usd": 0.000301,
    "latency_seconds": 1.24,
    "timestamp": "2024-02-01T14:32:11+00:00"
  }
}
```

---

## Project structure

```
support-ticket-classifier/
├── classifier.py          ← everything: models, providers, classify_ticket()
├── main.py                ← CLI entry point
├── tests/
│   └── test_emails.py     ← 20 real-world emails + batch runner
├── logs/                  ← created at runtime
│   ├── usage.jsonl        ← one line per API call
│   └── test_results_*.json
├── .env.example           ← copy to .env, fill in your keys
├── .gitignore
├── pyproject.toml         ← dependencies (replaces requirements.txt)
└── README.md
```

---

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
cd month-1/week-1/project
uv sync
```

### 3. Set your API keys

```bash
cp .env.example .env
# open .env and fill in your keys
```

---

## Usage

```bash
# Inline email
uv run python main.py --provider anthropic --email "I was charged twice this month"

# From a file
uv run python main.py --provider openai --file email.txt

# Pipe from stdin
echo "Service is down!" | uv run python main.py --provider anthropic

# Use a specific model
uv run python main.py --provider anthropic --model claude-sonnet-4-20250514 --email "..."
```

### Run the 20-email test suite

```bash
uv run python tests/test_emails.py
uv run python tests/test_emails.py --provider openai
```

---

## Supported models

| Provider  | Default                     | Alternatives               |
| --------- | --------------------------- | -------------------------- |
| Anthropic | `claude-haiku-4-5-20251001` | `claude-sonnet-4-20250514` |
| OpenAI    | `gpt-4o-mini`               | `gpt-4o`                   |

---

## Cost

Every API call is logged to `logs/usage.jsonl`. Typical costs:

| Model              | Cost per ticket |
| ------------------ | --------------- |
| Anthropic Haiku    | ~$0.0003        |
| OpenAI GPT-4o-mini | ~$0.0001        |
| 20-email test run  | ~$0.003–$0.006  |

---

## Error handling

| Error            | What happens                             |
| ---------------- | ---------------------------------------- |
| Rate limit       | Exponential backoff, retries up to 3×    |
| Timeout          | Retries up to 3×, then raises            |
| Connection error | Raises immediately                       |
| Malformed JSON   | `ValueError` with the raw response       |
| Empty input      | `ValueError` before any API call is made |

---
