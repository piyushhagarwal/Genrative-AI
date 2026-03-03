**Day 2 — Structured Outputs (this is critical for production)**

Raw text responses from LLMs are useless in production systems. You need JSON. Learn two approaches and understand when to use each.

First approach: JSON mode. Pass `response_format: { type: "json_object" }` and instruct the model in your prompt to return JSON. Simple but fragile — the model decides the schema.

Second approach: Structured outputs with Pydantic. This is what you'll use in production. Define a Pydantic model, pass it to the API, get a guaranteed schema back. The API literally won't return anything that doesn't match your schema.

## How to run the code

1. Install `uv` (see README for instructions).
2. Run `uv sync` to install dependencies.
3. Create a `.env` file with your OpenAI API key, check example.env for reference.
4. Run `uv run main.py` to see both approaches in action.
