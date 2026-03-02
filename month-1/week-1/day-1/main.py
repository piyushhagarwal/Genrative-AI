import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# ── 1. TOKENS ────────────────────────────────────────────────────────────────
print("\n── 1. TOKENS ──")
enc = tiktoken.encoding_for_model("gpt-4o")
text = "The quick brown fox jumps over the lazy dog"
tokens = enc.encode(text)
print(f"Text    : {text}")
print(f"Count   : {len(tokens)} tokens")
print(f"Split   : {[enc.decode([t]) for t in tokens]}")

# ── 2. BASIC CALL ─────────────────────────────────────────────────────────────
print("\n── 2. BASIC CALL ──")
r = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant. Be concise.",
    input="What is a Python decorator in one sentence?",
)
print(f"Output : {r.output_text}")
print(f"Tokens : {r.usage.input_tokens} in / {r.usage.output_tokens} out")

# ── 3. TEMPERATURE ────────────────────────────────────────────────────────────
print("\n── 3. TEMPERATURE (same prompt, 3 different temps) ──")
prompt = "Give me a one-line metaphor for learning to code."
for temp in [0, 0.7, 1.5]:
    r = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        temperature=temp,
        max_output_tokens=50,
    )
    print(f"  temp={temp}: {r.output_text.strip()}")

# ── 4. TEMPERATURE = 0 (run 3 times → nearly identical) ──────────────────────
print("\n── 4. TEMPERATURE=0 called 3 times (notice: near-identical) ──")
for i in range(3):
    r = client.responses.create(
        model="gpt-4o-mini",
        input="What is recursion? One sentence.",
        temperature=0,
        max_output_tokens=40,
    )
    print(f"  [{i+1}] {r.output_text.strip()}")

# ── 5. WEAK vs STRONG INSTRUCTIONS ───────────────────────────────────────────
print("\n── 5. WEAK vs STRONG instructions= ──")
question = "Explain async/await to me"

weak = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful assistant.",
    input=question,
    max_output_tokens=150,
)

strong = client.responses.create(
    model="gpt-4o-mini",
    instructions="""You are a senior Python engineer teaching beginners.
Always structure your answer as:
1. One-line ELI5
2. The problem it solves (one sentence)
3. A short code example""",
    input=question,
    max_output_tokens=200,
)

print(f"\nWEAK:\n{weak.output_text}")
print(f"\nSTRONG:\n{strong.output_text}")
