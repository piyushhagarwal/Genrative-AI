# Day 1 — OpenAI API Deep Dive

## Table of Contents

1. [How LLMs Actually Work](#1-how-llms-actually-work)
2. [Tokens & Tokenization](#2-tokens--tokenization)
3. [The API Structure](#3-the-api-structure)
4. [Temperature — The Deep Theory](#4-temperature--the-deep-theory)
5. [Every Parameter Explained](#5-every-parameter-explained)
6. [Message Roles](#6-message-roles)
7. [System Prompts — Your Most Powerful Lever](#7-system-prompts--your-most-powerful-lever)
8. [Cost & Pricing](#8-cost--pricing)
9. [Your 2-Hour Schedule](#9-your-2-hour-schedule)
10. [Day 1 Mastery Checklist](#10-day-1-mastery-checklist)

---

## 1. How LLMs Actually Work

### The Core Mental Model

A Large Language Model is fundamentally a **next-token predictor**.

Given all the text so far, it outputs a **probability distribution** over every possible next token. That's it. The entire sophistication of GPT-4 emerges from doing this one thing extremely well across billions of parameters.

When you send a message to the API, here's exactly what happens:

1. Your entire conversation is converted into a sequence of tokens
2. The model processes the full sequence through its transformer layers
3. It outputs a probability distribution: _"token A = 30% chance, token B = 25% chance..."_
4. A sampling strategy picks the next token from that distribution
5. That token is appended, and the whole process repeats until done

```
Input: ["The", "capital", "of", "France", "is"]
                    ↓
         Transformer (billions of params)
                    ↓
  Probability distribution:
    "Paris"   → 0.82  ████████████████████
    "Lyon"    → 0.05  ██
    "London"  → 0.02  █
    ...thousands more tokens...
                    ↓
  Sampling picks "Paris" → append → repeat
```

> 💡 **Why this matters:** Every parameter you'll learn — temperature, top*p, penalties — is just controlling \_how* you sample from that probability distribution. Understand this and everything else clicks.

### The Context Window

The model can only "see" a fixed number of tokens at once — called the **context window**.

- GPT-4o: **128,000 tokens** (~96,000 words)
- Everything outside this window is invisible to the model
- Long conversations degrade because older messages get truncated
- You pay for input tokens because the model processes your **entire conversation history** on every single API call

---

## 2. Tokens & Tokenization

### What is a Token?

A token is **not a word**. It's a chunk of text that the model's vocabulary recognizes as a unit. GPT models use the `cl100k_base` encoding from `tiktoken`.

```
"Hello, world!"           → ["Hello", ",", " world", "!"]       = 4 tokens
"ChatGPT is amazing"      → ["Chat", "G", "PT", " is", " amazing"]  = 5 tokens
"tokenization"            → ["token", "ization"]                 = 2 tokens
"supercalifragilistic"    → ["super", "cali", "fra", "gil", "istic"] = 5 tokens
```

**Rule of thumb:** `1 token ≈ 0.75 words` in English. A 1000-word essay ≈ ~1,300 tokens.

> ⚠️ **Non-English text costs more.** Chinese, Japanese, and Arabic scripts tokenize less efficiently — a single character can be 2–3 tokens. This directly impacts API costs.

### Counting Tokens Before You Call

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

def analyze_tokens(text):
    tokens = enc.encode(text)
    decoded = [enc.decode([t]) for t in tokens]
    print(f"Text: {text}")
    print(f"Token count: {len(tokens)}")
    print(f"Tokens: {decoded}")
    print("---")

# Experiment with these
analyze_tokens("Hello world")
analyze_tokens("The quick brown fox jumps over the lazy dog")
analyze_tokens("def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)")
```

### Pricing by Model

| Model           | Input / 1M tokens | Output / 1M tokens | Context |
| --------------- | ----------------- | ------------------ | ------- |
| **gpt-4o-mini** | $0.15             | $0.60              | 128K    |
| **gpt-4o**      | $2.50             | $10.00             | 128K    |
| **gpt-4-turbo** | $10.00            | $30.00             | 128K    |

> 💰 **Use `gpt-4o-mini` for all Day 1 experiments.** It's 16x cheaper than gpt-4o and nearly as capable for learning. Your entire 2 hours of experimenting will cost under $0.10.

---

## 3. The API Structure

### Every Field in a Request

```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.responses.create(
    model="gpt-4o-mini",

    instructions="You are a helpful assistant.",  # was: {"role": "system", ...}
    input="Explain gravity in one sentence.",      # was: {"role": "user", ...}

    temperature=0.7,
    max_output_tokens=500,        # was: max_tokens
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=None,
)

print(response.output_text)
```

### Reading the Response Object

```python
response.choices[0].message.content    # The actual text output
response.usage.prompt_tokens           # Input tokens (what you sent)
response.usage.completion_tokens       # Output tokens (what was generated)
response.usage.total_tokens            # Total (= what you pay for)
response.model                         # Exact model version used
response.choices[0].finish_reason      # "stop", "length", "content_filter"
```

> ⚠️ **Always check `finish_reason`.** If it returns `"length"`, your response was cut off mid-sentence because it hit `max_tokens`. Your output is incomplete. In production, always handle this case.

---

## 4. Temperature — The Deep Theory

### The Math Behind It

Temperature is a mathematical operation applied to the model's **logits** (raw unnormalized scores) before sampling. It comes from statistical thermodynamics.

**Formula:** `P(token) = exp(logit / T) / Σ exp(logits / T)`

Where **T is temperature**.

- **Low T (< 0.5):** Sharpens the distribution — top tokens dominate, rare tokens nearly vanish
- **High T (> 1.0):** Flattens the distribution — all tokens become more equally likely
- **T = 0:** Effectively always picks the highest-probability token (greedy decoding)

### Visualized

```
Original logits after "The sky is":
  "blue"   5.2,  "vast"  4.1,  "clear"  3.8,  "dark"  2.1

Temperature = 0.1  (very focused)
  "blue"  0.998  ████████████████████████████████████████
  "vast"  0.001  ░
  → Almost always picks "blue". Near-deterministic.

Temperature = 0.7  (balanced)
  "blue"  0.72   █████████████████████████████
  "vast"  0.15   ██████
  "clear" 0.11   ████
  "dark"  0.02   █
  → Usually "blue" but sometimes creative alternatives.

Temperature = 2.0  (near-chaos)
  "blue"  0.35   ██████████████
  "vast"  0.28   ███████████
  "clear" 0.26   ██████████
  "dark"  0.11   ████
  → All tokens nearly equal. Output degrades fast.
```

### The Experiment — Run This

```python
def call(temp, n=5, prompt="Give me a one-line creative metaphor for learning."):
    print(f"\n{'='*60}\nTEMPERATURE = {temp}\n{'='*60}")
    for i in range(n):
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=60
        )
        print(f"[{i+1}] {r.choices[0].message.content.strip()}")

call(0)    # Nearly identical outputs every time
call(0.7)  # Meaningful variety, still coherent
call(1.5)  # High creativity, sometimes incoherent

# Now test with a factual prompt
call(0, n=3, prompt="What year was Python created?")   # Correct every time
call(1, n=3, prompt="What year was Python created?")   # May hallucinate!
```

### When to Use Which Temperature

| Temperature | Behavior                  | Best For                                       | Avoid For                  |
| ----------- | ------------------------- | ---------------------------------------------- | -------------------------- |
| `0`         | Near-deterministic        | Code generation, data extraction, factual Q&A  | Creative writing           |
| `0.3`       | Focused, minimal variance | Summarization, translation, structured outputs | Brainstorming              |
| `0.7`       | Balanced (default)        | General chat, explanations, drafting           | JSON output                |
| `1.0`       | Creative, varied          | Brainstorming, marketing copy, ideation        | Factual tasks              |
| `1.5+`      | Unpredictable             | Experimental creative tasks only               | Anything needing coherence |

---

## 5. Every Parameter Explained

### `max_tokens` — Widely Misunderstood

**max_tokens does NOT limit how much text the model reads.** It only caps the number of tokens the model _generates_ in its response.

- Your input can be 50,000 tokens — max_tokens doesn't affect that
- If the model hits the limit before finishing, the response is cut off
- Set it slightly higher than you expect — the model stops on its own when done

### `top_p` — Nucleus Sampling

Instead of scaling the entire distribution (temperature), top_p considers only the smallest set of tokens whose probabilities add up to `p`.

```
top_p = 0.9 means:
  "Paris"   0.55  ████████████████████████████
  "Rome"    0.22  ████████████
  "London"  0.13  ████████        ← 0.55+0.22+0.13 = 0.90 ✓ stop here
  ─────────────── 90% threshold ──────────────────
  "Berlin"  0.06                  ← excluded
  ...rest   0.04                  ← excluded
```

> **Rule:** Use EITHER temperature OR top_p, not both. Alter one and leave the other at default (1.0).

### `frequency_penalty` vs `presence_penalty`

**frequency_penalty (0–2):** Reduces a token's probability based on _how many times_ it's already appeared. Scales with repetition — the more it's been used, the bigger the penalty. Good for reducing repetitive sentences.

**presence_penalty (0–2):** Reduces a token's probability if it has appeared _at all_, regardless of frequency. Binary — either it's been used or it hasn't. Good for encouraging new topics.

```python
# Observe the difference in longer outputs
def test_penalty(freq=0, pres=0):
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a 150-word story about a robot."}],
        frequency_penalty=freq,
        presence_penalty=pres,
        max_tokens=200
    )
    print(f"\nfreq={freq}, pres={pres}:\n{r.choices[0].message.content}")

test_penalty(0, 0)    # May repeat words/phrases
test_penalty(1.5, 0)  # Fewer repeated words
test_penalty(0, 1.5)  # More diverse topics introduced
```

### `stop` — Sequence Halting

The model stops generating the moment it produces any string in the `stop` list.

```python
# Stop before item 4 in a list
r = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 programming languages, one per line."}],
    stop=["\n4."],
    max_tokens=100
)

# Extract just a label from classification (no extra words)
r2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Sentiment (POSITIVE/NEGATIVE/NEUTRAL):\nI love this!\nAnswer:"}],
    stop=["\n"],   # Stops at newline → gets just the label
    max_tokens=10
)
```

### `n` — Multiple Completions

Generates `n` different responses in one API call. **Costs n times more** — the input prompt is processed n times internally. Useful for A/B testing outputs or programmatically picking the best of several options.

### `seed` — Reproducibility

Setting a fixed seed + temperature=0 gives near-identical outputs across runs. Essential for debugging and testing — a bug in your output becomes reproducible.

### `stream` — Real-Time Token Output

```python
# Get tokens back as they're generated (better UX for chatbots)
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## 6. Message Roles

### What the Model Actually "Sees"

The model doesn't see a structured conversation — it sees a **single long token sequence** with special markers separating each role. The roles have real weight because the model was fine-tuned to behave differently based on them.

```
[SYSTEM]    You are a helpful assistant. Be concise.
[USER]      What is recursion?
[ASSISTANT] Recursion is when a function calls itself...
[USER]      Give me an example in Python
[ASSISTANT] ← Model generates from here
```

### The Three Roles

| Role        | Purpose                                                        | When to Use                                   |
| ----------- | -------------------------------------------------------------- | --------------------------------------------- |
| `system`    | Sets behavior, persona, and constraints for the entire session | Always — first in your messages array         |
| `user`      | Represents what the human says                                 | Every human turn                              |
| `assistant` | Represents what the AI has previously said                     | Include past AI responses to maintain context |

### The API Has NO Memory — Build It Yourself

```python
conversation = [
    {"role": "system", "content": "You are a coding tutor. Be concise."}
]

def chat(user_message):
    conversation.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,  # Full history every time!
    )

    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})

    print(f"[{response.usage.total_tokens} tokens used] {reply}")

chat("What is a for loop?")
chat("Show me an example")
chat("What was my first question?")  # Works — history is passed

# Without history, the model has no context:
no_memory = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What was my first question?"}]
)
# Output: "I don't have context for previous questions."
```

> 💡 **Key insight:** Every API call is completely stateless. You are responsible for maintaining and passing conversation history. This also means token costs grow with each turn of a long conversation.

---

## 7. System Prompts — Your Most Powerful Lever

### Why They Matter So Much

The system prompt sets a "frame" the model uses to interpret everything that follows. Because it's processed first with special role weight, it shapes the probability distributions for every subsequent token. A great system prompt can transform mediocre output into excellent output — without changing the user message at all.

Think of the model as an extremely skilled actor. The system prompt is the director's brief. "You are a helpful assistant" tells the actor nothing. A real brief covers: who they are, their goal, constraints, style, audience, and output format.

### Weak vs Strong — Same User Message

**User message:** `"Explain async/await to me"`

**Weak system prompt:**

```
You are a helpful assistant.
```

**Strong system prompt:**

```
You are a senior backend engineer teaching junior developers.
Your explanations always follow this structure:
1. One-sentence ELI5 (explain like I'm 5)
2. The problem it solves (2 sentences max)
3. A real-world analogy
4. A working code example (10–15 lines, Python)
5. One common mistake beginners make

Be direct. No filler. Audience: developers with 6 months experience.
```

Run both and compare. The quality gap is dramatic — and the user message is identical.

### The Anatomy of an Excellent System Prompt

| Component       | What to Include                   | Example                                                 |
| --------------- | --------------------------------- | ------------------------------------------------------- |
| **Identity**    | WHO they are, their background    | "You are a senior UX writer at a Series B SaaS company" |
| **Goal**        | What they're trying to accomplish | "Your goal is to rewrite UI copy to improve conversion" |
| **Audience**    | Who they're speaking to           | "Users are non-technical SMB owners, aged 40–60"        |
| **Format**      | Exact output structure            | "Respond in JSON: {original, rewrite, reason}"          |
| **Constraints** | What NOT to do                    | "Never use jargon. Max 10 words per CTA."               |
| **Style**       | Tone, voice, energy               | "Tone: direct, confident, slightly playful"             |

### The Experiment

```python
user_msg = "Explain async/await to me"

# WEAK
weak = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": user_msg}
    ]
)

# STRONG
strong_system = """You are a senior backend engineer teaching junior developers.
Your explanations always follow this structure:
1. One-sentence ELI5
2. The problem it solves (2 sentences)
3. A real-world analogy
4. Working code example in Python (10-15 lines)
5. One common beginner mistake

Be direct. No filler. Max 200 words total."""

strong = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": strong_system},
        {"role": "user",   "content": user_msg}
    ]
)

print("=== WEAK ===")
print(weak.choices[0].message.content)
print("\n=== STRONG ===")
print(strong.choices[0].message.content)
```

> 💡 **Pro tip:** For complex system prompts, use XML-style tags like `<rules>`, `<format>`, `<examples>`. The model is trained on structured data and parses these very reliably.

> ⚠️ **The system prompt tax:** Your system prompt is sent as input tokens on _every single API call_. A 500-token system prompt × 10,000 calls/month = 5M extra input tokens. Optimize its length when you scale.

---

## 8. Cost & Pricing

### The Cost Estimator

```python
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},   # per 1M tokens
    "gpt-4o":      {"input": 2.50, "output": 10.00},
}

def project_monthly_cost(model, avg_input_tokens, avg_output_tokens, calls_per_day):
    p = PRICING[model]
    daily = (avg_input_tokens * p["input"] + avg_output_tokens * p["output"]) / 1_000_000 * calls_per_day
    print(f"Model: {model}")
    print(f"  Daily:   ${daily:.4f}")
    print(f"  Monthly: ${daily * 30:.2f}")
    print(f"  Yearly:  ${daily * 365:.2f}")

# Scenario: Customer support bot
# 500 input tokens (system + history + question), 200 output, 1000 calls/day
project_monthly_cost("gpt-4o-mini", 500, 200, 1000)
# → ~$0.45/day = ~$13.50/month

project_monthly_cost("gpt-4o", 500, 200, 1000)
# → ~$3.25/day = ~$97.50/month
```

### What Drives Costs Up

- Long system prompts (paid every call)
- Long conversation histories (grows each turn)
- Using `n > 1` (multiplies cost by n)
- High `max_tokens` (you don't pay for unused, but it invites longer output)
- Wrong model choice (gpt-4o is 16x more expensive than gpt-4o-mini per input token)

---

## 9. Your 2-Hour Schedule

### Setup (0:00 – 0:15)

- Get API key at `platform.openai.com`
- Store it in a `.env` file — never hardcode it
- Run `pip install openai tiktoken python-dotenv`
- Write a minimal API call and print the full response object

### Token Experiments (0:15 – 0:35)

- Run `tiktoken` on 5 different text types: a sentence, a paragraph, code, non-English text, a JSON blob
- Note how differently each tokenizes
- Manually estimate the cost of each using the pricing table

### Temperature Experiments (0:35 – 1:00)

- Run the temperature experiment: temperature 0, 0.7, and 1.5 — 5 times each
- Write down what you observe
- Repeat with a factual prompt and watch how high temperature causes hallucinations

### Message Roles & Context (1:00 – 1:20)

- Build the multi-turn conversation loop
- Ask something, get a reply, ask a follow-up that requires context
- Then call WITHOUT passing history and see it fail completely
- This single moment is a crucial intuition-builder

### System Prompt Deep Work (1:20 – 1:50)

- Choose a task relevant to your own work
- Write the worst possible system prompt, run it, read the output
- Spend 10 minutes crafting the best possible system prompt using the anatomy table
- Run it, compare — the difference should be dramatic

### Reflection (1:50 – 2:00)

- Write down 5 things you now understand that you didn't before
- Note what surprised you most
- Write down your open questions for Day 2

---

## 10. Day 1 Mastery Checklist

You've genuinely mastered Day 1 if you can say **YES** to all of these without hesitation:

- [ ] I can explain that an LLM is a next-token predictor working on probability distributions
- [ ] I know that a token ≠ a word, and can estimate token count from word count (~1.33x)
- [ ] I have run tiktoken and seen how different text types tokenize differently
- [ ] I understand that temperature scales the logit distribution before sampling
- [ ] I have personally seen temperature=0 produce near-identical outputs
- [ ] I have personally seen temperature=1.5 produce creative/incoherent outputs
- [ ] I know max_tokens controls OUTPUT length only, not input
- [ ] I know what finish_reason="length" means (truncation) and why it matters
- [ ] I can explain the difference between frequency_penalty and presence_penalty
- [ ] I understand the three message roles: system, user, assistant
- [ ] I know the API has NO memory — I must pass full history manually every call
- [ ] I have written both a weak and strong system prompt and compared their outputs
- [ ] I can estimate the monthly cost of an API-powered product before building it
- [ ] I know the system prompt costs input tokens on every single API call

---

## Resources

| Resource                                                                               | What For                             |
| -------------------------------------------------------------------------------------- | ------------------------------------ |
| [OpenAI Chat API Reference](https://platform.openai.com/docs/api-reference/chat)       | Every parameter in detail            |
| [OpenAI Playground](https://platform.openai.com/playground)                            | Visual testing without code          |
| [tiktoken on GitHub](https://github.com/openai/tiktoken)                               | Token counting library               |
| [OpenAI Pricing](https://openai.com/api/pricing)                                       | Always verify current prices here    |
| [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | System prompt patterns from OpenAI   |
| [Tokenizer Tool](https://platform.openai.com/tokenizer)                                | Visualize tokenization interactively |

---

> **You're ready for Day 2 when** you can explain temperature, tokens, and message structure to someone else from memory. Real understanding means you can teach it.
>
> **Day 2 covers:** Structured Outputs (JSON mode), Function Calling, Streaming, and building your first real mini-app.
