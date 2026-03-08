**Day 2 — Structured Outputs (this is critical for production)**

Raw text responses from LLMs are useless in production systems. You need JSON. Learn two approaches and understand when to use each.

First approach: JSON mode. Pass `response_format: { type: "json_object" }` and instruct the model in your prompt to return JSON. Simple but fragile — the model decides the schema.

Second approach: Structured outputs with Pydantic. This is what you'll use in production. Define a Pydantic model, pass it to the API, get a guaranteed schema back. The API literally won't return anything that doesn't match your schema.

## How to run the code

1. Install `uv` (see README for instructions).
2. Run `uv sync` to install dependencies.
3. Create a `.env` file with your OpenAI API key, check example.env for reference.
4. Run `uv run main.py` to see both approaches in action.

---

## What to Notice When You Run This

When Approach A runs, look at the raw dict. Run it a few times. Notice how the model sometimes changes the keys slightly — "priority" might become "urgency" or the values might be capitalised differently. Now imagine that inconsistency buried inside a production system at 2am.

When Approach B runs, look at how you access the data. `result.priority`. `result.needs_escalation`. `result.action_items[0].owner`. You're not doing string parsing. You're not checking if keys exist. You're using a typed Python object the same way you'd use any other class instance.

That's the shift. From parsing text and hoping, to working with objects and knowing.

---

## Key Takeaways

**JSON mode is a hint, not a contract.** The model tries to match your description but there are no guarantees on field names, capitalisation, or structure. Use it for exploration, not production.

**Pydantic is just a typed class.** Don't let the name intimidate you. You define fields with types, and Pydantic makes sure the data matches. That's the whole idea.

**`Literal` types eliminate value ambiguity.** Any field with a known set of options should use `Literal`. It removes an entire class of bugs before they can exist.

**`Field(description=...)` is documentation the AI reads.** The description goes into the JSON schema that the model receives. More context means better output, especially for free-text fields.

**Nested models represent nested reality.** Real data is hierarchical. Pydantic handles nesting naturally — define a class, use it as a field type, and the SDK takes care of the rest.

**`parse()` = `create()` + schema generation + parsing.** It's a convenience wrapper. The Pydantic class goes in, a typed object comes out. `model_dump()` converts it back to a dict when you need to talk to the rest of the world.

---

## The Mental Model

```
JSON mode     →  valid JSON, unknown shape  →  dict access, potential KeyErrors
Pydantic      →  guaranteed schema          →  attribute access, no surprises

parse()  =  create()  +  schema generation  +  json.loads()  +  Pydantic validation
```
