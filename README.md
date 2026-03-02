# Generative AI Developer Roadmap

**Month 1** — You understood LLMs and RAG at a mechanical level. You stopped being a developer who uses AI and started being a developer who understands AI.

**Month 2** — You built agents that do real autonomous work. You understand how they fail and how to make them recover.

**Month 3** — You built production-grade systems — fast, accurate, observable, and secure. You can measure quality and improve it systematically.

**Month 4** — You added voice, multimodal, and workflow automation. You have a portfolio of 10+ projects that prove capability rather than claiming it.

**Month 5** — You can ship to real clients professionally. You deploy on AWS, automate your deployments, integrate any tool via MCP, and hand off systems with documentation and dashboards that make clients feel taken care of.

# Setting Up and Running the Project with uv

This guide explains how to download `uv`, set up the project environment, and run it.

## What is `uv`?

`uv` is an extremely fast Python package installer and resolver, written in Rust. It's significantly faster than traditional tools like `pip` and makes dependency management seamless.

---

## Step 1: Installation

### macOS Installation

Using Homebrew (recommended):

```bash
brew install uv
```

### Verify Installation

```bash
uv --version
```

You should see the version number displayed.

---

## Step 2: How to add dependencies to the project

```bash
uv add openai python-dotenv tiktoken
```

## Step 3: Install Dependencies from already existing `pyproject.toml`

```bash
uv sync
```

## Step 4: Run the project

```bash
uv run main.py
```

This will automatically create a virtual environment and run the script.

---
