# Generative AI Developer Roadmap

[Plan](https://www.notion.so/Generative-AI-roadmap-31667a21d22d8062a208f59086e32d28?source=copy_link)

## 🗺️ The Roadmap

| Month       | Theme                         | Core Skills                                              |
| ----------- | ----------------------------- | -------------------------------------------------------- |
| **Month 1** | LLM Mastery + RAG Foundations | LLM APIs, Prompt Engineering, RAG from Scratch, Evals    |
| **Month 2** | AI Agents                     | ReAct, Tool Use, Multi-Agent, Browser Automation, Memory |
| **Month 3** | Production Systems            | Advanced RAG, Observability, Security, Fine-tuning       |
| **Month 4** | Advanced Capabilities         | Voice AI, Multimodal, Workflow Automation, Portfolio     |
| **Month 5** | Cloud + Client Delivery       | AWS, CI/CD, MCP, Client Handoff                          |

---

## Setting Up and Running the Project with uv

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
