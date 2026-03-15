"""
main.py
-------
CLI entry point. Run with:

    uv run python main.py --provider anthropic --email "I was charged twice"
    uv run python main.py --provider openai --file email.txt
    echo "Service is down!" | uv run python main.py --provider anthropic
"""

import argparse
import json
import sys

from classifier import classify_ticket, Provider


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="classify",
        description="Classify a customer support email into structured JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  uv run python main.py --provider anthropic --email "I was charged twice"
  uv run python main.py --provider openai --file email.txt
  echo "Service is down!" | uv run python main.py --provider anthropic
        """,
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default="anthropic",
        help="Which LLM provider to use (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the default model (optional)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Email text to classify, passed inline as a string",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a .txt file containing the email",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip writing usage stats to logs/usage.jsonl",
    )

    args = parser.parse_args()

    # Determine the email source: --email flag, --file flag, or stdin pipe
    if args.email:
        email_text = args.email
    elif args.file:
        try:
            with open(args.file, encoding="utf-8") as f:
                email_text = f.read()
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    elif not sys.stdin.isatty():
        email_text = sys.stdin.read()
    else:
        parser.print_help()
        sys.exit(0)

    email_text = email_text.strip()
    if not email_text:
        print("Error: email text is empty.", file=sys.stderr)
        sys.exit(1)

    try:
        result = classify_ticket(
            email_text,
            provider=Provider(args.provider),
            model=args.model,
            log_to_file=not args.no_log,
        )
        print(json.dumps(result, indent=2))

    except ValueError as e:
        print(f"Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()