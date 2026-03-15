"""
tests/test_emails.py
--------------------
20 realistic support emails + batch runner.

Run with:
    uv run python tests/test_emails.py
    uv run python tests/test_emails.py --provider openai

The 20 emails deliberately cover edge cases:
  - Angry emails with zero detail          (id 16)
  - Multi-issue tickets                    (id 17)
  - Charged after cancellation             (id 15) → billing or refund?
  - Suspicious third-party refund request  (id 20) → potential fraud
  - Mixed French/English                   (id 18)
  - Security breach                        (id 8)
  - Churn risk disguised as a question     (id 19)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import classify_ticket, Provider


TEST_EMAILS = [

    # ── BILLING ───────────────────────────────────────────────────────────────
    {
        "id": 1, "expected_category": "billing", "expected_priority": "medium",
        "label": "billing — double charge",
        "email": (
            "Subject: Double charged this month\n\n"
            "Hi, I just noticed my credit card was charged twice for my monthly subscription — "
            "once on the 1st and again on the 3rd. Both show as $49.99.\n\n"
            "My account email is sarah.jones@example.com. Can you please refund one charge?\n\n"
            "Thanks, Sarah"
        ),
    },
    {
        "id": 2, "expected_category": "billing", "expected_priority": "high",
        "label": "billing — enterprise 10x overcharge",
        "email": (
            "Subject: INVOICE #4521 — Wrong amount billed\n\n"
            "We are on the Enterprise plan ($299/mo) and this month's invoice shows $2,990. "
            "That is a 10x overcharge and is causing cash flow issues.\n\n"
            "Please correct immediately. Account ID: ENT-00823.\n\n"
            "Regards, Michael Chen, CFO"
        ),
    },
    {
        "id": 3, "expected_category": "general", "expected_priority": "low",
        "label": "billing — student discount question",
        "email": (
            "Hey, do you offer student discounts? "
            "I'm at university and $49/month is a bit steep. "
            "Just checking before I cancel.\n\nThanks, Jordan"
        ),
    },

    # ── TECHNICAL ─────────────────────────────────────────────────────────────
    {
        "id": 4, "expected_category": "technical", "expected_priority": "critical",
        "label": "technical — full production outage",
        "email": (
            "URGENT: Our entire production system is DOWN.\n\n"
            "Every request returns 503. We have been down 2 hours and are losing thousands "
            "per minute. This affects ALL customers. We need someone on the phone NOW.\n\n"
            "Account: enterprise-client-99"
        ),
    },
    {
        "id": 5, "expected_category": "technical", "expected_priority": "high",
        "label": "technical — 30% API error rate",
        "email": (
            "Subject: API returning 500 errors intermittently\n\n"
            "About 30% of our calls to /v2/data/export are failing with HTTP 500 since 3 hours ago. "
            "This is breaking our data pipeline.\n\n"
            "API key: sk-prod-...abc123 | Region: us-east-1\n\nPlease advise ASAP."
        ),
    },
    {
        "id": 6, "expected_category": "technical", "expected_priority": "medium",
        "label": "technical — Slack integration not working",
        "email": (
            "Hi, I can't connect the Slack integration. After clicking Authorize I get "
            "redirected back with no confirmation. Tried Chrome and Firefox, cleared cookies.\n\n"
            "Account: free_trial_user_442"
        ),
    },
    {
        "id": 7, "expected_category": "technical", "expected_priority": "low",
        "label": "technical — dark mode cosmetic bug",
        "email": (
            "Hello, the dark mode sidebar has dark gray text on a dark background — hard to read. "
            "Not urgent but wanted to flag it.\n\nSafari on macOS Sonoma 14.2. Cheers, Alex"
        ),
    },
    {
        "id": 8, "expected_category": "technical", "expected_priority": "critical",
        "label": "technical — security breach / compromised keys",
        "email": (
            "SECURITY INCIDENT — Immediate Response Required\n\n"
            "We believe our API keys were compromised. We are seeing calls from unknown IPs "
            "making bulk data exports. Started 4 hours ago. Keys are rotated but we are "
            "concerned about what was accessed. We may have regulatory obligations.\n\n"
            "Contact: security@company.com | Account: PRO-55821"
        ),
    },

    # ── REFUND ────────────────────────────────────────────────────────────────
    {
        "id": 9, "expected_category": "refund", "expected_priority": "medium",
        "label": "refund — within 7-day guarantee",
        "email": (
            "Subject: Refund request — cancelled within trial period\n\n"
            "I signed up 3 days ago for Pro but it does not have the multi-user support I need. "
            "I would like a full refund — your site says 7-day money-back guarantee.\n\n"
            "Order #: ORD-20240115-8834 | tom.k@gmail.com"
        ),
    },
    {
        "id": 10, "expected_category": "refund", "expected_priority": "high",
        "label": "refund — 3 months of outages, threatening chargeback",
        "email": (
            "I want my money back. NOW.\n\n"
            "I have paid $792 over 8 months. Your service has had constant outages for 3 months. "
            "I submitted 6 tickets that were never resolved.\n\n"
            "I am demanding a $297 refund or I am filing a chargeback and leaving reviews "
            "on G2, Capterra, and Trustpilot.\n\n— David Rawlings"
        ),
    },
    {
        "id": 11, "expected_category": "refund", "expected_priority": "low",
        "label": "refund — accidental annual upgrade 20 min ago",
        "email": (
            "Hi, I accidentally upgraded to the annual plan instead of monthly while updating "
            "my payment method. I did this literally 20 minutes ago. Can you reverse it?\n\n"
            "Thanks, Emma"
        ),
    },

    # ── GENERAL ───────────────────────────────────────────────────────────────
    {
        "id": 12, "expected_category": "general", "expected_priority": "low",
        "label": "general — CSV export question",
        "email": (
            "Hi, does your platform support exporting data to CSV? "
            "Also, is there an API endpoint for bulk operations?\n\nThanks"
        ),
    },
    {
        "id": 13, "expected_category": "general", "expected_priority": "low",
        "label": "general — compliment + keyboard shortcut question",
        "email": (
            "Hello! I just wanted to say how much I love your product — it has saved me hours. "
            "Maria from support was exceptional last week.\n\n"
            "One question: is there a keyboard shortcut to create a new project?\n\n"
            "Warm regards, Priya Patel"
        ),
    },
    {
        "id": 14, "expected_category": "general", "expected_priority": "medium",
        "label": "general — GDPR / data residency blocking procurement",
        "email": (
            "Subject: Compliance question — GDPR data residency\n\n"
            "Our legal team needs to know where customer data is stored. We are a German company "
            "and need EU data residency for GDPR compliance.\n\n"
            "1. What regions do you store data in?\n"
            "2. Do you offer EU-only residency?\n"
            "3. Can you provide a DPA?\n\n"
            "This is blocking our procurement approval. — Lars M., Legal & Compliance"
        ),
    },

    # ── EDGE CASES ────────────────────────────────────────────────────────────
    {
        "id": 15, "expected_category": "refund", "expected_priority": "medium",
        "label": "edge — charged after cancellation (billing vs refund?)",
        "email": (
            "Subject: Charged after cancellation\n\n"
            "I cancelled on January 28th (confirmation #CX-44421) but was still charged $49 "
            "on February 1st. I want this charge reversed. I have the cancellation email.\n\n"
            "— R. Thompson"
        ),
    },
    {
        "id": 16, "expected_category": "technical", "expected_priority": "high",
        "label": "edge — angry with zero detail",
        "email": (
            "Your service is GARBAGE. It never works. I have wasted hours on this. "
            "FIX IT or I am leaving.\n\nSent from my iPhone"
        ),
    },
    {
        "id": 17, "expected_category": "technical", "expected_priority": "critical",
        "label": "edge — four issues in one email",
        "email": (
            "Subject: Multiple issues — please read carefully\n\n"
            "ISSUE 1 — Login: Since last Tuesday I am logged out every 15 minutes. "
            "Tried Chrome, Firefox, different computer — same problem.\n\n"
            "ISSUE 2 — Missing Data: About 30 projects created Jan 10-15 are gone from my dashboard. "
            "My backup shows they existed. I am worried data was deleted.\n\n"
            "ISSUE 3 — Billing: My January invoice shows 45,000 API calls; my logs show 31,000. "
            "That is a ~$14 difference.\n\n"
            "ISSUE 4 — Feature Request: Could you add bulk export?\n\n"
            "Customer since 2021. This month has been rough.\n\n"
            "Carla Mendes | Account: PRO-carla-2021"
        ),
    },
    {
        "id": 18, "expected_category": "billing", "expected_priority": "high",
        "label": "edge — mixed French/English, payment urgent before presentation",
        "email": (
            "Subject: Problem with payment / Problème de paiement\n\n"
            "Hello / Bonjour,\n\n"
            "My payment is not going through. I tried 3 credit cards. Error says 'payment declined' "
            "but my bank says no issue on their end.\n\n"
            "I need to upgrade before tomorrow for a client presentation.\n\n"
            "Merci / Thank you, Baptiste Girard"
        ),
    },
    {
        "id": 19, "expected_category": "general", "expected_priority": "high",
        "label": "edge — churn risk disguised as renewal question",
        "email": (
            "Hi, I am evaluating whether to renew our team plan ($599/year, due March 15). "
            "Honestly I am on the fence — core features work but we had 3 outages in 2 months "
            "and your competitor just launched something compelling.\n\n"
            "What are you offering for renewal? Any loyalty discounts?\n\n"
            "If I do not hear back by March 10th I will go with the competitor.\n\n"
            "— James Whitfield, Head of Operations"
        ),
    },
    {
        "id": 20, "expected_category": "refund", "expected_priority": "low",
        "label": "edge — suspicious third-party refund to different account",
        "email": (
            "Hello, I am reaching out on behalf of a customer who wishes to remain anonymous. "
            "They purchased a subscription last month and would like a refund sent to a different "
            "payment method — specifically a bank wire transfer.\n\n"
            "The original purchaser cannot contact you directly at this time. "
            "Please process the refund of $299 to: [bank details would go here]\n\n"
            "Thank you for your prompt handling of this matter."
        ),
    },
]


def run_tests(provider: Provider) -> None:
    print(f"\n{'═' * 70}")
    print(f"  Support Ticket Classifier — Test Suite ({len(TEST_EMAILS)} emails)")
    print(f"  Provider : {provider.value}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 70}\n")

    results = []
    failures = []
    total_cost = 0.0
    total_tokens = 0

    for item in TEST_EMAILS:
        print(f"[{item['id']:02d}/20] {item['label']}")
        try:
            result = classify_ticket(item["email"], provider=provider, log_to_file=True)
            c = result["classification"]
            u = result["usage"]

            total_cost += u["estimated_cost_usd"]
            total_tokens += u["total_tokens"]

            cat_match = "✓" if c["category"] == item["expected_category"] else "✗"
            pri_match = "✓" if c["priority"] == item["expected_priority"] else "~"

            print(
                f"         {cat_match} category={c['category']:<10} "
                f"{pri_match} priority={c['priority']:<8} "
                f"sentiment={c['sentiment_score']:+.2f} "
                f"resolve={c['estimated_resolution_hours']}h "
                f"${u['estimated_cost_usd']:.5f} {u['latency_seconds']}s"
            )
            print(f"         ✎ {c['reasoning']}")
            results.append({**item, "result": result})

        except Exception as e:
            print(f"         ✗ FAILED: {e}")
            failures.append({"id": item["id"], "label": item["label"], "error": str(e)})

        time.sleep(0.4)

    # ── Summary ────────────────────────────────────────────────────────────────
    cat_correct = sum(1 for r in results if r["result"]["classification"]["category"] == r["expected_category"])
    pri_correct = sum(1 for r in results if r["result"]["classification"]["priority"] == r["expected_priority"])

    print(f"\n{'═' * 70}")
    print(f"  SUMMARY")
    print(f"{'═' * 70}")
    print(f"  Processed        : {len(results)}/{len(TEST_EMAILS)}")
    print(f"  Failures         : {len(failures)}")
    print(f"  Category correct : {cat_correct}/{len(results)}")
    print(f"  Priority correct : {pri_correct}/{len(results)}")
    print(f"  Total tokens     : {total_tokens:,}")
    print(f"  Estimated cost   : ${total_cost:.4f}")

    if failures:
        print(f"\n  Failed:")
        for f in failures:
            print(f"    [{f['id']:02d}] {f['label']} — {f['error']}")

    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"logs/test_results_{provider.value}_{ts}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "provider": provider.value,
            "summary": {
                "processed": len(results),
                "failures": len(failures),
                "category_accuracy": f"{cat_correct}/{len(results)}",
                "priority_accuracy": f"{pri_correct}/{len(results)}",
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(total_cost, 6),
            },
            "results": [
                {
                    "id": r["id"],
                    "label": r["label"],
                    "expected_category": r["expected_category"],
                    "expected_priority": r["expected_priority"],
                    "got_category": r["result"]["classification"]["category"],
                    "got_priority": r["result"]["classification"]["priority"],
                    "full": r["result"],
                }
                for r in results
            ],
            "failures": failures,
        }, f, indent=2)
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 20-email test suite.")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    args = parser.parse_args()
    run_tests(Provider(args.provider))