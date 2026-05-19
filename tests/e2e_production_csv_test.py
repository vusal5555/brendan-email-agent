"""
End-to-end stress tests using messy real email bodies from production CSV exports.

Runs the same ``run_direct`` / ``run_http`` pipeline as ``e2e_test.py`` against
fixtures in ``tests/fixtures/production_csv_emails.py`` (regenerate that module
from the CSVs if you refresh samples).

Usage:
    python tests/e2e_production_csv_test.py
    python tests/e2e_production_csv_test.py --http --url http://localhost:8000
    python tests/e2e_production_csv_test.py --discover   # delegates to e2e_test DB helper
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fixtures.production_csv_emails import (  # noqa: E402
    EMAIL_NOQ_SEBEBE_LONG_EN_LEGAL,
    EMAIL_NOQ_SEBEBE_REPLY_CHAIN_DE,
    EMAIL_NOQ_TPALMA_FORWARD_CAUTION,
    EMAIL_Q_GHEEXMA_ROOM_PLACEMENT_EN,
    EMAIL_Q_HFHAER_CONNECTING_PARKING_WARN,
    EMAIL_Q_HFHAER_PARKING_WARN,
    EMAIL_Q_JUMAAAD_PURE_BOOKING_EN,
    EMAIL_Q_LEOBCS_BOOKING_FAQ_DE,
    EMAIL_Q_SCHAMEM_AVAIL_PARKING_DE,
    EMAIL_Q_TIBECH_DOG_FORWARD_HEADERS_DE,
)

from e2e_test import (  # noqa: E402
    SEPARATOR,
    HEADER,
    ExpectedAction,
    TestCase,
    TestCategory,
    TestResult,
    discover_hotels,
    format_result,
    print_summary,
    run_direct,
    run_http,
)

# Expected outcomes are human judgment targets (not Brendan ``selected_questions``).
# Tune ranges if the classifier or retrieval drifts.

TEST_CASES: list[TestCase] = [
    # --- “no questions” CSV (booking-heavy / operational noise) ---
    TestCase(
        name="CSV no-q SEBEBE — DE reply chain + buried rate question (7159 chars)",
        email=EMAIL_NOQ_SEBEBE_REPLY_CHAIN_DE,
        hotel_code="SEBEBE",
        expected_action=ExpectedAction.FORWARD,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=4,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    TestCase(
        name="CSV no-q SEBEBE — EN thread + CAUTION + image placeholders + legal (9163 chars)",
        email=EMAIL_NOQ_SEBEBE_LONG_EN_LEGAL,
        hotel_code="SEBEBE",
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=6,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    TestCase(
        name="CSV no-q TPALMA — longest forward chain + CAUTION headers (11090 chars)",
        email=EMAIL_NOQ_TPALMA_FORWARD_CAUTION,
        hotel_code="TPALMA",
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=5,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    # --- “question” CSV (FAQ signal buried in noise) ---
    # Booking-intent (CSV intent=booking); not routed in production — documents classifier behavior.
    TestCase(
        name="CSV q HFHAER — Warnung header + parking PKW (Empire Riverside) [booking]",
        email=EMAIL_Q_HFHAER_PARKING_WARN,
        hotel_code="HFHAER",
        expected_action=ExpectedAction.NO_QUESTIONS,
        expected_language="de",
        expected_has_questions=False,
        expected_question_count_min=0,
        expected_question_count_max=0,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    TestCase(
        name="CSV q HFHAER — Warnung + connecting rooms + parking",
        email=EMAIL_Q_HFHAER_CONNECTING_PARKING_WARN,
        hotel_code="HFHAER",
        expected_action=ExpectedAction.ANSWER,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=2,
        expected_question_count_max=4,
        category=TestCategory.REAL_WORLD_CSV,
        expected_topics=["park"],
    ),
    TestCase(
        name="CSV q LEOBCS — email_name prefix + availability + breakfast + parking",
        email=EMAIL_Q_LEOBCS_BOOKING_FAQ_DE,
        hotel_code="LEOBCS",
        expected_action=ExpectedAction.ANSWER,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=2,
        expected_question_count_max=5,
        category=TestCategory.REAL_WORLD_CSV,
        expected_topics=["park"],
    ),
    TestCase(
        name="CSV q TIBECH — Von/Gesendet smashed line + dog + spa booking",
        email=EMAIL_Q_TIBECH_DOG_FORWARD_HEADERS_DE,
        hotel_code="TIBECH",
        expected_action=ExpectedAction.ANSWER,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=2,
        expected_question_count_max=4,
        category=TestCategory.REAL_WORLD_CSV,
        expected_topics=["hund"],
    ),
    # Booking-intent (CSV intent=booking); not routed in production — documents classifier behavior.
    TestCase(
        name="CSV q SCHAMEM — short DE availability check [booking]",
        email=EMAIL_Q_SCHAMEM_AVAIL_PARKING_DE,
        hotel_code="SCHAMEM",
        expected_action=ExpectedAction.FORWARD,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    # Booking-intent (CSV intent=booking); not routed in production — documents classifier behavior.
    TestCase(
        name="CSV q JUMAAAD — pure EN pricing request [booking]",
        email=EMAIL_Q_JUMAAAD_PURE_BOOKING_EN,
        hotel_code="JUMAAAD",
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.REAL_WORLD_CSV,
    ),
    TestCase(
        name="CSV q GHEEXMA — EN rooms-near-each-other + Sophos banner duplication",
        email=EMAIL_Q_GHEEXMA_ROOM_PLACEMENT_EN,
        hotel_code="GHEEXMA",
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=3,
        category=TestCategory.REAL_WORLD_CSV,
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E stress tests from production CSV email exports"
    )
    parser.add_argument(
        "--http", action="store_true", help="Run tests against HTTP endpoint"
    )
    parser.add_argument(
        "--url", default="http://localhost:8000", help="Base URL for HTTP mode"
    )
    parser.add_argument(
        "--discover", action="store_true", help="Discover hotel codes from DB"
    )
    parser.add_argument(
        "--category", type=str, help="Run only tests matching this category"
    )
    parser.add_argument(
        "--name", type=str, help="Run only tests matching this name substring"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full response text"
    )
    parser.add_argument("--json-output", type=str, help="Write results to JSON file")
    args = parser.parse_args()

    if args.discover:
        discover_hotels()
        return

    cases = TEST_CASES
    if args.category:
        cases = [
            tc for tc in cases if args.category.lower() in tc.category.value.lower()
        ]
    if args.name:
        cases = [tc for tc in cases if args.name.lower() in tc.name.lower()]

    if not cases:
        print("No test cases match the filters.")
        return

    mode = "HTTP" if args.http else "DIRECT"
    print(f"\nRunning {len(cases)} production CSV test cases in {mode} mode\n")
    print(SEPARATOR)
    print(HEADER)
    print(SEPARATOR)

    results: list[TestResult] = []

    for idx, tc in enumerate(cases, 1):
        try:
            if args.http:
                result = run_http(tc, args.url)
            else:
                result = run_direct(tc)
        except Exception as e:
            result = TestResult(
                test_case=tc,
                passed=False,
                failures=[f"EXCEPTION: {type(e).__name__}: {e}"],
                duration_ms=0,
            )

        results.append(result)
        print(format_result(idx, result))

        if args.verbose and result.response_text:
            print(f"    >>> {result.response_text}")

    print_summary(results)

    if args.json_output:
        json_results = []
        for r in results:
            json_results.append(
                {
                    "name": r.test_case.name,
                    "category": r.test_case.category.value,
                    "passed": r.passed,
                    "failures": r.failures,
                    "actual_language": r.actual_language,
                    "actual_question_count": r.actual_question_count,
                    "actual_action": r.actual_action,
                    "actual_confidence": r.actual_confidence,
                    "response_text": r.response_text,
                    "duration_ms": r.duration_ms,
                }
            )
        with open(args.json_output, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults written to {args.json_output}")

    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
