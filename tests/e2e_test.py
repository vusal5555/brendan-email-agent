"""
End-to-end test harness for the hotel FAQ pipeline.

Runs representative inputs against the live pipeline and produces a pass/fail table.
Supports two modes:
  - Direct: imports pipeline functions (default, faster, shows intermediate state)
  - HTTP: sends requests to a running /answer endpoint (full stack integration)

Usage:
    python tests/e2e_test.py              # direct mode
    python tests/e2e_test.py --http       # against running server
    python tests/e2e_test.py --discover   # discover hotel codes & FAQ counts from DB
"""

import sys
import os
import time
import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ExpectedAction(Enum):
    ANSWER = "answer"
    FORWARD = "forward"
    NO_QUESTIONS = "no_questions"


class TestCategory(Enum):
    LANGUAGE = "language"
    COVERAGE_RICH = "coverage_rich"
    COVERAGE_MEDIUM = "coverage_medium"
    COVERAGE_THIN = "coverage_thin"
    COVERAGE_SPARSE = "coverage_sparse"
    CLASSIFIER_BOOKING_QUESTION = "classifier_booking+question"
    CLASSIFIER_INDIRECT = "classifier_indirect"
    CLASSIFIER_BURIED = "classifier_buried"
    CLASSIFIER_MIXED_LANG = "classifier_mixed_language"
    CLASSIFIER_THANKS = "classifier_thanks"
    CLASSIFIER_COMPLAINT = "classifier_complaint"
    MULTI_QUESTION = "multi_question"
    FORWARD = "forward_to_reception"
    REAL_WORLD_CSV = "real_world_csv"


@dataclass
class TestCase:
    name: str
    email: str
    hotel_code: str
    expected_action: ExpectedAction
    expected_language: str
    expected_has_questions: bool
    category: TestCategory
    expected_question_count_min: int = 0
    expected_question_count_max: int = 10
    expected_topics: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    test_case: TestCase
    passed: bool
    failures: list[str] = field(default_factory=list)
    # Intermediate pipeline state
    actual_has_questions: bool | None = None
    actual_language: str | None = None
    actual_question_count: int | None = None
    actual_action: str | None = None
    actual_confidence: float | None = None
    response_text: str | None = None
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Test cases (20+)
# ---------------------------------------------------------------------------

# Verified hotel codes from the database (run --discover to refresh):
#   EURMAR:  455 FAQs (EN) - RICH
#   EURETH:   97 FAQs (EN) - MEDIUM
#   AT10001:  15 FAQs (EN) - THIN
#   AMBERRA:   4 FAQs (EN) - SPARSE
#
# Non-English FAQs exist for many languages; retrieve_faqs uses the detected
# language. Language tests verify detection plus answer vs forward when chunks
# exist or confidence / model behavior defers to reception.

RICH_HOTEL = "EURMAR"  # 455 FAQs (EN)
MEDIUM_HOTEL = "EURETH"  # 97 FAQs (EN)
THIN_HOTEL = "AT10001"  # 15 FAQs (EN)
SPARSE_HOTEL = "AMBERRA"  # 4 FAQs (EN)

TEST_CASES: list[TestCase] = [
    # =========================================================================
    # LANGUAGE COVERAGE (7 tests)
    # With localized FAQs, non-EN often answers when retrieval matches; some cases still
    # forward when detail is missing or ``response_is_canned_forward`` matches a desk handoff.
    # =========================================================================
    TestCase(
        name="EN - parking question",
        email="Hi, do you have parking available at the hotel? What is the daily rate per car?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.LANGUAGE,
        expected_topics=["parking"],
    ),
    TestCase(
        name="DE - German breakfast question (DE FAQs)",
        email="Guten Tag, wann wird das Frühstück serviert und was kostet es?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.LANGUAGE,
        expected_topics=["frühstück", "breakfast"],
    ),
    TestCase(
        name="ES - Spanish pool question (partial FAQ + forward on schedule)",
        email="Hola, ¿tiene el hotel piscina? ¿Cuál es el horario?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="es",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.LANGUAGE,
    ),
    TestCase(
        name="FR - French check-in question (FR FAQs)",
        email="Bonjour, à quelle heure est le check-in? Merci d'avance.",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="fr",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.LANGUAGE,
        expected_topics=["check-in", "check in", "15h", "15:00", "15h00"],
    ),
    TestCase(
        name="IT - Italian Wi-Fi question (sparse Wi-Fi FAQ coverage)",
        email="Salve, c'è il Wi-Fi gratuito nelle camere? Grazie.",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="it",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.LANGUAGE,
        expected_topics=["wifi", "wi-fi", "internet", "reception", "recezione"],
    ),
    # 6 test
    TestCase(
        name="PT - Portuguese pet question (PT FAQs)",
        email="Olá, aceitam animais de estimação no hotel? Temos um cão pequeno.",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="pt",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.LANGUAGE,
        expected_topics=["animais", "estimação", "permitid", "não"],
    ),
    TestCase(
        name="CA - Catalan airport transfer question (CA FAQs)",
        email="Bon dia, oferiu servei de transfer des de l'aeroport? Gràcies.",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="ca",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.LANGUAGE,
        expected_topics=["transfer", "trasllat", "aeroport", "36"],
    ),
    # =========================================================================
    # HOTEL COVERAGE TIERS (4 tests)
    # English questions across hotels with different EN FAQ depth.
    # Tests that richer hotels answer while sparser ones forward.
    # =========================================================================
    TestCase(
        name="Rich hotel - breakfast question (EURMAR, 455 FAQs)",
        email="Hello, what does breakfast include and when is it served?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.COVERAGE_RICH,
        expected_topics=["breakfast"],
    ),
    TestCase(
        name="Medium hotel - spa question (EURETH, 97 FAQs)",
        email="Hi, are children allowed at the spa? What age is the minimum?",
        hotel_code=MEDIUM_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.COVERAGE_MEDIUM,
        expected_topics=["spa", "children"],
    ),
    TestCase(
        name="Thin hotel - parking question (AT10001, 15 FAQs)",
        email="Hi, where can I park my car at the hotel?",
        hotel_code=THIN_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.COVERAGE_THIN,
        expected_topics=["park"],
    ),
    TestCase(
        name="Sparse hotel - obscure question (AMBERRA, 4 FAQs)",
        email="Do you have a concierge service that can book restaurant reservations for us?",
        hotel_code=SPARSE_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.COVERAGE_SPARSE,
    ),
    # =========================================================================
    # CLASSIFIER ADVERSARIAL SCENARIOS (6 tests)
    # =========================================================================
    # 12 test
    TestCase(
        name="Booking + question mix",
        email="Hi, I'd like to book a double room for July 14-17 for two adults. Also, do you offer private transport or transfers from the airport?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.CLASSIFIER_BOOKING_QUESTION,
        expected_topics=["transport", "transfer"],
    ),
    # 13 test
    TestCase(
        name="Indirect statement implying questions",
        email="Hey, was wondering about the parking situation. Also not sure if you have room service.",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=2,
        category=TestCategory.CLASSIFIER_INDIRECT,
        expected_topics=["parking", "room service"],
    ),
    TestCase(
        name="Buried question in long email",
        email=(
            "Hello team, hope you're doing well. We're a family of four flying in from "
            "Manchester on the 22nd, arriving around 4pm at the airport. We've been planning "
            "this trip for almost a year and the kids are very excited. We booked the suite "
            "with the sea view based on the photos on your website — it looks stunning. "
            "We're also planning a day trip to the old town and one to the vineyards nearby. "
            "Quick thing though, is early check-in possible?"
        ),
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.CLASSIFIER_BURIED,
        expected_topics=["check-in", "early"],
    ),
    TestCase(
        name="Mixed-language email (DE dominant with EN question)",
        email="Hallo, wir kommen am Freitag an. Quick question — do you have a spa on site? Vielen Dank!",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="de",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.CLASSIFIER_MIXED_LANG,
    ),
    TestCase(
        name="Pure thanks - no question",
        email=(
            "Hi team, just wanted to drop a quick note to say thank you for being so "
            "responsive over the past few weeks. We've been planning this anniversary trip "
            "for months and it means a lot that you've been so accommodating with all our "
            "requests. Really looking forward to finally arriving on Saturday. See you soon!"
        ),
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.NO_QUESTIONS,
        expected_language="en",
        expected_has_questions=False,
        expected_question_count_min=0,
        expected_question_count_max=0,
        category=TestCategory.CLASSIFIER_THANKS,
    ),
    TestCase(
        name="Complaint with no question",
        email=(
            "The room was filthy when we checked in yesterday and the front desk staff was "
            "rude when we brought it up. This is not the experience we paid for."
        ),
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.NO_QUESTIONS,
        expected_language="en",
        expected_has_questions=False,
        expected_question_count_min=0,
        expected_question_count_max=0,
        category=TestCategory.CLASSIFIER_COMPLAINT,
    ),
    # =========================================================================
    # MULTI-QUESTION / Brendan case (2 tests)
    # Tests that pipeline handles multiple extracted questions and retrieves
    # context for each independently.
    # =========================================================================
    TestCase(
        name="Multi-question: 2 topics (breakfast + transfers)",
        email=(
            "Hello, two quick questions. First, when is breakfast served in the "
            "morning? And second, do you offer private transport to the airport?"
        ),
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=2,
        expected_question_count_max=2,
        category=TestCategory.MULTI_QUESTION,
        expected_topics=["breakfast", "transport"],
    ),
    TestCase(
        name="Multi-question: 3 topics (parking + pets + checkout)",
        email=(
            "Hi there! We're arriving next week and have a few questions:\n"
            "1. Do you have parking available?\n"
            "2. Do you accept pets? We have a small dog.\n"
            "3. What are the check-out times?\n"
            "Thanks so much!"
        ),
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=3,
        expected_question_count_max=3,
        category=TestCategory.MULTI_QUESTION,
        expected_topics=["parking", "pet", "check-out"],
    ),
    # =========================================================================
    # FORWARD-TO-RECEPTION CASES (3 tests)
    # Questions with no matching FAQ — system should forward, not hallucinate.
    # =========================================================================
    TestCase(
        name="Forward: helicopter tour (no FAQ match)",
        email="Hi, can you arrange a private helicopter tour of the coastline for our anniversary?",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.FORWARD,
    ),
    TestCase(
        name="Forward: FR on sparse hotel (no FR FAQs + sparse)",
        email="Bonjour, est-ce que vous avez un service de blanchisserie?",
        hotel_code=SPARSE_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="fr",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.FORWARD,
    ),
    TestCase(
        name="Forward: drone flying area (no FAQ match)",
        email="Do you have a dedicated area for drone flying near the property?",
        hotel_code=MEDIUM_HOTEL,
        expected_action=ExpectedAction.FORWARD,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.FORWARD,
    ),
    # =========================================================================
    # EDGE CASES (2 tests)
    # =========================================================================
    TestCase(
        name="Edge: empty email body",
        email="",
        hotel_code=RICH_HOTEL,
        expected_action=ExpectedAction.NO_QUESTIONS,
        expected_language="en",
        expected_has_questions=False,
        expected_question_count_min=0,
        expected_question_count_max=0,
        category=TestCategory.CLASSIFIER_THANKS,
    ),
    TestCase(
        name="Edge: sparse hotel known FAQ (dogs allowed)",
        email="Hi, are dogs allowed at your hotel?",
        hotel_code=SPARSE_HOTEL,
        expected_action=ExpectedAction.ANSWER,
        expected_language="en",
        expected_has_questions=True,
        expected_question_count_min=1,
        expected_question_count_max=1,
        category=TestCategory.COVERAGE_SPARSE,
        expected_topics=["dog"],
    ),
]


# ---------------------------------------------------------------------------
# Pipeline runner (direct mode)
# ---------------------------------------------------------------------------


def _collapse_ws(s: str) -> str:
    """Strip ends and collapse internal whitespace (exact forward strings stay case-sensitive)."""

    return " ".join(s.strip().split())


def _fold_for_match(s: str) -> str:
    """Lowercase and strip combining marks so ``réception`` / ``recepción`` match shared substrings."""

    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn").lower()


# Free-form paraphrases of ``language_forward``: need BOTH “cannot answer from here” language
# AND routing to reception/front desk (supported locales: en, de, es, fr, it, pt, ca).
_CANNED_FORWARD_UNAVAILABLE_HINTS: tuple[str, ...] = (
    # English
    "don't have",
    "do not have",
    "don't know",
    "do not know",
    "unable to",
    "not able to",
    "cannot answer",
    "can't answer",
    "cannot provide",
    "can't provide",
    "cannot find",
    "can't find",
    "cannot help",
    "can't help",
    "no information",
    "not covered",
    "outside my knowledge",
    "beyond my knowledge",
    "couldn't find",
    "could not find",
    "not mentioned",
    "not specified",
    "i'm not able",
    "i am not able",
    "no details",
    "insufficient information",
    "not included in",
    "not available in",
    "doesn't cover",
    "does not cover",
    # German — short substrings like "leider keine" / "nicht enthalten" match factual
    # negatives in real FAQ answers (e.g. "leider keine zusätzlichen Kosten", "im Preis nicht enthalten").
    # Keep assistant-deferral / inability phrasing only.
    "habe keine",
    "keine information",
    "keine auskunft",
    "nicht beantworten",
    "kann ich nicht",
    # Spanish
    "no tengo",
    "no dispongo",
    "no puedo dar",
    "no puedo proporcionar",
    "no puedo responder",
    "sin información",
    "sin informacion",
    "no encuentro",
    "no figura",
    "no consta",
    # French
    "je n'ai pas",
    "je nai pas",
    "nous n'avons pas",
    "nous navons pas",
    "je ne peux pas",
    "pas d'information",
    "pas dinformation",
    "aucune information",
    "impossible de vous renseigner",
    "je ne dispose pas",
    # Italian
    "non ho",
    "non posso",
    "non sono in grado",
    "non disponibile",
    "non risulta",
    # Portuguese
    "não tenho",
    "nao tenho",
    "não posso",
    "nao posso",
    "não consigo",
    "nao consigo",
    "sem informação",
    "sem informacao",
    # Catalan
    "no tinc",
    "no puc",
    "sense informació",
    "sense informacio",
    "no disposem",
)
_CANNED_FORWARD_RECEPTION_HINTS: tuple[str, ...] = (
    "reception",
    "recepcion",
    "recepcio",
    "rececao",
    "rezeption",
    "ricevimento",
    "front desk",
    "front office",
    "receptionist",
    "recepcionista",
    "receptionniste",
    "hotel reception",
    "desk reception",
    "l'accueil",
    "l accueil",
    "à l'accueil",
    "a l'accueil",
)


def _folded_matches_unavailable_hint(folded: str, hint: str) -> bool:
    """Substring match for deferral hints, with escapes for common false positives."""

    # ``keine information`` is a substring of factual ``keine informationen ...`` in German FAQs.
    if hint == "keine information":
        return re.search(r"keine information(?!en)", folded) is not None
    # Avoid matching "ich habe keine zusätzlichen ..." (factual negation) — require a deferral object.
    if hint == "habe keine":
        return re.search(
            r"\bhabe keine\s+(?:informationen|information\b|auskunft|daten|antwort)\b",
            folded,
        ) is not None
    return hint in folded


def response_is_canned_forward(text: str) -> bool:
    """Detect API canned forwards and close paraphrases used when the model defers to the desk.

    Requires both (1) language that the requested detail is missing or out of scope and
    (2) directing the guest to reception or front desk, so answers that only mention reception
    as optional contact do not qualify.
    """

    if not (text or "").strip():
        return False

    norm = _collapse_ws(text)
    folded = _fold_for_match(norm)
    lacks_info = any(
        _folded_matches_unavailable_hint(folded, h)
        for h in _CANNED_FORWARD_UNAVAILABLE_HINTS
    )
    routes_desk = any(h in folded for h in _CANNED_FORWARD_RECEPTION_HINTS)
    return lacks_info and routes_desk


def validate_matched_faqs(answer_items: list[dict]) -> list[str]:
    failures = []
    for item_idx, item in enumerate(answer_items):
        matched = item.get("matched_faqs")
        if matched is None:
            failures.append(f"answers[{item_idx}] missing matched_faqs")
            continue
        if not isinstance(matched, list):
            failures.append(f"answers[{item_idx}].matched_faqs must be a list")
            continue

        distances: list[float] = []
        for faq_idx, faq in enumerate(matched):
            for field in ("id", "distance", "rank"):
                if field not in faq:
                    failures.append(
                        f"answers[{item_idx}].matched_faqs[{faq_idx}] missing {field}"
                    )
            if faq.get("rank") != faq_idx + 1:
                failures.append(
                    f"answers[{item_idx}].matched_faqs[{faq_idx}] rank must be {faq_idx + 1}"
                )
            distances.append(float(faq["distance"]))

        if distances and distances != sorted(distances):
            failures.append(
                f"answers[{item_idx}].matched_faqs must be ordered by ascending distance"
            )

    return failures


def action_from_answer_items(answer_items: list[dict]) -> tuple[str, float, str]:
    if not answer_items:
        return "forward", 0.0, ""

    confidence = max(item["confidence"] for item in answer_items)
    non_empty_answers = [item["answer"] for item in answer_items if item["answer"]]

    if not non_empty_answers:
        return "forward", confidence, ""

    response_text = " ".join(non_empty_answers)
    if response_is_canned_forward(response_text):
        return "forward", confidence, response_text

    return "answer", confidence, response_text


def run_direct(test_case: TestCase) -> TestResult:
    from classifier import classify_question
    from retrieve import retrieve_faqs
    from generate import email_agent
    from db import Session

    start = time.time()
    failures = []

    # Handle empty input
    if test_case.email.strip() == "":
        duration = (time.time() - start) * 1000
        actual_action = "no_questions"
        passed = test_case.expected_action == ExpectedAction.NO_QUESTIONS
        if not passed:
            failures.append(
                f"Expected action={test_case.expected_action.value}, got empty input handling"
            )
        return TestResult(
            test_case=test_case,
            passed=passed,
            failures=failures,
            actual_has_questions=False,
            actual_language=None,
            actual_question_count=0,
            actual_action=actual_action,
            actual_confidence=0.0,
            response_text="No question provided",
            duration_ms=duration,
        )

    # Step 1: Classify
    classification = classify_question(test_case.email)
    has_questions = classification.get("has_questions", False)
    extracted_questions = classification.get("extracted_questions", [])
    language = classification.get("language", "unknown")

    # Step 2: Evaluate classification
    if has_questions != test_case.expected_has_questions:
        failures.append(
            f"has_questions: expected={test_case.expected_has_questions}, got={has_questions}"
        )

    if language != test_case.expected_language:
        failures.append(
            f"language: expected={test_case.expected_language}, got={language}"
        )

    q_count = len(extracted_questions)
    if not (
        test_case.expected_question_count_min
        <= q_count
        <= test_case.expected_question_count_max
    ):
        failures.append(
            f"question_count: expected [{test_case.expected_question_count_min}-{test_case.expected_question_count_max}], got={q_count}"
        )

    # Step 3: If no questions, we're done
    if not has_questions:
        duration = (time.time() - start) * 1000
        actual_action = "no_questions"
        if test_case.expected_action != ExpectedAction.NO_QUESTIONS:
            failures.append(
                f"action: expected={test_case.expected_action.value}, got=no_questions"
            )
        return TestResult(
            test_case=test_case,
            passed=len(failures) == 0,
            failures=failures,
            actual_has_questions=has_questions,
            actual_language=language,
            actual_question_count=q_count,
            actual_action=actual_action,
            actual_confidence=0.0,
            response_text="No answer found",
            duration_ms=duration,
        )

    # Step 4: Retrieve and generate per extracted question (matches /answer)
    from api import CONFIDENCE_THRESHOLD, build_matched_faqs

    session = Session()
    try:
        answer_items = []
        for question in extracted_questions:
            faqs = retrieve_faqs(
                question, test_case.hotel_code, session, language=language
            )
            if not faqs:
                answer_items.append(
                    {
                        "question": question,
                        "answer": "",
                        "confidence": 0.0,
                        "matched_faqs": [],
                    }
                )
                continue

            matched_faqs = build_matched_faqs(faqs)
            distances = [item[1] for item in faqs]
            confidence = 1 - min(distances)
            if confidence >= CONFIDENCE_THRESHOLD:
                chunks = [item[0] for item in faqs]
                answer_text = email_agent(question, chunks)
            else:
                answer_text = ""

            answer_items.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "confidence": confidence,
                    "matched_faqs": [faq.model_dump() for faq in matched_faqs],
                }
            )
    finally:
        session.close()

    failures.extend(validate_matched_faqs(answer_items))

    effective_action, confidence, response_text = action_from_answer_items(
        answer_items
    )
    duration = (time.time() - start) * 1000

    if (
        test_case.expected_action == ExpectedAction.ANSWER
        and effective_action != "answer"
    ):
        failures.append(
            f"action: expected=answer, got={effective_action} (confidence={confidence:.3f})"
        )
    elif (
        test_case.expected_action == ExpectedAction.FORWARD
        and effective_action != "forward"
    ):
        failures.append(
            f"action: expected=forward, got={effective_action} (confidence={confidence:.3f})"
        )

    # Step 5: Check topics if answering
    if effective_action == "answer" and test_case.expected_topics:
        response_lower = response_text.lower()
        missing_topics = [
            t for t in test_case.expected_topics if t.lower() not in response_lower
        ]
        if missing_topics and len(missing_topics) == len(test_case.expected_topics):
            failures.append(
                f"topics: none of {test_case.expected_topics} found in response"
            )

    return TestResult(
        test_case=test_case,
        passed=len(failures) == 0,
        failures=failures,
        actual_has_questions=has_questions,
        actual_language=language,
        actual_question_count=q_count,
        actual_action=effective_action,
        actual_confidence=confidence,
        response_text=response_text[:200] if response_text else None,
        duration_ms=duration,
    )


# ---------------------------------------------------------------------------
# Pipeline runner (HTTP mode)
# ---------------------------------------------------------------------------


def run_http(
    test_case: TestCase, base_url: str = "http://localhost:8000"
) -> TestResult:
    import requests

    start = time.time()
    failures = []

    try:
        resp = requests.post(
            f"{base_url}/answer",
            json={"hotel_code": test_case.hotel_code, "question": test_case.email},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        duration = (time.time() - start) * 1000
        return TestResult(
            test_case=test_case,
            passed=False,
            failures=[f"HTTP error: {e}"],
            duration_ms=duration,
        )

    duration = (time.time() - start) * 1000
    answers_list = data.get("answers", [])

    if not answers_list:
        actual_action = "no_questions"
        confidence = 0.0
        answer = ""
    else:
        answer_items = [
            {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "confidence": item.get("confidence", 0.0),
                "matched_faqs": item.get("matched_faqs", []),
            }
            for item in answers_list
        ]
        failures.extend(validate_matched_faqs(answer_items))
        actual_action, confidence, answer = action_from_answer_items(answer_items)

    # Evaluate action
    if test_case.expected_action == ExpectedAction.ANSWER and actual_action != "answer":
        failures.append(f"action: expected=answer, got={actual_action}")
    elif (
        test_case.expected_action == ExpectedAction.FORWARD
        and actual_action != "forward"
    ):
        failures.append(f"action: expected=forward, got={actual_action}")
    elif (
        test_case.expected_action == ExpectedAction.NO_QUESTIONS
        and actual_action != "no_questions"
    ):
        failures.append(f"action: expected=no_questions, got={actual_action}")

    # Check topics in HTTP mode
    if actual_action == "answer" and test_case.expected_topics:
        response_lower = answer.lower()
        missing_topics = [
            t for t in test_case.expected_topics if t.lower() not in response_lower
        ]
        if missing_topics and len(missing_topics) == len(test_case.expected_topics):
            failures.append(
                f"topics: none of {test_case.expected_topics} found in response"
            )

    return TestResult(
        test_case=test_case,
        passed=len(failures) == 0,
        failures=failures,
        actual_action=actual_action,
        actual_confidence=confidence,
        response_text=answer[:200] if answer else None,
        duration_ms=duration,
    )


# ---------------------------------------------------------------------------
# Results table formatter
# ---------------------------------------------------------------------------

HEADER = (
    f"{'#':<3} {'Status':<6} {'Category':<26} {'Name':<45} "
    f"{'Lang':<6} {'Qs':<4} {'Action':<10} {'Conf':<6} {'Time':<8} {'Failures'}"
)
SEPARATOR = "-" * 160


def format_result(idx: int, r: TestResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    lang = f"{r.actual_language or '?'}" + (
        ""
        if r.actual_language == r.test_case.expected_language
        else f"!={r.test_case.expected_language}"
    )
    qs = str(r.actual_question_count) if r.actual_question_count is not None else "?"
    action = r.actual_action or "?"
    conf = f"{r.actual_confidence:.2f}" if r.actual_confidence is not None else "?"
    time_str = f"{r.duration_ms:.0f}ms" if r.duration_ms is not None else "?"
    fail_str = "; ".join(r.failures) if r.failures else ""

    color_start = "\033[92m" if r.passed else "\033[91m"
    color_end = "\033[0m"

    return (
        f"{color_start}{idx:<3} {status:<6}{color_end} "
        f"{r.test_case.category.value:<26} {r.test_case.name:<45} "
        f"{lang:<6} {qs:<4} {action:<10} {conf:<6} {time_str:<8} {fail_str}"
    )


def print_summary(results: list[TestResult]):
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)
    total_time = sum(r.duration_ms for r in results if r.duration_ms)

    print(f"\n{'=' * 160}")
    print(
        f"SUMMARY: {passed}/{total} passed, {failed} failed | Total time: {total_time:.0f}ms"
    )
    print(f"{'=' * 160}")

    # Group failures by category
    if failed > 0:
        print("\nFAILURES BY CATEGORY:")
        by_cat: dict[str, list[TestResult]] = {}
        for r in results:
            if not r.passed:
                cat = r.test_case.category.value
                by_cat.setdefault(cat, []).append(r)
        for cat, fails in sorted(by_cat.items()):
            print(f"\n  [{cat}]")
            for r in fails:
                print(f"    - {r.test_case.name}: {'; '.join(r.failures)}")
                if r.response_text:
                    print(f"      Response: {r.response_text[:120]}...")


# ---------------------------------------------------------------------------
# Hotel discovery helper
# ---------------------------------------------------------------------------


def discover_hotels():
    """Query the database to show hotel codes and their FAQ counts per language."""
    from db import Session
    from models import FaqChunk
    from sqlalchemy import func

    session = Session()
    try:
        results = (
            session.query(
                FaqChunk.hotel_code,
                FaqChunk.language,
                func.count(FaqChunk.id).label("faq_count"),
            )
            .group_by(FaqChunk.hotel_code, FaqChunk.language)
            .order_by(func.count(FaqChunk.id).desc())
            .all()
        )

        print(f"\n{'Hotel Code':<15} {'Language':<10} {'FAQ Count':<10} {'Tier'}")
        print("-" * 50)
        for hotel_code, language, count in results:
            if count >= 100:
                tier = "RICH (100+)"
            elif count >= 20:
                tier = "MEDIUM (20-99)"
            elif count >= 5:
                tier = "THIN (5-19)"
            else:
                tier = "SPARSE (<5)"
            print(f"{hotel_code:<15} {language:<10} {count:<10} {tier}")

        # Also show totals per hotel
        print(f"\n\n{'Hotel Code':<15} {'Total FAQs':<12} {'Tier'}")
        print("-" * 40)
        totals = (
            session.query(
                FaqChunk.hotel_code,
                func.count(FaqChunk.id).label("total"),
            )
            .group_by(FaqChunk.hotel_code)
            .order_by(func.count(FaqChunk.id).desc())
            .all()
        )
        for hotel_code, total in totals:
            if total >= 100:
                tier = "RICH"
            elif total >= 20:
                tier = "MEDIUM"
            elif total >= 5:
                tier = "THIN"
            else:
                tier = "SPARSE"
            print(f"{hotel_code:<15} {total:<12} {tier}")

    finally:
        session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="E2E test harness for hotel FAQ pipeline"
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

    # Filter test cases
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

    print(
        f"\nRunning {len(cases)} test cases in {'HTTP' if args.http else 'DIRECT'} mode\n"
    )
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

    # JSON output
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

    # Exit with non-zero if any failures
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
