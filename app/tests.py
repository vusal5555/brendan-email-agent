import requests
import pytest

BASE_URL = "http://localhost:8000"


def post_answer(hotel_code: str, question: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/answer",
        json={"hotel_code": hotel_code, "question": question},
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()


# ── helpers ──────────────────────────────────────────────────────────


def assert_structure(data: dict):
    """Every response must contain these three keys with the right types."""
    assert "answer" in data
    assert "confidence" in data
    assert "source_chunks" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["confidence"], (int, float))
    assert isinstance(data["source_chunks"], list)


# ── 1. Single question – English ─────────────────────────────────────


def test_single_question_english():
    data = post_answer("HTL001", "Hi, what time is check-in?")

    assert_structure(data)
    assert data["answer"] != "No answer found"
    assert len(data["source_chunks"]) > 0
    assert "14:00" in data["answer"], "Expected check-in time (14:00) in the answer"


# ── 2. Single question – German ──────────────────────────────────────


def test_single_question_german():
    data = post_answer(
        "HTL026",
        "Hallo, wann kann ich einchecken?",
    )

    assert_structure(data)
    assert data["answer"] != "No answer found"
    assert len(data["source_chunks"]) > 0

    answer_lower = data["answer"].lower()
    has_german = any(
        w in answer_lower
        for w in ["check-in", "uhr", "einchecken", "anreise", "14:30"]
    )
    assert has_german, f"Expected German-language answer, got: {data['answer']}"


# ── 3. Multi-question email (parking + pets + breakfast) ─────────────


def test_multi_question_email():
    email = (
        "Hello, we're planning to visit next week. "
        "Could you tell us if parking is available at the hotel? "
        "Also, are pets allowed in the rooms? "
        "And what time is breakfast served?"
    )
    data = post_answer("HTL001", email)

    assert_structure(data)
    assert data["answer"] != "No answer found"
    assert len(data["source_chunks"]) >= 3, "Expected chunks for each sub-question"

    answer_lower = data["answer"].lower()
    assert any(w in answer_lower for w in ["parking", "car park"]), \
        "Answer should address parking"
    assert any(w in answer_lower for w in ["pet", "animal", "dog"]), \
        "Answer should address pets"
    assert any(w in answer_lower for w in ["breakfast", "buffet"]), \
        "Answer should address breakfast"


# ── 4. Non-question email ────────────────────────────────────────────


def test_non_question_email():
    data = post_answer(
        "HTL001",
        "Thanks for the confirmation! We look forward to our stay.",
    )

    assert_structure(data)
    assert data["answer"] == "No answer found", \
        "Non-question emails should return early without hitting the LLM"
    assert data["confidence"] == 0.0
    assert data["source_chunks"] == []


# ── 5. Question not covered by FAQ data ──────────────────────────────


def test_question_not_in_faq():
    data = post_answer(
        "HTL001",
        "Do you have a helicopter landing pad on the roof?",
    )

    assert_structure(data)
    # Even if the retriever returns some distant chunks, the answer should
    # either say it's not covered or forward to reception.
    assert len(data["answer"]) > 0


# ── 6. Invalid hotel code ────────────────────────────────────────────


def test_invalid_hotel_code():
    data = post_answer(
        "INVALID_HOTEL_999",
        "What time is check-in?",
    )

    assert_structure(data)
    assert data["answer"] == "No answer found", \
        "Invalid hotel code should yield no matching FAQs"
    assert data["source_chunks"] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
