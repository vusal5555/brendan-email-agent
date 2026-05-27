import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from api import build_matched_faqs


class _FakeChunk:
    def __init__(
        self,
        chunk_id: int,
        source_id: int | None,
        question: str | None,
        source_type: str | None,
    ):
        self.id = chunk_id
        self.source_id = source_id
        self.question = question
        self.source_type = source_type


def test_build_matched_faqs_ranks_and_preserves_fields():
    faqs = [
        (_FakeChunk(10, 501, "Is parking available?", "mysql_faq"), 0.12),
        (_FakeChunk(11, None, "Where can I park?", "website"), 0.34),
    ]

    matched = build_matched_faqs(faqs)

    assert len(matched) == 2
    assert matched[0].id == 10
    assert matched[0].source_id == 501
    assert matched[0].question == "Is parking available?"
    assert matched[0].source_type == "mysql_faq"
    assert matched[0].distance == 0.12
    assert matched[0].rank == 1
    assert matched[1].rank == 2


def test_build_matched_faqs_returns_empty_list_for_no_results():
    assert build_matched_faqs([]) == []
