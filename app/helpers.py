import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from aws_bedrock import get_bedrock_client


client = get_bedrock_client()

CONFIDENCE_THRESHOLD = 0.20
PAGE_CAP = 50
MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 50

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

JUNK_PATHS = (
    "/booking",
    "/reserv",
    "/login",
    "/careers",
    "/legal",
    "/privacy",
    "/impressum",
    "/cookie",
    "/admin",
    "/cart",
    "/checkout",
    "/search",
)


def parse_sitemap(response) -> list[str] | None:
    if response.status_code != 200:
        return None

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None

    tag = root.tag.split("}")[-1]
    if tag not in ("urlset", "sitemapindex"):
        return None

    return [
        loc.text for loc in root.iter() if loc.tag.split("}")[-1] == "loc" and loc.text
    ]


def cleaner(soup: BeautifulSoup) -> str:

    for elements in soup.find_all(["script", "style", "nav", "footer", "header"]):
        elements.decompose()

    return soup.get_text()


def _pack_segments(segments: list[str], joiner: str, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for segment in segments:
        added_len = len(segment) + (len(joiner) if current else 0)
        if current and current_len + added_len > max_chars:
            chunks.append(joiner.join(current))
            current = [segment]
            current_len = len(segment)
        elif len(segment) > max_chars:
            if current:
                chunks.append(joiner.join(current))
                current = []
                current_len = 0
            chunks.extend(_split_oversized(segment, max_chars))
        else:
            current.append(segment)
            current_len += added_len

    if current:
        chunks.append(joiner.join(current))
    return chunks


def _split_oversized(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) > 1:
        line_chunks = _pack_segments(lines, "\n", max_chars)
        if all(len(chunk) <= max_chars for chunk in line_chunks):
            return line_chunks

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(text)
        if sentence.strip()
    ]
    if len(sentences) > 1:
        return _pack_segments(sentences, " ", max_chars)

    return [text[:max_chars].rstrip(), text[max_chars:].lstrip()]


def chunk_page_content(title: str, content: str) -> list[str]:
    """Split cleaned page text into LLM-sized chunks with page title context."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", content.strip())
        if paragraph.strip()
    ]

    bodies: list[str] = []
    for paragraph in paragraphs:
        bodies.extend(_split_oversized(paragraph))

    title_prefix = f"{title.strip()}\n\n" if title.strip() else ""
    chunks: list[str] = []
    for body in bodies:
        if len(body) < MIN_CHUNK_CHARS:
            continue
        chunks.append(f"{title_prefix}{body}" if title_prefix else body)

    return chunks
