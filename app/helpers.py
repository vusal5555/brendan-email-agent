import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
from aws_bedrock import get_bedrock_client
from pydantic import BaseModel
from generate import faq_generator
from models import FaqChunk
from db import Session as db_session
import logging
from datetime import datetime, timezone
import json
import requests
import urllib.parse
from embed_faqs import embed_faqs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.20


class WebsiteRequest(BaseModel):
    hotel_code: str
    urls: list[str]
    language: str


class WebsiteResponse(BaseModel):
    title: str
    content: str


client = get_bedrock_client()

CONFIDENCE_THRESHOLD = 0.20
PAGE_CAP = 50
MAX_CHUNK_CHARS = 1000
MIN_CHUNK_CHARS = 150

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


def crawl_website(request: WebsiteRequest):
    try:
        db = db_session()
        start_time = datetime.now(timezone.utc)
        logger.info("Starting crawl for %s", request.urls)

        visited_urls = set()

        for url in request.urls:
            page_contents = []
            response = requests.get(f"{url}/sitemap.xml")
            urls = parse_sitemap(response)

            if urls:
                for url_item in urls:
                    if url_item in visited_urls:
                        continue
                    visited_urls.add(url_item)

                    if len(page_contents) >= PAGE_CAP:
                        break
                    if any(path in url_item for path in JUNK_PATHS):
                        continue

                    response = requests.get(url_item)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        title = soup.title.string

                        if title is None:
                            title = ""
                        content = cleaner(soup)
                        page_contents.append(
                            WebsiteResponse(title=title, content=content)
                        )

            else:
                response = requests.get(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    title = soup.title.string
                    if title is None:
                        title = ""

                    hrefs = [link.get("href") for link in soup.find_all("a")]
                    content = cleaner(soup)
                    page_contents.append(WebsiteResponse(title=title, content=content))

                    visited_urls.add(url)

                    for href in hrefs:
                        if len(page_contents) >= PAGE_CAP:
                            break
                        if href is None:
                            continue
                        if any(path in href for path in JUNK_PATHS):
                            continue

                        absolute_url = urllib.parse.urljoin(url, href)

                        if absolute_url in visited_urls:
                            continue
                        visited_urls.add(absolute_url)

                        if (
                            urllib.parse.urlparse(absolute_url).netloc
                            != urllib.parse.urlparse(url).netloc
                        ):
                            continue
                        response = requests.get(absolute_url)

                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, "html.parser")
                            title = soup.title.string if soup.title else ""
                            content = cleaner(soup)
                            page_contents.append(
                                WebsiteResponse(title=title, content=content)
                            )

            logger.info("Crawled %d pages", len(page_contents))

            db.query(FaqChunk).filter(FaqChunk.hotel_code == request.hotel_code).filter(
                FaqChunk.language == request.language
            ).filter(FaqChunk.source_type == "website").filter(
                FaqChunk.source_origin == url
            ).delete()

            chunks = []
            for page_content in page_contents:
                chunks.extend(
                    chunk_page_content(page_content.title, page_content.content)
                )

            logger.info("Generated %d chunks", len(chunks))

            if len(chunks) > 100:
                chunks = chunks[:100]

            total_faqs = 0
            total_chunks = len(chunks)
            all_faqs = []
            for i, chunk in enumerate(chunks, start=1):
                logger.info("Processing chunk %d/%d", i, total_chunks)
                faqs = faq_generator(chunk, request.language)
                logger.info("Generated %d FAQs from chunk %d", len(faqs), i)
                total_faqs += len(faqs)
                all_faqs.extend(faqs)

            all_embedding_inputs = ["Question: " + faq["question"] for faq in all_faqs]

            all_embeddings = []

            loop_embeddings = range(0, len(all_embedding_inputs), 100)
            for i in loop_embeddings:
                embeddings = embed_faqs(all_embedding_inputs[i : i + 100])
                all_embeddings.extend(embeddings)

            for faq, embedding in zip(all_faqs, all_embeddings):
                db.add(
                    FaqChunk(
                        hotel_code=request.hotel_code,
                        source_origin=url,
                        question=faq["question"],
                        answer=faq["answer"],
                        embedding=embedding,
                        language=request.language,
                        source_type="website",
                        embedding_input=f"Question: {faq['question']}\nAnswer: {faq['answer']}",
                        embedding_model="amazon.titan-embed-text-v2:0",
                    )
                )

            db.commit()
        logger.info("Inserted %d FAQs for %s", total_faqs, request.hotel_code)
    except Exception:
        logger.error(
            json.dumps(
                {
                    "hotel_code": request.hotel_code,
                    "urls": request.urls,
                    "response_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()
                    * 1000,
                }
            ),
            exc_info=True,
        )
    finally:
        db.close()
