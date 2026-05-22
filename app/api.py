from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from retrieve import retrieve_faqs
from classifier import classify_question
from generate import email_agent, faq_generator
from models import FaqChunk
from db import get_db
from sqlalchemy.orm import Session
import logging
from datetime import datetime, timezone
import json
import requests
from bs4 import BeautifulSoup
import urllib.parse
from helpers import parse_sitemap, cleaner, chunk_page_content, PAGE_CAP, JUNK_PATHS
from embed_faqs import embed_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.20


class AnswerRequest(BaseModel):
    hotel_code: str
    question: str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    confidence: float


class AnswerResponse(BaseModel):
    answers: list[QuestionAnswer]


class WebsiteRequest(BaseModel):
    hotel_code: str
    url: str
    language: str


class WebsiteResponse(BaseModel):
    title: str
    content: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/website")
def ingest_website(request: WebsiteRequest, db: Session = Depends(get_db)):
    try:
        start_time = datetime.now(timezone.utc)
        logger.info("Starting crawl for %s", request.url)
        response = requests.get(f"{request.url}/sitemap.xml")
        urls = parse_sitemap(response)
        page_contents = []
        visited_urls = set()
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
                    page_contents.append(WebsiteResponse(title=title, content=content))

        else:
            response = requests.get(request.url)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string
                if title is None:
                    title = ""

                hrefs = [link.get("href") for link in soup.find_all("a")]
                content = cleaner(soup)
                page_contents.append(WebsiteResponse(title=title, content=content))

                visited_urls.add(request.url)

                for href in hrefs:
                    if len(page_contents) >= PAGE_CAP:
                        break
                    if href is None:
                        continue
                    if any(path in href for path in JUNK_PATHS):
                        continue

                    absolute_url = urllib.parse.urljoin(request.url, href)

                    if absolute_url in visited_urls:
                        continue
                    visited_urls.add(absolute_url)

                    if (
                        urllib.parse.urlparse(absolute_url).netloc
                        != urllib.parse.urlparse(request.url).netloc
                    ):
                        continue
                    response = requests.get(absolute_url)

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "html.parser")
                        soup.title.string if soup.title else ""
                        content = cleaner(soup)
                        page_contents.append(
                            WebsiteResponse(title=title, content=content)
                        )

        logger.info("Crawled %d pages", len(page_contents))

        chunks = []
        for page_content in page_contents:
            chunks.extend(chunk_page_content(page_content.title, page_content.content))

        logger.info("Generated %d chunks", len(chunks))

        db.query(FaqChunk).filter(FaqChunk.hotel_code == request.hotel_code).filter(
            FaqChunk.language == request.language
        ).filter(FaqChunk.source_type == "website").delete()

        total_faqs = 0
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks, start=1):
            logger.info("Processing chunk %d/%d", i, total_chunks)
            faqs = faq_generator(chunk, request.language)
            logger.info("Generated %d FAQs from chunk %d", len(faqs), i)
            total_faqs += len(faqs)
            for faq in faqs:
                db.add(
                    FaqChunk(
                        hotel_code=request.hotel_code,
                        question=faq["question"],
                        answer=faq["answer"],
                        language=request.language,
                        embedding=embed_text("Question: " + faq["question"]),
                        embedding_input=f"Question: {faq['question']}\nAnswer: {faq['answer']}",
                        embedding_model="amazon.titan-embed-text-v2:0",
                        source_type="website",
                    )
                )
        db.commit()
        logger.info("Inserted %d FAQs for %s", total_faqs, request.hotel_code)
    except Exception:
        logger.error(
            json.dumps(
                {
                    "hotel_code": request.hotel_code,
                    "url": request.url,
                    "response_time": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds()
                    * 1000,
                }
            ),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")


@app.post("/answer")
def answer(request: AnswerRequest, db: Session = Depends(get_db)):

    try:
        start_time = datetime.now()
        detected_language = None
        extracted_questions = []
        answers: list[QuestionAnswer] = []
        question_confidences: list[float] = []

        if request.question.strip() != "":
            classification = classify_question(request.question)

            if classification["has_questions"]:
                extracted_questions = classification["extracted_questions"]
                detected_language = classification["language"]

                for question in extracted_questions:
                    faqs = retrieve_faqs(
                        question, request.hotel_code, db, language=detected_language
                    )

                    if not faqs:
                        question_confidences.append(0.0)
                        answers.append(
                            QuestionAnswer(question=question, answer="", confidence=0.0)
                        )
                        continue

                    distances = [item[1] for item in faqs]
                    question_top_distance = min(distances)

                    confidence = 1 - question_top_distance
                    question_confidences.append(confidence)
                    if confidence >= CONFIDENCE_THRESHOLD:
                        chunks = [item[0] for item in faqs]
                        answer_text = email_agent(question, chunks)
                    else:
                        answer_text = ""

                    answers.append(
                        QuestionAnswer(
                            question=question,
                            answer=answer_text,
                            confidence=confidence,
                        )
                    )

        logger.info(
            json.dumps(
                {
                    "hotel_code": request.hotel_code,
                    "extracted_questions": extracted_questions,
                    "question_confidences": question_confidences,
                    "detected_language": detected_language,
                    "response_time": (datetime.now() - start_time).total_seconds()
                    * 1000,
                }
            )
        )

        return AnswerResponse(answers=answers)

    except Exception:
        logger.error(
            json.dumps(
                {
                    "hotel_code": request.hotel_code,
                    "response_time": (datetime.now() - start_time).total_seconds()
                    * 1000,
                }
            ),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")
