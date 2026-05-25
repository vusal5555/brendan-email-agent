from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from pydantic import BaseModel
from retrieve import retrieve_faqs
from classifier import classify_question
from generate import email_agent
from db import get_db
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import json
from helpers import crawl_website, ingest_pdf, PdfRequest


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.20
MAX_PDF_BYTES = 10 * 1024 * 1024
PDF_MAGIC = b"%PDF"


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
    urls: list[str]
    language: str


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/website")
async def ingest_website(
    request: WebsiteRequest,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(crawl_website, request)
    return {"message": "Website ingestion started"}


@app.post("/pdf")
async def ingest_pdf_endpoint(
    background_tasks: BackgroundTasks,
    pdf_file: UploadFile = File(...),
    hotel_code: str = Form(...),
    language: str = Form(...),
):
    pdf_bytes = await pdf_file.read()

    if pdf_file.content_type and pdf_file.content_type not in (
        "application/pdf",
        "application/x-pdf",
    ):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF exceeds maximum size of {MAX_PDF_BYTES // (1024 * 1024)}MB",
        )

    if not pdf_bytes.startswith(PDF_MAGIC):
        raise HTTPException(status_code=400, detail="File is not a valid PDF")

    request = PdfRequest(
        hotel_code=hotel_code,
        pdf_file=pdf_bytes,
        language=language,
        filename=pdf_file.filename or "unknown.pdf",
    )
    background_tasks.add_task(ingest_pdf, request)
    return {"message": "PDF ingestion started"}


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
