from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from retrieve import retrieve_faqs
from classifier import classify_question
from generate import email_agent
from db import get_db
from sqlalchemy.orm import Session
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    hotel_code: str
    question: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/answer")
def answer(request: AnswerRequest, db: Session = Depends(get_db)):

    try:
        start_time = datetime.now()
        detected_language = None
        extracted_questions = []
        top_chunk_distance = None
        result_answer = ""
        confidence = 0.0

        if request.question.strip() == "":
            result_answer = ""
        else:
            classification = classify_question(request.question)

            if not classification["has_questions"]:
                result_answer = ""
            else:
                extracted_questions = classification["extracted_questions"]
                detected_language = classification["language"]

                all_faqs = []

                for question in extracted_questions:
                    faqs = retrieve_faqs(
                        question, request.hotel_code, db, language=detected_language
                    )
                    all_faqs.extend(faqs)

                if len(all_faqs) == 0:
                    result_answer = ""
                else:
                    chunks = [item[0] for item in all_faqs]
                    distances = [item[1] for item in all_faqs]
                    top_chunk_distance = min(distances)
                    confidence = 1 - min(distances)

                    result_answer = email_agent(
                        request.question, extracted_questions, chunks
                    )

        logger.info(
            json.dumps(
                {
                    "hotel_code": request.hotel_code,
                    "extracted_questions": extracted_questions,
                    "detected_language": detected_language,
                    "top_chunk_distance": top_chunk_distance,
                    "response_time": (datetime.now() - start_time).total_seconds()
                    * 1000,
                }
            )
        )

        return AnswerResponse(
            answer=result_answer,
            confidence=confidence,
        )

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
