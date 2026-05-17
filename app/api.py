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

language_forward = {
    "en": "I will forward this question to the hotel reception.",
    "de": "Ich werde diese Frage an das Hotelrezeption weiterleiten.",
    "es": "Sere reenviaré esta pregunta al recepcionista del hotel.",
    "fr": "Je vais transmettre cette question au réceptionniste de l'hôtel",
    "it": "Rivolgerò questa domanda al ricevitore dell'hotel.",
    "pt": "Vou enviar esta pergunta para o recepcionista do hotel.",
    "ca": "Enviaré aquesta pregunta al recepcionista de l'hotel.",
}


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
        did_forward = False
        result_answer = "No answer found"
        confidence = 0.0

        if request.question.strip() == "":
            result_answer = "No question provided"
        else:
            classification = classify_question(request.question)

            if not classification["has_questions"]:
                result_answer = "No answer found"
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
                    result_answer = "No answer found"
                else:
                    chunks = [item[0] for item in all_faqs]
                    distances = [item[1] for item in all_faqs]
                    top_chunk_distance = min(distances)
                    confidence = 1 - min(distances)

                    if confidence < 0.20:
                        did_forward = True
                        result_answer = language_forward.get(
                            detected_language,
                            "I will forward this question to the hotel reception.",
                        )
                    else:
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
                    "did_forward": did_forward,
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
                    "question": request.question,
                    "response_time": (datetime.now() - start_time).total_seconds()
                    * 1000,
                }
            )
        )
        raise HTTPException(status_code=500, detail="Service temporarily unavailable")
