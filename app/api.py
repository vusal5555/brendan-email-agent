from fastapi import FastAPI, Depends
from pydantic import BaseModel
from retrieve import retrieve_faqs
from classifier import classify_question
from generate import email_agent
from db import get_db
from sqlalchemy.orm import Session


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

    classification = classify_question(request.question)

    if not classification["has_questions"]:
        return AnswerResponse(
            answer="No answer found",
            confidence=0.0,
        )

    extracted_questions = classification["extracted_questions"]

    all_faqs = []

    for question in extracted_questions:
        faqs = retrieve_faqs(question, request.hotel_code, db)
        all_faqs.extend(faqs)

    if len(all_faqs) == 0:
        return AnswerResponse(
            answer="No answer found",
            confidence=0.0,
        )

    chunks = [item[0] for item in all_faqs]
    distances = [item[1] for item in all_faqs]
    confidence = 1 - min(distances)

    if confidence < 0.4:
        return AnswerResponse(
            answer="I will forward this question to the hotel reception.",
            confidence=confidence,
        )

    answer = email_agent(request.question, extracted_questions, chunks)

    return AnswerResponse(
        answer=answer,
        confidence=confidence,
    )
