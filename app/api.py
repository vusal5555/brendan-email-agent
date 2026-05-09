from fastapi import FastAPI
from pydantic import BaseModel
from retrieve import retrieve_faqs
from classifier import classify_question
from generate import email_agent


class AnswerRequest(BaseModel):
    hotel_code: str
    question: str


class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    source_chunks: list[str]


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/answer")
def answer(request: AnswerRequest):

    classification = classify_question(request.question)

    if not classification["has_questions"]:
        return AnswerResponse(
            answer="No answer found",
            confidence=0.0,
            source_chunks=[],
        )

    extracted_questions = classification["extracted_questions"]

    all_faqs = []

    for question in extracted_questions:
        faqs = retrieve_faqs(question, request.hotel_code)
        all_faqs.extend(faqs)

    if len(all_faqs) == 0:
        return AnswerResponse(
            answer="No answer found",
            confidence=0.0,
            source_chunks=[],
        )

    answer = email_agent(request.question, extracted_questions, all_faqs)

    return AnswerResponse(
        answer=answer,
        confidence=0.0,
        source_chunks=[faq.content for faq in all_faqs],
    )
