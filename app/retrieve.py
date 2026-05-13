from embed_faqs import embed_text
from models import FaqChunk
from sqlalchemy.orm import Session


def retrieve_faqs(query, hotel_code, db: Session, k=3, language="en"):

    embedding = embed_text(query)

    distance = FaqChunk.embedding.cosine_distance(embedding).label("distance")
    faqs = (
        db.query(FaqChunk, distance)
        .filter(FaqChunk.hotel_code == hotel_code)
        .filter(FaqChunk.language == language)
        .order_by(distance)
        .limit(k)
        .all()
    )

    return faqs
