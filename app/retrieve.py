from embed_faqs import embed_text
from sqlalchemy import create_engine
from models import Base, FaqChunk
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://hotel-db:hotel-db@localhost:5432/hotel-db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


def retrieve_faqs(query, hotel_code, k=3):

    embedding = embed_text(query)

    faqs = (
        session.query(FaqChunk)
        .filter(FaqChunk.hotel_code == hotel_code)
        .order_by(FaqChunk.embedding.cosine_distance(embedding))
        .limit(k)
        .all()
    )

    return faqs


if __name__ == "__main__":
    faqs = retrieve_faqs("What is the check-in time?", "HTL001")
    for faq in faqs:
        print(faq.question)
        print(faq.answer)
        print("-" * 100)
