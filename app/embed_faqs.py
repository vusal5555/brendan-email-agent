from sqlalchemy import create_engine
from models import Base, FaqChunk
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

engine = create_engine("postgresql://hotel-db:hotel-db@localhost:5432/hotel-db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


def embed_text(text):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=384
    )
    embeddings = response.data[0].embedding
    return embeddings


def embed_faqs(text):
    response = client.embeddings.create(
        input=text, model="text-embedding-3-small", dimensions=384
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings


if __name__ == "__main__":
    faqs = session.query(FaqChunk).all()

    faqs_content = [faq.content for faq in faqs]

    embeddings = embed_faqs(faqs_content)

    for faq, embedding in zip(faqs, embeddings):
        faq.embedding = embedding

    session.commit()
