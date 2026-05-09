from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Hotel(Base):
    __tablename__ = "hotels"
    hotel_code = Column(String, primary_key=True)
    name = Column(String)
    default_language = Column(String)


class FaqChunk(Base):
    __tablename__ = "faq_chunks"
    id = Column(Integer, primary_key=True)
    hotel_code = Column(String, ForeignKey("hotels.hotel_code"), index=True)
    question = Column(String)
    answer = Column(String)
    content = Column(String)
    embedding = Column(Vector(384))
    language = Column(String)
