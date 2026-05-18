from sqlalchemy import Column, String, Integer, Text, DateTime, UniqueConstraint, Index
from pgvector.sqlalchemy import Vector
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class FaqChunk(Base):
    __tablename__ = "faq_chunks"
    id = Column(Integer, primary_key=True)
    hotel_code = Column(String(50), index=True)
    question = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    embedding = Column(Vector(1024))
    language = Column(String(2), nullable=False)
    source_id = Column(Integer)
    embedding_input = Column(Text, nullable=True)
    embedding_model = Column(String(255))
    source_updated_at = Column(DateTime, nullable=True)
    ingested_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("source_id", "language", name="uix_hotel_source_language"),
        Index(
            "idx_hnsw_embedding",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 24, "ef_construction": 200},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
