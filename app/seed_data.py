from sqlalchemy import create_engine
from models import Base, Hotel, FaqChunk
from sqlalchemy.orm import sessionmaker
import json

engine = create_engine("postgresql://hotel-db:hotel-db@localhost:5432/hotel-db")
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

with open("data/seed.json", "r") as f:
    data = json.load(f)

for hotel in data["hotels"]:
    hotel_data = Hotel(
        hotel_code=hotel["hotel_code"],
        name=hotel["name"],
        default_language=hotel["default_language"],
    )
    session.add(hotel_data)
    session.commit()

    for faq in hotel["faqs"]:
        faq_data = FaqChunk(
            hotel_code=hotel["hotel_code"],
            question=faq["question"],
            answer=faq["answer"],
            content=f"{faq['question']} {faq['answer']}",
            language=hotel["default_language"],
        )
        session.add(faq_data)
    session.commit()
