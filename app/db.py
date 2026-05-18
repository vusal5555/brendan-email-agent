from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DB_URL"), pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def get_db():

    session = Session()
    try:
        yield session
    finally:
        session.close()
