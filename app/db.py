from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from aws_secrets import get_secrets


load_dotenv()


db_url = get_secrets()

# engine = create_engine(os.getenv("DB_URL"), pool_pre_ping=True)
engine = create_engine(db_url, pool_pre_ping=True)
Session = sessionmaker(bind=engine)


def get_db():

    session = Session()
    try:
        yield session
    finally:
        session.close()
