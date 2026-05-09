from models import Base
from sqlalchemy import create_engine

engine = create_engine("postgresql://hotel-db:hotel-db@localhost:5432/hotel-db")
Base.metadata.create_all(engine)
