from pymysql import connect
from dotenv import load_dotenv
import os
import argparse
import markdownify
from pymysql.cursors import DictCursor
from embed_faqs import embed_faqs
from db import engine
from models import FaqChunk
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--languages",
    type=str,
    nargs="+",
    default=["en", "de", "es", "fr", "it", "pt", "ca"],
)
parser.add_argument(
    "--hotel-code",
    type=str,
)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument(
    "--dry-run",
    action="store_true",
)
args = parser.parse_args()

languages = args.languages
hotel_code = args.hotel_code
limit = args.limit
dry_run = args.dry_run


def get_faqs():

    session = Session(engine)
    conn = connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DB"),
        cursorclass=DictCursor,
    )

    cursor = conn.cursor()

    for lang in languages:
        params = []
        values = []
        if hotel_code:
            params.append("AND Hotelcode = %s")
            values.append(hotel_code)
        if limit:
            params.append("LIMIT %s")
            values.append(limit)

        query = f"SELECT ID, Hotelcode, Question, Answer, Timestamp FROM FAQ_{lang} WHERE Deleted = 0 {''.join(params)}"

        embedding_list = []
        cursor.execute(query, values)
        faqs = cursor.fetchall()

        for faq in faqs:
            if faq["Question"] is None or faq["Answer"] is None:
                continue

            hotel_id = faq["ID"]
            question = faq["Question"]
            answer = faq["Answer"]
            timestamp = faq["Timestamp"]

            answer = markdownify.markdownify(answer)

            embedding_input = f"Question: {question}\nAnswer: {answer}"

            embedding_list.append(
                {
                    "id": hotel_id,
                    "hotel_code": faq["Hotelcode"],
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp,
                    "embedding_input": embedding_input,
                }
            )

        embedding_input_list = [item["embedding_input"] for item in embedding_list]

        all_embeddings = []

        loop_embeddings = range(0, len(embedding_input_list), 100)
        for i in loop_embeddings:
            embeddings = embed_faqs(embedding_input_list[i : i + 100])
            all_embeddings.extend(embeddings)

        for faq, embedding in zip(embedding_list, all_embeddings):
            faq["embedding"] = embedding
            faq_chunk = (
                insert(FaqChunk)
                .values(
                    hotel_code=faq["hotel_code"],
                    question=faq["question"],
                    answer=faq["answer"],
                    embedding=embedding,
                    language=lang,
                    source_id=faq["id"],
                    source_updated_at=faq["timestamp"],
                    embedding_input=faq["embedding_input"],
                    embedding_model="text-embedding-3-small",
                )
                .on_conflict_do_update(
                    index_elements=["source_id", "language"],
                    set_={
                        "question": faq["question"],
                        "answer": faq["answer"],
                        "embedding": embedding,
                        "embedding_input": faq["embedding_input"],
                        "embedding_model": "text-embedding-3-small",
                        "source_updated_at": faq["timestamp"],
                    },
                )
            )
            session.execute(faq_chunk)
        session.commit()
    conn.close()


if __name__ == "__main__":
    get_faqs()
