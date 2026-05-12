from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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
