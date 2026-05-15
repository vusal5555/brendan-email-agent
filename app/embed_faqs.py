from dotenv import load_dotenv
import json
from aws_bedrock import get_bedrock_client

load_dotenv()


client = get_bedrock_client()


def embed_text(text):
    response = client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text, "dimensions": 1024, "normalize": True}),
    )
    embeddings = json.loads(response["body"].read())["embedding"]
    return embeddings


def embed_faqs(text_list):
    embeddings = [embed_text(text) for text in text_list]
    return embeddings
