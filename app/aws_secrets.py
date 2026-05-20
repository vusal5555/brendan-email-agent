import boto3
import json
from urllib.parse import quote_plus
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secrets():

    try:
        client = boto3.client("secretsmanager", region_name="eu-central-1")
        response = client.get_secret_value(SecretId="EmbeddingsDatabaseAdminAccess")

        secret = json.loads(response["SecretString"])

        username = quote_plus(secret["username"])
        password = quote_plus(secret["password"])
        host = secret["host"]
        port = secret["port"]
        database = "postgres"

        return f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    except Exception:
        return os.getenv("DB_URL")
