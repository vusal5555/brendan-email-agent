import boto3
import os
from dotenv import load_dotenv

load_dotenv()


def get_bedrock_client():

    client = boto3.client(
        service_name="bedrock-runtime",
        region_name="eu-central-1",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    return client
