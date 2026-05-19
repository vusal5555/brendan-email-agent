from models import FaqChunk
from botocore.exceptions import ClientError
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type
from aws_bedrock import get_bedrock_client


client = get_bedrock_client()

system_prompt = """

Role: You are a helpful assistant that can answer questions about hotel services, policies, and amenities.

Task:
1. Only Answer the questions about the hotel services, policies, and amenities based on the given FAQ context.
2. Only answer the questions based on the given FAQ context do not make up information.
3. If the provided FAQ context does not contain any information about the question, say You will forward this question to the hotel reception.
4. If the question is not about the hotel services, policies, or amenities, say You will forward this question to the hotel reception.
5.If the FAQ context below doesn't actually answer the question, say you'll forward to reception. Don't stretch.
6. Tone: direct, professional, and concise. State the answer plainly—no greetings, pleasantries, filler phrases, enthusiasm, or sign-offs unless the guest used them.
7. Respond in the same language as the question.


"""

user_message = """
Question: {question}
Extracted Questions: {extracted_questions}
FAQ Context: {faq_chunks}
"""


@retry(
    retry=retry_if_exception_type((ClientError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def email_agent(
    question: str, extracted_questions: list[str], chunks: list[FaqChunk]
) -> str:

    faqs = "\n".join([chunk.embedding_input for chunk in chunks])
    response = client.converse(
        modelId="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        system=[{"text": system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": user_message.format(
                            question=question,
                            extracted_questions=extracted_questions,
                            faq_chunks=faqs,
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"temperature": 0, "maxTokens": 500},
    )

    assistant_message = response["output"]["message"]
    for block in assistant_message["content"]:
        if "text" in block:
            return block["text"]
    raise ValueError("Model did not return a text block")
