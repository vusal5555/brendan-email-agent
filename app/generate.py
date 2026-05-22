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
FAQ Context: {faq_chunks}
"""


@retry(
    retry=retry_if_exception_type((ClientError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def email_agent(question: str, chunks: list[FaqChunk]) -> str:

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


faq_system_prompt = """
Role: You generate hotel FAQ pairs from website page content for a guest-facing knowledge base.

Task: From the page content provided, generate question-and-answer pairs that a hotel guest would realistically ask.

Rules:
1. Generate 3–7 Q/A pairs per chunk, depending on how much factual hotel information the text contains. Sparse chunks → fewer pairs; dense chunks → more (up to 7).
2.Generate all Q/A pairs in specified language regardless of the source text language.
2. Write questions and answers in the same language as the page content.
3. Only create pairs where the answer is clearly supported by the text. Do not invent policies, prices, times, or amenities not stated in the content.
4. Questions: natural guest style (e.g. "Do you have…?", "What time is…?", "Is … included?").
5. Answers: direct, factual, concise—no greetings or filler.
6. Skip boilerplate, navigation, or marketing fluff that does not answer a concrete guest question.
7. If the chunk has almost no usable facts, return an empty faqs array.

Output: Use the generate_faqs tool only. Do not include markdown, preamble, or any text outside the tool call.
"""

faq_user_message = """
Page content:
{chunk}
Language: {language}
"""

faq_tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "generate_faqs",
                "description": "Generate hotel guest FAQ pairs supported by the page content.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "faqs": {
                                "type": "array",
                                "description": "3–7 Q/A pairs when content supports them; fewer or empty if sparse.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {
                                            "type": "string",
                                            "description": "Natural guest-style question",
                                        },
                                        "answer": {
                                            "type": "string",
                                            "description": "Direct factual answer from the text only",
                                        },
                                    },
                                    "required": ["question", "answer"],
                                },
                            }
                        },
                        "required": ["faqs"],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "generate_faqs"}},
}


@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def faq_generator(chunk: str, language: str) -> list[dict[str, str]]:
    response = client.converse(
        modelId="eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        system=[{"text": faq_system_prompt}],
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": faq_user_message.format(chunk=chunk, language=language)}
                ],
            }
        ],
        inferenceConfig={"temperature": 0, "maxTokens": 1500},
        toolConfig=faq_tool_config,
    )

    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"]["faqs"]

    raise ValueError("Model did not return a tool call")
