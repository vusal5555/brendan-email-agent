from openai import OpenAI
from dotenv import load_dotenv
import os
from models import FaqChunk

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """

Role: You are a helpful assistant that can answer questions about hotel services, policies, and amenities.

Task:
1. Only Answer the questions about the hotel services, policies, and amenities based on the given FAQ context.
2. Only answer the questions based on the given FAQ context do not make up information.
3. If the provided FAQ context does not contain any information about the question, say You will forward this question to the hotel reception.
4. If the question is not about the hotel services, policies, or amenities, say You will forward this question to the hotel reception.
5. Tone of the answer should be friendly, professional and concise.
6. Respond in the same language as the question.


"""

user_message = """
Question: {question}
Extracted Questions: {extracted_questions}
FAQ Context: {faq_chunks}
"""


def email_agent(
    question: str, extracted_questions: list[str], chunks: list[FaqChunk]
) -> str:

    faqs = "\n".join([chunk.embedding_input for chunk in chunks])
    response = client.chat.completions.create(
        model="gpt-5.4",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_message.format(
                    question=question,
                    extracted_questions=extracted_questions,
                    faq_chunks=faqs,
                ),
            },
        ],
    )
    return response.choices[0].message.content
