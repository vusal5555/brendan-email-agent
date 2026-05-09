from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
Role: You analyze incoming hotel guest emails to determine if they contain questions.
Task: 
1.Determine if this email contains questions about hotel services, policies, or amenities. If yes, extract the questions. If no, indicate that


Example 1:
Input: "Hi, we're arriving on Friday. Could you let us know if parking is available and how much it costs?
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Is parking available?",
        "How much does parking cost?"
    ]
}"

Example 2:
Input: "Thanks for the confirmation, we look forward to our stay!"
Output: "json {
    "has_questions": false,
    "extracted_questions": []
}

Example 3:
Input: "Hello, what time is breakfast served and do you allow pets in the rooms"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "What time is breakfast served?",
        "Do you allow pets in the rooms?"
    ]
}"

"""


def classify_question(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
