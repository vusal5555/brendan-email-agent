from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
import os
import json
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """
Role: You analyze incoming hotel guest emails to determine if they contain questions.
Task: 
1.Determine if this email contains questions about hotel services, policies, or amenities. If yes, extract the questions in the same language as the email. If no, indicate that
2. Determine the language of the email.

Example 1:
Input: "Hi, we're arriving on Friday. Could you let us know if parking is available and how much it costs?
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Is parking available?",
        "How much does parking cost?"
    ],
    "language": "en"
}"

Example 2:
Input: "Thanks for the confirmation, we look forward to our stay!"
Output: "json {
    "has_questions": false,
    "extracted_questions": [],
    "language": "en"
}

Example 3:
Input: "Hello, what time is breakfast served and do you allow pets in the rooms"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "What time is breakfast served?",
        "Do you allow pets in the rooms?"
    ],
    "language": "en"
}"

Example 4:
Input: "Hola, ¿tiene parking el hotel?"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "¿Tiene parking el hotel?"
    ],
    "language": "es"
}"

Example 5:
Input: "Bonjour, avez-vous un parking à l'hôtel?"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Avez-vous un parking à l'hôtel?"
    ],
    "language": "fr"
}"

Example 6 (booking mixed with question — extract question only):
Input: "Hi, I'd like to book a double room for July 14-17 for two adults. Also, do you have airport shuttle service?"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Do you have airport shuttle service?"
    ],
    "language": "en"
}"

Example 7 (indirect statement implying a question):
Input: "Hey, was wondering about the pool situation. Also not sure if you have a gym on site."
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Is there a pool and what are the details?",
        "Is there a gym on site?"
    ],
    "language": "en"
}"

Example 8 (long email with question buried at the end):
Input: "Hello team, hope you're doing well. We're a family of four flying in from Manchester on the 22nd, arriving around 4pm at the airport. We've been planning this trip for almost a year and the kids are very excited. We booked the suite with the sea view based on the photos on your website — it looks stunning. We're also planning a day trip to the old town and one to the vineyards nearby. Quick thing though, is early check-in possible?"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Is early check-in possible?"
    ],
    "language": "en"
}"

Example 9 (mixed-language email — report the dominant language):
Input: "Hallo, wir kommen am Freitag an. Quick question — do you have a spa on site? Vielen Dank!"
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Do you have a spa on site?"
    ],
    "language": "de"
}"

Example 10 (long warm thanks, no question — must not hallucinate):
Input: "Hi team, just wanted to drop a quick note to say thank you for being so responsive over the past few weeks. We've been planning this anniversary trip for months and it means a lot that you've been so accommodating with all our requests. Really looking forward to finally arriving on Saturday. See you soon!"
Output: "json {
    "has_questions": false,
    "extracted_questions": [],
    "language": "en"
}"

Example 11 (complaint with no question):
Input: "The room was filthy when we checked in yesterday and the front desk staff was rude when we brought it up. This is not the experience we paid for."
Output: "json {
    "has_questions": false,
    "extracted_questions": [],
    "language": "en"
}"  

"""


@retry(
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def classify_question(question: str) -> dict:
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
