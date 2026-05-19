from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential
from tenacity.retry import retry_if_exception_type
from aws_bedrock import get_bedrock_client

load_dotenv()

client = get_bedrock_client()


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

Example 12 (noisy email with security banners and forwarded chain — extract the guest's actual question):
Input: "CAUTION: This message was sent from outside your organization.sophospsmartbannerend

Still hoping our two rooms can be side by side.

From: Front Desk <stay@hotel.com> Sent: Tue, 15 Jul 2026 11:00 To: You
Thanks — we will note your preferences.

CAUTION: This message was sent from outside your organization.sophospsmartbannerend

Still hoping our two rooms can be side by side.

From: Maya Chen <maya@guest.com> Sent: Tue, 15 Jul 2026 08:40 To: Front Desk
Booking #8841 for Aug 2–5. Still hoping our two rooms can be side by side."
Output: "json {
    "has_questions": true,
    "extracted_questions": [
        "Can our two rooms be side by side?"
    ],
    "language": "en"
}"

"""

tool_config = {
    "tools": [
        {
            "toolSpec": {
                "name": "classify_email",
                "description": "Classify a hotel guest email and extract any questions.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "has_questions": {
                                "type": "boolean",
                                "description": "True if the email contains questions about hotel services, policies, or amenities",
                            },
                            "extracted_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions extracted in the same language as the email. Empty array if none.",
                            },
                            "language": {
                                "type": "string",
                                "description": "ISO 639-1 language code (e.g. 'en', 'de', 'es', 'fr')",
                            },
                        },
                        "required": [
                            "has_questions",
                            "extracted_questions",
                            "language",
                        ],
                    }
                },
            }
        }
    ],
    "toolChoice": {"tool": {"name": "classify_email"}},  # force the tool call
}


@retry(
    retry=retry_if_exception_type(ClientError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
)
def classify_question(question: str) -> dict:

    response = client.converse(
        modelId="eu.amazon.nova-micro-v1:0",
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 500},
        toolConfig=tool_config,
    )
    for block in response["output"]["message"]["content"]:
        if "toolUse" in block:
            return block["toolUse"]["input"]

    raise ValueError("Model did not return a tool call")
