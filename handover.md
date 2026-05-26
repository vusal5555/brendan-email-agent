# Hotel Brendan — Handover Guide

Sample requests, code map, and deployment notes for the hotel FAQ service.

---

## API Overview

This is a **FastAPI** application (default port **8000**). There is **no authentication** on the endpoints today.

| Method | Path       | Purpose                                              |
|--------|------------|------------------------------------------------------|
| `GET`  | `/`        | Health check                                         |
| `POST` | `/answer`  | Answer guest email/questions from ingested FAQs      |
| `POST` | `/website` | Crawl hotel website URLs and ingest FAQs (async)     |
| `POST` | `/pdf`     | Upload a PDF and ingest FAQs (async)                 |

**Interactive docs** (when the server is running):

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Sample Requests

### 1. Health Check

```bash
curl http://localhost:8000/
```

**Response:**

```json
{"message": "Hello, World!"}
```

---

### 2. Answer Guest Questions — `POST /answer`

**Content-Type:** `application/json`

The `question` field can be a **full email** (not just one sentence). The service classifies it, extracts questions, retrieves FAQ chunks, and generates answers.

**Single question (English):**

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_code": "EURMAR",
    "question": "Hi, do you have parking available at the hotel? What is the daily rate per car?"
  }'
```

**Multi-question email:**

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_code": "EURMAR",
    "question": "Hello, two quick questions. First, when is breakfast served? And second, do you offer private transport to the airport?"
  }'
```

**German example:**

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_code": "EURMAR",
    "question": "Guten Tag, wann wird das Frühstück serviert und was kostet es?"
  }'
```

**Request body schema:**

| Field        | Type   | Required | Description                                       |
|--------------|--------|----------|---------------------------------------------------|
| `hotel_code` | string | yes      | Hotel identifier (must exist in the FAQ database) |
| `question`   | string | yes      | Guest email or question text                      |

**Typical response:**

```json
{
  "answers": [
    {
      "question": "Do you have parking available at the hotel?",
      "answer": "...generated answer from FAQ context...",
      "confidence": 0.85
    },
    {
      "question": "What is the daily rate per car?",
      "answer": "...",
      "confidence": 0.72
    }
  ]
}
```

**How to interpret the response:**

- `answers` is a list — one entry per extracted question.
- `confidence` = `1 - retrieval_distance` (threshold in code: **0.20**).
- Empty `answer` or a canned “please contact reception” reply → treat as **forward to front desk** (no confident FAQ match).
- Empty `answers` list → no questions detected in the input.

**Known hotel codes in test data** (must have ingested FAQs in the DB):

| Code      | Coverage | Approx. FAQs |
|-----------|----------|--------------|
| `EURMAR`  | Rich     | ~455         |
| `EURETH`  | Medium   | ~97          |
| `AT10001` | Thin     | ~15          |
| `AMBERRA` | Sparse   | ~4           |

---

### 3. Ingest from Website — `POST /website`

**Content-Type:** `application/json`

Runs in the **background**; the API returns immediately while crawling/ingesting continues.

```bash
curl -X POST http://localhost:8000/website \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_code": "MYHOTEL01",
    "urls": [
      "https://example-hotel.com/",
      "https://example-hotel.com/amenities"
    ],
    "language": "en"
  }'
```

**Request body schema:**

| Field        | Type          | Required | Description                    |
|--------------|---------------|----------|--------------------------------|
| `hotel_code` | string        | yes      | Hotel identifier               |
| `urls`       | array[string] | yes      | Pages or sitemap URLs to crawl |
| `language`   | string        | yes      | 2-letter ISO code (see below)  |

**Response:**

```json
{"message": "Website ingestion started"}
```

**Supported language codes:** `en`, `de`, `es`, `fr`, `it`, `pt`, `ca`

---

### 4. Ingest from PDF — `POST /pdf`

**Content-Type:** `multipart/form-data` (not JSON)

```bash
curl -X POST http://localhost:8000/pdf \
  -F "pdf_file=@/path/to/hotel-info.pdf" \
  -F "hotel_code=MYHOTEL01" \
  -F "language=en"
```

**Form fields:**

| Field        | Type | Required | Description           |
|--------------|------|----------|-----------------------|
| `pdf_file`   | file | yes      | PDF document          |
| `hotel_code` | text | yes      | Hotel identifier      |
| `language`   | text | yes      | 2-letter ISO code     |

**Constraints:**

- Max file size: **10 MB**
- Must be a valid PDF (`%PDF` magic bytes)
- Allowed content types: `application/pdf`, `application/x-pdf`

**Response:**

```json
{"message": "PDF ingestion started"}
```

---

## Code Map

| File | Purpose |
|------|---------|
| `app/api.py` | HTTP endpoints and request/response models |
| `app/classifier.py` | Language detection and question extraction |
| `app/retrieve.py` | Vector search against FAQ chunks in PostgreSQL |
| `app/generate.py` | LLM answer generation via AWS Bedrock |
| `app/helpers.py` | Website crawl and PDF ingest pipelines |
| `app/db.py` | Database session and engine setup |
| `app/aws_secrets.py` | DB URL from AWS Secrets Manager or `DB_URL` env |
| `app/aws_bedrock.py` | Bedrock client (`eu-central-1`) |
| `app/models.py` | `faq_chunks` table schema (pgvector embeddings) |
| `app/etl.py` | Bulk FAQ import from MySQL |
| `app/create_tables.py` | Create DB tables |
| `tests/e2e_test.py` | End-to-end tests (`--http` mode hits live `/answer`) |

**Main request models** (`app/api.py`):

```python
class AnswerRequest(BaseModel):
    hotel_code: str
    question: str


class WebsiteRequest(BaseModel):
    hotel_code: str
    urls: list[str]
    language: str
```

---

## Deployment

### Prerequisites

1. **Python 3.13** (or use the provided Docker image)
2. **PostgreSQL with pgvector** (see `docker-compose.yml`)
3. **AWS credentials** with access to:
   - **Bedrock Runtime** in `eu-central-1` (embeddings + LLM)
   - Optionally **Secrets Manager** secret `EmbeddingsDatabaseAdminAccess` (production DB)
4. **Environment variable** (if not using AWS Secrets Manager):

   ```env
   DB_URL=postgresql+psycopg2://hotel-db:hotel-db@localhost:5432/hotel-db
   ```

### Option A — Local Development

```bash
# 1. Start Postgres
docker compose up -d

# 2. Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Create tables
PYTHONPATH=app python app/create_tables.py

# 4. Run API
cd app && uvicorn api:app --host 0.0.0.0 --port 8000
```

### Option B — Docker for the API

```bash
docker build -t hotel-brendan .
docker run -p 8000:8000 \
  -e DB_URL="postgresql+psycopg2://..." \
  -e AWS_ACCESS_KEY_ID=... \
  -e AWS_SECRET_ACCESS_KEY=... \
  -e AWS_DEFAULT_REGION=eu-central-1 \
  hotel-brendan
```

> **Note:** `docker-compose.yml` only runs the **database**, not the API container. Run the API separately (locally or via `docker run`).

### Option C — Production

- Point the app at shared Postgres via AWS Secrets Manager (`app/aws_secrets.py`).
- Ensure the host has IAM permissions for Bedrock and Secrets Manager.
- Populate FAQs first (ETL via `app/etl.py`, or ingest via `/website` / `/pdf`).
- Put a reverse proxy (nginx, ALB, etc.) in front if exposed publicly.

### Deploying on Another Machine

1. Clone the repository onto the target machine.
2. Install Docker (for Postgres) and/or Python 3.13.
3. Configure AWS credentials (IAM role on EC2, or `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`).
4. Set `DB_URL` in `.env` for local/dev, or rely on Secrets Manager in production.
5. Start Postgres: `docker compose up -d`
6. Create tables: `PYTHONPATH=app python app/create_tables.py`
7. Run the API (locally or via Docker as shown above).
8. Verify with `curl http://<host>:8000/` and optionally `python tests/e2e_test.py --http --url http://<host>:8000`.

### Verify After Deploy

```bash
curl http://<host>:8000/

python tests/e2e_test.py --http --url http://<host>:8000
```

---

## Pipeline Flow (`/answer`)

```
Guest email
    → classify_question()     (language + extracted questions)
    → retrieve_faqs()           (vector search per question)
    → email_agent()             (LLM answer if confidence ≥ 0.20)
    → AnswerResponse
```

---

## Common Follow-ups

- **Which hotel codes exist?** Run `python tests/e2e_test.py --discover` against a connected DB.
- **How does retrieval work?** See `app/inspect_retrieval_parking.py` for a debug script.
- **How to add FAQs?** Use `/website`, `/pdf`, or the MySQL ETL in `app/etl.py`.
- **How to run tests?** `python tests/e2e_test.py` (direct) or `python tests/e2e_test.py --http` (against a running server).
