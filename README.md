# Hotel Brendan

FastAPI service that answers guest emails using hotel FAQ knowledge. It classifies incoming text, retrieves relevant FAQ chunks from PostgreSQL (pgvector), and generates replies via AWS Bedrock.

For a fuller code walkthrough and handover notes, see [handover.md](handover.md).

---

## Prerequisites

- **Python 3.13** (or Docker)
- **PostgreSQL with pgvector** — provided via `docker-compose.yml` for local dev
- **AWS credentials** with access to:
  - **Bedrock Runtime** in `eu-central-1`
  - Optionally **Secrets Manager** (`EmbeddingsDatabaseAdminAccess`) for production DB credentials

**Environment (local/dev):** create a `.env` file in the project root if you are not using AWS Secrets Manager:

```env
DB_URL=postgresql+psycopg2://hotel-db:hotel-db@localhost:5432/hotel-db
AWS_DEFAULT_REGION=eu-central-1
```

---

## Deployment

### Local development

```bash
# 1. Start Postgres (pgvector)
docker compose up -d

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Create database tables
PYTHONPATH=app python app/create_tables.py

# 4. Run the API
cd app && uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive API docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Docker (API container)

`docker-compose.yml` runs **only the database**. Build and run the API separately:

```bash
docker build -t hotel-brendan .
docker run -p 8000:8000 \
  -e DB_URL="postgresql+psycopg2://hotel-db:hotel-db@host.docker.internal:5432/hotel-db" \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_DEFAULT_REGION=eu-central-1 \
  hotel-brendan
```

Adjust `DB_URL` so the container can reach your Postgres host.

### Another machine / production

1. Clone the repo onto the target machine.
2. Ensure AWS credentials are available (IAM role or env vars).
3. Point the app at Postgres:
   - **Production:** `app/aws_secrets.py` reads `EmbeddingsDatabaseAdminAccess` from Secrets Manager automatically.
   - **Dev/staging:** set `DB_URL` in `.env`.
4. Start Postgres (or use a managed instance with pgvector enabled).
5. Run `PYTHONPATH=app python app/create_tables.py` once to create tables.
6. Start the API (uvicorn locally or via Docker as above).
7. Populate FAQs before expecting answers (ETL, `/website`, or `/pdf` — see [handover.md](handover.md)).
8. Optionally put nginx or a load balancer in front for public access.

**Verify:**

```bash
curl http://<host>:8000/
python tests/e2e_test.py --http --url http://<host>:8000
```

---

## API endpoints

| Method | Path       | Description                          |
|--------|------------|--------------------------------------|
| `GET`  | `/`        | Health check                         |
| `POST` | `/answer`  | Answer guest email / questions       |
| `POST` | `/website` | Ingest FAQs from website URLs (async)|
| `POST` | `/pdf`     | Ingest FAQs from PDF upload (async)  |

There is **no authentication** on these endpoints today.

---

## Sample requests

Replace `localhost:8000` with your host when testing a remote deployment.

### Health check

```bash
curl http://localhost:8000/
```

```json
{"message": "Hello, World!"}
```

### Answer — `POST /answer`

Send JSON. The `question` field can be a full guest email; the service extracts one or more questions automatically.

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

**Example response:**

```json
{
  "answers": [
    {
      "question": "Do you have parking available at the hotel?",
      "answer": "...",
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

| Field        | Type   | Description                          |
|--------------|--------|--------------------------------------|
| `hotel_code` | string | Hotel code with ingested FAQs in DB  |
| `question`   | string | Guest email or question text         |

**Reading the response:**

- One object in `answers` per extracted question.
- `confidence` is based on FAQ retrieval similarity (threshold in code: **0.20**).
- Empty `answer` → treat as forward to reception (no confident match).

**Test hotel codes** (when connected to a populated DB): `EURMAR`, `EURETH`, `AT10001`, `AMBERRA`.

### Ingest website — `POST /website`

Returns immediately; ingestion runs in the background.

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

```json
{"message": "Website ingestion started"}
```

Language codes: `en`, `de`, `es`, `fr`, `it`, `pt`, `ca`.

### Ingest PDF — `POST /pdf`

Multipart form upload (max **10 MB**, valid PDF only).

```bash
curl -X POST http://localhost:8000/pdf \
  -F "pdf_file=@/path/to/hotel-info.pdf" \
  -F "hotel_code=MYHOTEL01" \
  -F "language=en"
```

```json
{"message": "PDF ingestion started"}
```

---

## Project layout

```
app/
  api.py           # HTTP endpoints
  classifier.py    # Question extraction + language detection
  retrieve.py      # Vector search (pgvector)
  generate.py      # Bedrock LLM answers
  helpers.py       # Website + PDF ingestion
  db.py            # Database connection
  aws_secrets.py   # DB credentials (Secrets Manager / DB_URL)
tests/
  e2e_test.py      # End-to-end tests (--http for live API)
docker-compose.yml # Local Postgres + pgvector
Dockerfile         # API container image
```

---

## Further reading

- [handover.md](handover.md) — detailed handover guide (code map, pipeline flow, FAQ population, common questions)
