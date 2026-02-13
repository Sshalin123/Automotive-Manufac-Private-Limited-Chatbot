# AMPL Automotive RAG Chatbot

An AI-powered chatbot for automotive sales with RAG (Retrieval-Augmented Generation), lead scoring, and CRM integration.

## Features

- **Intelligent RAG Pipeline**: Accurate, document-grounded answers for vehicle specs, pricing, and offers
- **Lead Scoring (90%+ accuracy)**: Intent classification + entity extraction + rule-based scoring
- **24/7 Digital Showroom**: Inventory queries, financing info, test drive booking
- **CRM Integration**: Automatic lead routing to Zoho, HubSpot, Salesforce, or custom webhooks
- **Multi-namespace Vector Storage**: Separate indexes for inventory, sales, insurance, FAQs

## Quick Start

### Prerequisites

- Python 3.11+
- Pinecone account
- AWS account (for Bedrock) or OpenAI API key
- Docker (optional)

### Installation

1. Clone and setup:
```bash
cd ampl-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your credentials
```

3. Run the API:
```bash
uvicorn api.main:app --reload
```

4. Access the API:
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Docker Deployment

```bash
cd docker
docker-compose up -d
```

## Project Structure

```
ampl-chatbot/
├── api/                    # FastAPI application
│   ├── routes/            # API endpoints (chat, leads, admin)
│   └── middleware/        # Auth, logging
├── ingest_ampl/           # Data ingestion module
│   ├── inventory_loader.py
│   ├── sales_docs_loader.py
│   ├── insurance_loader.py
│   └── faq_loader.py
├── lead_scoring/          # Lead qualification
│   ├── intent_classifier.py
│   ├── entity_extractor.py
│   ├── scoring_model.py
│   └── lead_router.py
├── retrieval/             # RAG retrieval layer
│   ├── embedder.py
│   ├── pinecone_client.py
│   └── context_builder.py
├── llm/                   # LLM orchestration
│   ├── orchestrator.py
│   └── prompt_templates.py
├── config/                # Configuration files
│   ├── scoring_rules.yaml
│   └── metadata_schema.yaml
└── docker/               # Docker configuration
```

## API Endpoints

### Chat
- `POST /api/v1/chat` - Send message, get AI response
- `GET /api/v1/chat/{id}/history` - Get conversation history

### Leads
- `POST /api/v1/leads` - Create lead
- `GET /api/v1/leads` - List leads (with filtering)
- `PATCH /api/v1/leads/{id}` - Update lead
- `GET /api/v1/leads/stats/summary` - Get lead statistics

### Admin
- `GET /api/v1/admin/health` - System health
- `GET /api/v1/admin/index/stats` - Vector index stats
- `POST /api/v1/admin/ingestion/trigger` - Trigger ingestion

## Lead Scoring

Scoring rules (0-100):
- BUY intent: +30
- FINANCE intent: +25
- TEST_DRIVE intent: +20
- Model mentioned: +15
- Budget mentioned: +10
- Immediate timeline: +15
- Contact provided: +10

Priority thresholds:
- **Hot (≥70)**: Immediate CRM webhook
- **Warm (50-69)**: Queue for review
- **Cold (<50)**: Nurture campaign

## Environment Variables

Key variables (see `.env.example` for full list):

```bash
# Pinecone
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=ampl-chatbot

# AWS Bedrock
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BEDROCK_LLM_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# Or OpenAI
OPENAI_API_KEY=xxx
LLM_PROVIDER=openai

# CRM Integration
CRM_WEBHOOK_URL=https://your-crm.com/webhook
```

## Data Ingestion

1. Prepare data files in `data/` directory
2. Run ingestion:
```bash
python -m ingest_ampl.main --source-type local --path ./data
```

Or with Docker:
```bash
docker-compose --profile ingestion up ingest-worker
```

## Architecture

```
User → API → Intent Classifier → Embedding → Pinecone Query
                    ↓                              ↓
              Entity Extractor            Context Builder
                    ↓                              ↓
              Lead Scorer  ←────────────────  LLM Response
                    ↓
              Lead Router → CRM Webhook
```

## Testing

```bash
pytest tests/ -v
```

## Production Deployment

For production, ensure:
- [ ] HTTPS with valid SSL certificate
- [ ] API key authentication enabled
- [ ] Rate limiting configured
- [ ] Monitoring (Sentry, CloudWatch) enabled
- [ ] Database for persistent storage
- [ ] Redis for caching
- [ ] Auto-scaling configured

## License

Proprietary - AMPL Automotive
