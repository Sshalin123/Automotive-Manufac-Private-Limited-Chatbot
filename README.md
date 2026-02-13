# AMPL Automotive RAG Chatbot

An AI-powered chatbot for automotive sales with RAG (Retrieval-Augmented Generation), lead scoring, CRM integration, and full customer lifecycle management — from enquiry to delivery to post-sale service.

## Features

### Core AI Pipeline
- **Intelligent RAG Pipeline**: Document-grounded answers from Pinecone vector DB (4 namespaces: inventory, sales, insurance, FAQs)
- **15 Intent Classification**: BUY, FINANCE, TEST_DRIVE, EXCHANGE, INSURANCE, SERVICE, COMPLAINT, INFO, COMPARISON + BOOKING_CONFIRM, PAYMENT_CONFIRM, SERVICE_REMINDER, FEEDBACK, ESCALATION, DELIVERY_UPDATE
- **Entity Extraction (30+ fields)**: Phone, email, budget, model, city, color, trade-in, buyer type, customer type, demographics, need analysis, feedback/NPS
- **Lead Scoring (0-100)**: Rule-based + LLM fallback with automatic CRM routing for hot leads
- **Multilingual Support**: Hindi (Devanagari + Hinglish), Marathi, Gujarati — auto-detected from message script

### Customer Lifecycle
- **Enquiry Handling**: Post-enquiry RM greeting, 15-day tracking, source capture (widget/WhatsApp/walk-in), pre-booking enforcement
- **Need Analysis**: 6-question profiling (purchase mode, income, travel, competitors, priorities, usage), buyer type (first-time/replacement/additional), customer type (individual/corporate)
- **Booking & Payment**: Payment confirmation (Yes/No flow), receipt trail (Enquiry ID -> Booking ID -> Customer ID), DMS receipt links
- **Delivery & Post-Delivery**: Delivery confirmation with photo, delay notifications, post-delivery sequence (Day-0/Day-1/Day-15), RC pendency follow-up
- **Feedback & NPS**: Rating collection (Poor/Fair/Very Good/Excellent), NPS (0-10), LLM sentiment analysis, keyword extraction
- **Service Reminders**: Milestone-based (7-day welcome, 1000km, 5000km, 10000km), toll-free number in all service responses
- **Escalation**: Full matrix (RM -> Sales Head -> Service Head -> GM), 15-day SLA tracking

### Infrastructure
- **CRM Integration**: Automatic lead routing to Zoho, HubSpot, Salesforce, or custom webhooks
- **Multi-Channel Notifications**: Chat widget, WhatsApp API, Email — with bulk send and delivery logging
- **DMS Webhooks**: Payment, delivery, service-complete, job-card event receivers
- **Scheduled Messaging**: Follow-ups (N+0/N+1/N+15), service reminders, SLA checks
- **Mock Fallback**: Works when LLM services are unavailable

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
│   ├── routes/
│   │   ├── chat.py         # Chat, feedback, payment-confirm endpoints
│   │   ├── leads.py        # Lead management endpoints
│   │   ├── admin.py        # Admin & health endpoints
│   │   ├── webhooks.py     # DMS event webhooks (payment, delivery, service, job-card)
│   │   ├── scheduled.py    # Scheduled messages (follow-ups, reminders, SLA)
│   │   └── notifications.py # Multi-channel dispatcher (widget, WhatsApp, email)
│   ├── services.py         # Service initialization & dependency injection
│   └── main.py             # App factory & router registration
├── lead_scoring/           # Lead qualification
│   ├── intent_classifier.py  # 15-intent classifier (rule-based + LLM)
│   ├── entity_extractor.py   # 30+ field extraction (regex-based)
│   ├── scoring_model.py      # Rule-based scoring engine
│   └── lead_router.py        # CRM webhook routing (Zoho/HubSpot/Salesforce)
├── retrieval/              # RAG retrieval layer
│   ├── embedder.py           # Embedding service (OpenAI / Bedrock Titan)
│   ├── pinecone_client.py    # Vector DB client (4 namespaces)
│   ├── context_builder.py    # Priority-based context assembly
│   └── reranker.py           # Result reranking
├── llm/                    # LLM orchestration
│   ├── orchestrator.py       # Full pipeline (intent -> RAG -> LLM -> score -> route)
│   └── prompt_templates.py   # 12 prompt types with system + user templates
├── ingest_ampl/            # Data ingestion module
│   ├── inventory_loader.py
│   ├── sales_docs_loader.py
│   ├── insurance_loader.py
│   └── faq_loader.py
├── config/                 # Configuration
│   ├── settings.py           # Pydantic settings (env-based)
│   ├── scoring_rules.yaml    # Lead scoring rules
│   └── metadata_schema.yaml  # Vector metadata schema
├── data/                   # Source data
│   ├── faqs.json             # 23 FAQs (financing, service, delivery, escalation, etc.)
│   └── inventory.csv         # Vehicle inventory
├── frontend/               # Chat widget
└── docker/                 # Docker configuration
```

## API Endpoints

### Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/chat` | Send message, get AI response with lead scoring |
| `GET` | `/api/v1/chat/{id}/history` | Get conversation history |
| `DELETE` | `/api/v1/chat/{id}` | Clear conversation |
| `GET` | `/api/v1/chat/stats` | Chat statistics |
| `POST` | `/api/v1/chat/feedback` | Submit feedback rating + NPS + sentiment analysis |
| `POST` | `/api/v1/chat/payment-confirm` | Send payment confirmation (Yes/No) to customer |

### Webhooks (DMS Events)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/webhooks/payment` | Payment received — sends confirmation to customer |
| `POST` | `/api/v1/webhooks/delivery` | Delivery event — confirmation, delay, or welcome message |
| `POST` | `/api/v1/webhooks/service-complete` | Service done — sends follow-up + feedback request |
| `POST` | `/api/v1/webhooks/job-card` | Job card events (opened, in_progress, closed) |

### Scheduled Messages
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/scheduled/followup` | Schedule follow-ups (enquiry/delivery/service) |
| `POST` | `/api/v1/scheduled/service-reminders` | Schedule service milestone reminders |
| `POST` | `/api/v1/scheduled/sla-check` | Check for 15-day stale complaints |
| `GET` | `/api/v1/scheduled/messages` | List scheduled messages |
| `DELETE` | `/api/v1/scheduled/messages/{id}` | Cancel a scheduled message |

### Notifications
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/notifications/send` | Send via widget, WhatsApp, or email |
| `POST` | `/api/v1/notifications/bulk` | Bulk send to multiple customers |
| `GET` | `/api/v1/notifications/log` | Notification delivery log |

### Leads
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/leads` | Create lead |
| `GET` | `/api/v1/leads` | List leads (with filtering) |
| `PATCH` | `/api/v1/leads/{id}` | Update lead |
| `GET` | `/api/v1/leads/stats/summary` | Lead statistics |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/admin/health` | System health |
| `GET` | `/api/v1/admin/index/stats` | Vector index stats |
| `POST` | `/api/v1/admin/ingestion/trigger` | Trigger data ingestion |

## Lead Scoring

Scoring rules (0-100):

| Intent | Score |
|--------|-------|
| BUY | +30 |
| BOOKING_CONFIRM | +25 |
| FINANCE | +25 |
| TEST_DRIVE | +20 |
| PAYMENT_CONFIRM | +20 |
| EXCHANGE | +15 |
| DELIVERY_UPDATE | +15 |
| INSURANCE | +10 |
| SERVICE / SERVICE_REMINDER | +5 |
| FEEDBACK | +5 |
| INFO | 0 |
| ESCALATION | -10 |
| COMPLAINT | -20 |

Entity bonuses: Model mentioned (+15), Budget mentioned (+10), Budget > 15L (+5), Immediate timeline (+15), Contact provided (+10), Trade-in mentioned (+5)

Priority thresholds:
- **Hot (>=70)**: Immediate CRM webhook routing
- **Warm (50-69)**: Queued for review
- **Cold (<50)**: Nurture campaign

## Environment Variables

Key variables (see `.env.example` for full list):

```bash
# LLM Provider
LLM_PROVIDER=openai              # or "bedrock"
OPENAI_API_KEY=xxx
# Or AWS Bedrock
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
BEDROCK_LLM_MODEL_ID=us.anthropic.claude-sonnet-4-20250514-v1:0

# Pinecone
PINECONE_API_KEY=xxx
PINECONE_INDEX_NAME=ampl-chatbot

# CRM Integration
CRM_WEBHOOK_URL=https://your-crm.com/webhook
CRM_PROVIDER=zoho                # zoho, hubspot, salesforce, custom

# Relationship Manager
RM_NAME=AMPL Sales Team
RM_PHONE=+91-XXXXXXXXXX
RM_EMAIL=sales@amplindia.com
WEBSITE_URL=https://www.amplindia.com
TOLL_FREE_NUMBER=1800-XXX-XXXX

# Escalation (JSON array)
ESCALATION_CONTACTS='[{"role":"RM","name":"...","phone":"..."},{"role":"Sales Head","name":"...","phone":"..."}]'

# Service Milestones (JSON array)
SERVICE_MILESTONES='[{"name":"Welcome Call","days":7},{"name":"1st Free Service","km":1000,"days":30},{"name":"2nd Free Service","km":5000,"days":180},{"name":"3rd Free Service","km":10000,"days":365}]'

# Follow-up Intervals (days after enquiry)
FOLLOWUP_INTERVALS='[0, 1, 15]'
```

## Architecture

```
                          ┌──────────────────────────────────┐
                          │         DMS / External           │
                          │  (Payment, Delivery, Service)    │
                          └──────────┬───────────────────────┘
                                     │ Webhooks
                                     v
User ──> API ──> Intent Classifier (15 intents)
                      │
                      ├──> Entity Extractor (30+ fields)
                      │         │
                      │         v
                      │    Lead Scorer ──> Lead Router ──> CRM Webhook
                      │
                      ├──> Language Detection (Hindi/Marathi/Gujarati)
                      │
                      ├──> Embedding ──> Pinecone Query ──> Context Builder
                      │                                          │
                      v                                          v
                 Prompt Builder ──────────────────────────> LLM Response
                      │                                          │
                      │    ┌─────────────────────────────────────┘
                      v    v
                 Sentiment Analysis (feedback messages)
                      │
                      v
              Stage Tracker (enquiry -> booked -> delivered -> servicing)
                      │
                      v
         ┌────────────┴────────────┐
         │   Notification Router   │
         ├─────────┬───────┬───────┤
         │ Widget  │ WhatsApp │ Email │
         └─────────┴───────┴───────┘
                      │
                      v
              Scheduled Messages
         (Follow-ups, Reminders, SLA)
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
- [ ] Database for persistent storage (replace in-memory stores)
- [ ] Redis for caching
- [ ] Task queue (Celery/APScheduler) for scheduled messages
- [ ] WhatsApp Business API provider (Gupshup/Twilio/Meta Cloud API)
- [ ] Email service (SendGrid/SES) for email notifications
- [ ] Auto-scaling configured

## License

Proprietary - AMPL Automotive
