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

### Enterprise Modules
- **Database Layer**: SQLAlchemy async ORM with PostgreSQL (AsyncPG), Alembic migrations, repository pattern for conversations, leads, and messages
- **Authentication & Authorization**: JWT-based auth with role-based access control (admin/agent/viewer), secure password hashing via Passlib+bcrypt
- **Analytics Collector**: Real-time metrics for conversations, leads, intents, and feedback — with time-range filtering and aggregation
- **Multi-Channel Messaging**: WhatsApp Business API integration and Email (SendGrid/AWS SES) with unified channel abstraction
- **Real-Time Communication**: WebSocket and SSE (Server-Sent Events) for live chat streaming, connection management with heartbeat
- **Human Handoff**: Seamless escalation from AI to human agents with queue management and resolution tracking
- **A/B Experiments**: Prompt and model variant testing with traffic splitting, metric collection, and statistical analysis
- **GDPR Compliance**: Data export (JSON), data erasure, and full audit logging for regulatory compliance
- **Multi-Tenant Support**: Tenant isolation with per-tenant configuration, usage tracking, and scoped data access
- **Knowledge Management**: Document versioning, ingestion status tracking, scheduled refresh, and document lifecycle management
- **LLM Guardrails**: Input sanitization and output safety filters to prevent prompt injection and harmful content
- **Query Preprocessing**: Query decomposition for complex questions, intelligent namespace routing, and token estimation for cost control
- **Cumulative Lead Scoring**: Session-aware scoring that accumulates across conversation turns with decay and recalculation

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
├── api/                         # FastAPI application
│   ├── routes/
│   │   ├── chat.py              # Chat, feedback, payment-confirm endpoints
│   │   ├── leads.py             # Lead management endpoints
│   │   ├── admin.py             # Admin & health endpoints
│   │   ├── auth.py              # JWT login & registration
│   │   ├── analytics.py         # Conversation/lead/intent/feedback analytics
│   │   ├── compliance.py        # GDPR export, erasure, audit log
│   │   ├── experiments.py       # A/B experiment management
│   │   ├── handoff.py           # Human handoff queue & resolution
│   │   ├── knowledge.py         # Knowledge base ingestion & management
│   │   ├── realtime.py          # WebSocket & SSE live chat
│   │   ├── tenants.py           # Multi-tenant management
│   │   ├── webhooks.py          # DMS event webhooks (payment, delivery, service, job-card)
│   │   ├── scheduled.py         # Scheduled messages (follow-ups, reminders, SLA)
│   │   └── notifications.py     # Multi-channel dispatcher (widget, WhatsApp, email)
│   ├── analytics/
│   │   └── collector.py         # Metrics collector & aggregation engine
│   ├── channels/
│   │   ├── base.py              # Abstract channel interface
│   │   ├── whatsapp.py          # WhatsApp Business API client
│   │   └── email.py             # Email (SendGrid / AWS SES) client
│   ├── experiments/
│   │   └── engine.py            # A/B experiment engine with traffic splitting
│   ├── flows/
│   │   ├── definitions.py       # Conversation flow definitions
│   │   └── engine.py            # Flow execution engine
│   ├── handoff/
│   │   └── manager.py           # Human handoff queue manager
│   ├── knowledge/
│   │   └── version_tracker.py   # Document version & refresh tracker
│   ├── middleware/
│   │   ├── metrics.py           # Prometheus metrics middleware
│   │   └── rate_limit.py        # Token-bucket rate limiter
│   ├── realtime/
│   │   └── connection_manager.py # WebSocket/SSE connection manager
│   ├── tenants/
│   │   └── manager.py           # Tenant isolation & config manager
│   ├── services.py              # Service initialization & dependency injection
│   └── main.py                  # App factory & router registration
│
├── database/                    # Database layer
│   ├── models.py                # SQLAlchemy ORM models (conversations, leads, messages)
│   ├── repositories.py          # Repository pattern for DB operations
│   └── session.py               # Async session factory & connection pool
│
├── lead_scoring/                # Lead qualification
│   ├── intent_classifier.py     # 15-intent classifier (rule-based + LLM)
│   ├── entity_extractor.py      # 30+ field extraction (regex-based)
│   ├── scoring_model.py         # Rule-based scoring engine
│   ├── cumulative_scorer.py     # Session-aware cumulative scoring
│   └── lead_router.py           # CRM webhook routing (Zoho/HubSpot/Salesforce)
│
├── retrieval/                   # RAG retrieval layer
│   ├── embedder.py              # Embedding service (OpenAI / Bedrock Titan)
│   ├── pinecone_client.py       # Vector DB client (4 namespaces)
│   ├── context_builder.py       # Priority-based context assembly
│   ├── reranker.py              # Result reranking
│   ├── namespace_router.py      # Intent-to-namespace routing
│   ├── query_decomposer.py      # Complex query decomposition
│   ├── query_preprocessor.py    # Query cleaning & preprocessing
│   └── token_estimator.py       # Token counting for cost control
│
├── llm/                         # LLM orchestration
│   ├── orchestrator.py          # Full pipeline (intent -> RAG -> LLM -> score -> route)
│   ├── prompt_templates.py      # 12 prompt types with system + user templates
│   ├── conversation_store.py    # In-memory conversation history store
│   ├── db_conversation_store.py # Database-backed conversation store
│   └── guardrails.py            # Input/output safety filters
│
├── ingest_ampl/                 # Data ingestion module
│   ├── main.py                  # CLI ingestion script
│   ├── inventory_loader.py
│   ├── sales_docs_loader.py
│   ├── insurance_loader.py
│   └── faq_loader.py
│
├── config/                      # Configuration
│   ├── settings.py              # Pydantic settings (env-based)
│   ├── scoring_rules.yaml       # Lead scoring rules
│   └── metadata_schema.yaml     # Vector metadata schema
├── data/                        # Source data
│   ├── faqs.json                # 23 FAQs (financing, service, delivery, escalation, etc.)
│   └── inventory.csv            # Vehicle inventory (10 models)
├── frontend/                    # Chat widget
│   └── widget/                  # Embeddable HTML/JS/CSS chat widget
├── tests/                       # Test suite
│   ├── test_chat.py             # Chat endpoint tests
│   ├── test_lead_scoring.py     # Lead scoring tests
│   ├── test_ingestion.py        # Data ingestion tests
│   └── test_retrieval.py        # Retrieval pipeline tests
├── scripts/                     # Startup & utility scripts
│   ├── setup.sh / setup.bat     # Environment setup
│   ├── run_api.sh / run_api.bat # API server startup
│   └── ingest_data.sh / .bat    # Data ingestion runner
└── docker/                      # Docker configuration
    ├── Dockerfile
    ├── Dockerfile.ingest
    └── docker-compose.yml
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
| `GET` | `/api/v1/admin/config` | View current configuration |
| `PATCH` | `/api/v1/admin/config` | Update runtime configuration |

### Authentication
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/login` | Login with email/password, returns JWT token |
| `POST` | `/api/v1/auth/register` | Register new user (admin/agent/viewer roles) |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/analytics/conversations` | Conversation metrics (volume, duration, resolution) |
| `GET` | `/api/v1/analytics/leads` | Lead funnel metrics (scores, conversion, sources) |
| `GET` | `/api/v1/analytics/intents` | Intent distribution analytics |
| `GET` | `/api/v1/analytics/feedback` | Feedback & NPS analytics |

### Compliance (GDPR)
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/compliance/export/{customer_id}` | Export all customer data (JSON) |
| `DELETE` | `/api/v1/compliance/erase/{customer_id}` | Erase all customer data (right to be forgotten) |
| `GET` | `/api/v1/compliance/audit-log` | View audit trail of data operations |

### Experiments
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/experiments` | Create A/B experiment (prompt/model variants) |
| `GET` | `/api/v1/experiments` | List all experiments with status |
| `GET` | `/api/v1/experiments/{id}` | Get experiment details & results |
| `DELETE` | `/api/v1/experiments/{id}` | Stop and archive experiment |

### Human Handoff
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/handoff/active` | List active handoff requests in queue |
| `POST` | `/api/v1/handoff/resolve/{id}` | Resolve a handoff (mark as handled) |

### Knowledge Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/knowledge/ingest` | Ingest new documents into knowledge base |
| `POST` | `/api/v1/knowledge/refresh` | Trigger refresh of stale documents |
| `GET` | `/api/v1/knowledge/status` | Document ingestion status & versions |
| `DELETE` | `/api/v1/knowledge/document/{id}` | Remove document from knowledge base |

### Real-Time
| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/api/v1/realtime/ws/chat/{conversation_id}` | WebSocket live chat connection |
| `GET` | `/api/v1/realtime/sse/chat/{conversation_id}` | SSE stream for chat updates |

### Tenants
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/tenants` | Create new tenant |
| `GET` | `/api/v1/tenants` | List all tenants |
| `GET` | `/api/v1/tenants/{id}` | Get tenant details & usage |
| `DELETE` | `/api/v1/tenants/{id}` | Deactivate tenant |

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

# Database (Enterprise)
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ampl_chatbot

# Authentication (Enterprise)
JWT_SECRET=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=480

# WhatsApp Business API (Enterprise)
WHATSAPP_API_URL=https://graph.facebook.com/v18.0
WHATSAPP_API_TOKEN=xxx
WHATSAPP_PHONE_NUMBER_ID=xxx

# Email (Enterprise)
EMAIL_PROVIDER=sendgrid            # sendgrid or ses
SENDGRID_API_KEY=xxx
# Or AWS SES
AWS_SES_REGION=ap-south-1
EMAIL_FROM=noreply@amplindia.com

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60

# Multi-Tenant
ENABLE_MULTI_TENANT=false
DEFAULT_TENANT_ID=ampl-default
```

## Architecture

```
                    ┌──────────────────────────────────┐
                    │         DMS / External           │
                    │  (Payment, Delivery, Service)    │
                    └──────────┬───────────────────────┘
                               │ Webhooks
                               v
User ──> API Gateway ──> Auth Middleware (JWT / API Key / RBAC)
              │               │
              │          Rate Limiter ──> Prometheus Metrics (/metrics)
              │               │
              v               v
         ┌─────────── Orchestrator Pipeline (parallel) ────────────┐
         │                                                         │
         │  [1] Intent Classifier ──┐                              │
         │      (15 intents +       │                              │
         │       Hindi/Hinglish     │  asyncio.gather()            │
         │       keywords +         ├──────────────────────┐       │
         │       LLM fallback)      │                      │       │
         │                          │                      │       │
         │  [2] Entity Extractor ───┘                      │       │
         │      (30+ fields +                              │       │
         │       Hindi/Devanagari +                        │       │
         │       word-boundary)                            │       │
         │                                                 │       │
         │  [3] Query Preprocessor ──> Namespace Router    │       │
         │      (Hinglish expand,      (intent-based)      │       │
         │       filler removal)            │               │       │
         │                                  v               │       │
         │  [4] Embedding (+ LRU Cache) ──> Pinecone      │       │
         │      Token Estimator             (targeted NS)  │       │
         │                                  │               │       │
         │  [5] Query Decomposer ──────────>│               │       │
         │      (multi-facet fanout)        v               │       │
         │                          Reranker + Hybrid Search│       │
         │                                  │               │       │
         │                          Context Builder         │       │
         │                          (Jaccard dedup)         │       │
         │                                  │               │       │
         │  [6] Context Window Manager ─────┘               │       │
         │      (priority-based truncation)                 │       │
         │                                                  │       │
         │  [7] Flow Engine ──> Prompt Builder ──> LLM     │       │
         │      (guided conversations)              │       │       │
         │                                          v       │       │
         │  [8] Guardrails (response verification) ─┘       │       │
         │                                                  │       │
         │  [9] Cumulative Lead Scorer ──> Lead Router ──> CRM     │
         │      (60% avg + 40% peak)                               │
         │                                                         │
         │  [10] Handoff Manager ──> Agent Queue                   │
         │       (trigger detection)                               │
         └─────────────────────────────────────────────────────────┘
              │                                          │
              v                                          v
    ┌─────────────────┐                     ┌──────────────────┐
    │ PostgreSQL (async)│                     │  A/B Experiments │
    │ Conversations    │                     │  Variant assign  │
    │ Leads + Events   │                     │  Metric collect  │
    │ Messages         │                     └──────────────────┘
    │ Feedback         │
    └─────────────────┘
              │
              v
    ┌─────────────────────────────────┐
    │       Notification Router       │
    ├──────────┬──────────┬───────────┤
    │  Widget  │ WhatsApp │   Email   │
    │  (WS/SSE)│(Meta/Gup)│(SG/SES)  │
    └──────────┴──────────┴───────────┘
              │
              v
    ┌─────────────────────────────────┐
    │  Multi-Tenant Manager           │
    │  Knowledge Version Tracker      │
    │  Analytics Collector            │
    │  GDPR Compliance (export/erase) │
    │  Scheduled Messages             │
    └─────────────────────────────────┘
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
- [x] JWT authentication enabled (`JWT_SECRET` set) — implemented in `api/routes/auth.py`
- [x] Rate limiting configured (`RATE_LIMIT_REQUESTS`) — implemented in `api/middleware/rate_limit.py`
- [x] PostgreSQL database configured (`DATABASE_URL`) — implemented in `database/session.py`
- [ ] Alembic migrations applied (`alembic upgrade head`)
- [x] Monitoring — Prometheus metrics at `/metrics` — implemented in `api/middleware/metrics.py`
- [ ] Redis for caching and session management
- [ ] Task queue (Celery/APScheduler) for scheduled messages
- [x] WhatsApp Business API provider configured — implemented in `api/channels/whatsapp.py`
- [x] Email service (SendGrid/SES) configured — implemented in `api/channels/email.py`
- [x] GDPR compliance audit logging enabled — implemented in `api/routes/compliance.py`
- [x] LLM guardrails enabled for input/output filtering — implemented in `llm/guardrails.py`
- [ ] Auto-scaling configured (ECS/K8s HPA)

## License

Proprietary - AMPL Automotive
