# sir_vonwastaken
# AI-Powered Content Trend Intelligence & Brand Management Assistant

An autonomous AI assistant that monitors YouTube, Reddit, and Google Trends for emerging
content opportunities matching your creator style, ranks them, generates video ideas
(titles/hooks/outlines/scripts), and separately monitors your Gmail inbox for sponsorship
opportunities — drafting professional replies that always wait for your approval before
anything is sent.

Built with **FastAPI**, **MongoDB**, and **OpenAI** (chat + embeddings).

---

## Features

| # | Module | What it does |
|---|--------|---------------|
| 1 | Multi-Platform Data Collection | Pulls content from YouTube, Reddit, and Google Trends, plus emails from Gmail |
| 2 | Data Processing | Deduplicates, filters low-signal content, normalizes into one schema |
| 3 | Creator Profile | Learns your style from your own channel's video history |
| 4 | Similarity Analysis | Embeds content and compares it against your creator profile |
| 5 | Trend Scoring & Ranking | Combines growth velocity, engagement, freshness, similarity, and cross-platform presence into one score |
| 6 | AI Content Generation | Generates titles, hooks, outlines, and script drafts for top-ranked trends |
| 7 | Brand Deal Email Assistant | Detects sponsorship emails, summarizes them, and extracts key details |
| 8 | Draft Reply + Human Approval | Drafts professional replies as real Gmail drafts — nothing sends without your explicit approval |
| 9 | Notifications | Desktop (macOS), Discord, Telegram, and email alerts for high-value trends |
| 10 | MongoDB Integration | Single source of truth for raw content, processed content, profiles, trends, generated content, and emails |
| 11 | Logging & Configuration | Centralized `.env`-driven config and shared logger across every module |

---

## Project Structure

```
.
├── main.py                        # FastAPI app entrypoint
├── api/
│   └── routes.py                  # All API endpoints
├── config/
│   └── settings.py                # Central .env-driven configuration
├── utils/
│   ├── logger.py                  # Shared logger
│   └── llm_client.py              # OpenAI chat/embedding wrapper
├── database/
│   ├── mongodb.py                 # MongoDB connection + generic helpers
│   └── vector_store.py            # Embedding storage + cosine similarity search
├── data_collectors/
│   ├── youtube_collector.py       # YouTube Data API v3
│   ├── reddit_collector.py        # Reddit via PRAW
│   ├── google_trends_collector.py # Google Trends via pytrends
│   ├── gmail_collector.py         # Gmail API (OAuth2)
│   └── data_processor.py          # Dedup / filter / normalize raw content
├── creator_profile/
│   └── build_creator_profile.py   # Module 2: creator style learning
├── content_similarity_check/
│   ├── embedding_search.py        # Embeds processed content
│   ├── vector_search.py           # Similarity search against creator profile
│   └── similarity_engine.py       # Pairwise cosine similarity
├── AI_analysis/
│   └── content_analyzer.py        # Category / format / topic classification
├── trend_ranking/
│   └── ranking_engine.py          # Module 4: trend scoring
├── AI_generator/
│   ├── generate_content.py        # Module 5: titles/hooks/outlines/scripts
│   └── prompts.py
├── email_assistant/
│   ├── detect_sponsorship.py      # Module 7: sponsorship detection
│   ├── summarize_email.py         # Email summarization
│   ├── draft_replies.py           # AI-drafted Gmail replies
│   └── wait_for_approval.py       # Human approval workflow
└── notification_system/
    ├── desktop_notifications.py   # macOS native notifications
    ├── discord.py                 # Discord webhook
    ├── telegram.py                # Telegram Bot API
    └── email.py                   # SMTP email alerts
```

---

## Prerequisites

- Python 3.10+
- A MongoDB instance (local or [Atlas](https://www.mongodb.com/atlas))
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [YouTube Data API v3](https://console.cloud.google.com/apis/library/youtube.googleapis.com) key
- Reddit API credentials ([create an app here](https://www.reddit.com/prefs/apps))
- A Gmail OAuth2 client (Desktop app type) if you want the email assistant — see [Gmail Setup](#gmail-setup) below
- macOS, if you want native desktop notifications (other notification channels work cross-platform)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd sir_vonwastaken-main
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Then fill in `.env` with your real credentials. See [Environment Variables](#environment-variables) below for what each one does.

### 3. Gmail Setup

The email assistant needs a real OAuth2 client, not just an API key:

1. Go to [console.cloud.google.com](https://console.cloud.google.com) → create/select a project
2. **APIs & Services → Library** → enable the **Gmail API**
3. **APIs & Services → OAuth consent screen** → choose "External," fill in basic info, add your own Gmail as a test user
4. **APIs & Services → Credentials → Create Credentials → OAuth client ID** → type **Desktop app**
5. Download the resulting JSON, rename it to `credentials.json`, and place it in the project root (same folder as `main.py`)
6. In `.env`, set:
   ```dotenv
   GMAIL_CREDENTIALS_FILE=credentials.json
   GMAIL_TOKEN_FILE=token.json
   GMAIL_QUERY=newer_than:2d
   ```
7. `token.json` doesn't need to exist yet — it's created automatically the first time you call a Gmail endpoint (a browser window opens for you to log in and approve access; every run after that is silent).

### 4. Run the app

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: **http://localhost:8000/docs**

---

## Environment Variables

| Variable | Required for | Notes |
|---|---|---|
| `MONGODB_URI`, `MONGODB_DB_NAME` | Everything | Connection string + database name |
| `OPENAI_API_KEY` | AI features | Powers chat completions + embeddings |
| `OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL` | AI features | Defaults: `gpt-4o-mini`, `text-embedding-3-small` |
| `YOUTUBE_API_KEY` | YouTube collection | From Google Cloud Console |
| `YOUTUBE_CHANNEL_ID` | Creator profile | **Your own** channel — used to learn your style |
| `YOUTUBE_WATCH_CHANNELS` | YouTube collection | Comma-separated channel IDs to monitor (niche/competitor channels) |
| `YOUTUBE_WATCH_QUERIES` | YouTube collection | Comma-separated search terms to monitor |
| `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` | Reddit collection | From your Reddit app |
| `REDDIT_SUBREDDITS` | Reddit collection | Comma-separated subreddit names (no `r/`) |
| `GOOGLE_TRENDS_GEO` | Google Trends | Region code, e.g. `US` |
| `GOOGLE_TRENDS_KEYWORDS` | Google Trends | Comma-separated keywords to track |
| `GMAIL_CREDENTIALS_FILE`, `GMAIL_TOKEN_FILE`, `GMAIL_QUERY` | Email assistant | See [Gmail Setup](#gmail-setup) |
| `NOTIFY_DESKTOP_ENABLED` / `NOTIFY_DISCORD_ENABLED` / `NOTIFY_TELEGRAM_ENABLED` / `NOTIFY_EMAIL_ENABLED` | Notifications | Toggle each channel independently |
| `DISCORD_WEBHOOK_URL` | Discord notifications | Incoming webhook URL |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram notifications | From [@BotFather](https://t.me/BotFather) |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `NOTIFY_EMAIL_TO` | Email notifications | SMTP relay credentials (e.g. Gmail app password) |
| `TREND_NOTIFY_SCORE_THRESHOLD` | Trend ranking | Score (0–1) above which a notification fires |
| `TREND_WEIGHT_GROWTH`, `TREND_WEIGHT_ENGAGEMENT`, `TREND_WEIGHT_FRESHNESS`, `TREND_WEIGHT_SIMILARITY`, `TREND_WEIGHT_CROSS_PLATFORM` | Trend ranking | Tune what matters most in the final score |

Full template with placeholders: [`.env.example`](./.env.example).

---

## Typical Workflow

```bash
# 1. Build your creator profile (once, or whenever you want to refresh it)
curl -X POST "http://localhost:8000/api/creator-profile/YOUR_CHANNEL_ID/build"

# 2. Collect fresh content from all platforms
curl -X POST "http://localhost:8000/api/collect/all"

# 3. Clean, dedupe, and normalize what was collected
curl -X POST "http://localhost:8000/api/process/run"

# 4. Classify content (category / format / topics)
curl -X POST "http://localhost:8000/api/analysis/run"

# 5. Embed content so it can be compared to your creator profile
curl -X POST "http://localhost:8000/api/similarity/embed-pending"

# 6. Score and rank everything into trend candidates
curl -X POST "http://localhost:8000/api/trends/rank?channel_id=YOUR_CHANNEL_ID"

# 7. Generate content ideas for a top trend
curl -X POST "http://localhost:8000/api/content/generate/CONTENT_ID?channel_id=YOUR_CHANNEL_ID"
```

### Email assistant workflow

```bash
# Sync recent inbox messages
curl -X POST "http://localhost:8000/api/collect/gmail/sync"

# Detect sponsorship emails
curl -X POST "http://localhost:8000/api/emails/scan-sponsorships"

# Summarize one
curl -X POST "http://localhost:8000/api/emails/EMAIL_ID/summarize"

# Generate a draft reply (creates a real Gmail draft, does NOT send)
curl -X POST "http://localhost:8000/api/emails/EMAIL_ID/draft-reply"

# Review pending drafts
curl "http://localhost:8000/api/emails/drafts"

# Approve (sends the Gmail draft) or reject
curl -X POST "http://localhost:8000/api/emails/drafts/DRAFT_ID/approve"
curl -X POST "http://localhost:8000/api/emails/drafts/DRAFT_ID/reject"
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | MongoDB connectivity check |
| POST | `/api/collect/youtube` | Collect trending + watched YouTube content |
| POST | `/api/collect/youtube/search` | Ad-hoc YouTube search |
| POST | `/api/collect/reddit` | Collect from configured subreddits |
| POST | `/api/collect/google-trends` | Collect trending searches + interest data |
| POST | `/api/collect/gmail/sync` | Sync recent Gmail messages |
| POST | `/api/collect/all` | Run all collectors (YouTube, Reddit, Trends) |
| POST | `/api/process/run` | Normalize/dedupe/filter raw content |
| POST | `/api/creator-profile/{channel_id}/build` | Build creator profile from your channel |
| GET | `/api/creator-profile/{channel_id}` | Get stored profile summary |
| POST | `/api/similarity/embed-pending` | Embed newly processed content |
| GET | `/api/similarity/{channel_id}/matches` | Content most similar to your style |
| POST | `/api/analysis/run` | Run AI categorization on pending content |
| POST | `/api/trends/rank` | Score and rank all processed content |
| GET | `/api/trends` | List top-ranked trend candidates |
| POST | `/api/content/generate/{content_id}` | Generate titles/hooks/outline/script |
| POST | `/api/content/regenerate/{content_id}` | Regenerate a single field |
| GET | `/api/content/history/{channel_id}` | Past generated content |
| POST | `/api/emails/scan-sponsorships` | Classify inbox for sponsorship emails |
| GET | `/api/emails/sponsorships` | List detected sponsorship emails |
| POST | `/api/emails/{email_id}/summarize` | Summarize an email |
| POST | `/api/emails/{email_id}/draft-reply` | Generate a draft Gmail reply |
| GET | `/api/emails/drafts` | List drafts pending approval |
| POST | `/api/emails/drafts/{draft_id}/approve` | Approve and send a draft |
| POST | `/api/emails/drafts/{draft_id}/reject` | Reject a draft |
| POST | `/api/notify/test` | Send a test notification |
| GET | `/api/dashboard/{channel_id}` | Aggregated dashboard snapshot |

Full request/response schemas are available at `/docs` (Swagger UI) once the app is running.

---

## Notes & Design Decisions

- **Vector store**: content embeddings are stored in MongoDB and compared via brute-force cosine similarity (NumPy) rather than a dedicated vector DB like Qdrant. This keeps the stack to what's already in `requirements.txt` and is fine at this data scale; swapping in Qdrant later only requires changing `database/vector_store.py`.
- **Telegram notifications**: sent via a direct HTTP call to the Telegram Bot API rather than the `python-telegram-bot` library, since that library's modern API is async and not a natural fit for a single notification fired from a sync FastAPI request.
- **Nothing is ever auto-sent**: AI-drafted email replies are created as real Gmail drafts and only sent after you explicitly call the `/approve` endpoint.
- **Desktop notifications** only work on macOS (uses `osascript`), matching the local-Mac deployment model in the original proposal.

---

## Troubleshooting

| Problem | Likely cause |
|---|---|
| `OPENAI_API_KEY is not set` | Missing/empty value in `.env` |
| `MONGODB_URI is not set` | Missing/empty value in `.env`, or `.env` not loaded (check you're running from the project root) |
| Gmail browser login doesn't open | `credentials.json` missing or wrong path — check `GMAIL_CREDENTIALS_FILE` |
| `/api/trends/rank` returns an empty list | Run `/api/collect/all` then `/api/process/run` first — there's no processed content yet |
| Desktop notifications silently do nothing | You're not on macOS, or `NOTIFY_DESKTOP_ENABLED=false` |