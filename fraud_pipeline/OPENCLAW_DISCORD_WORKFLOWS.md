# OpenClaw Discord Workflows

This document tracks the upgraded Discord fraud workflows added on top of the existing Streamlit and ChatOps stack.

## Implemented

- Case-thread aware delivery for case-specific alerts, reminders, and review updates.
- Slash-style text workflows in Discord:
  - `/triage TX000275`
  - `/triage AC00454`
  - `/top-accounts`
  - `/pending-review`
  - `/merchant M026`
  - `/send-oof-brief`
  - `/why-flagged TX000275`
- Shared channel workspace memory in `outputs/chatops/discord_bot_state.json`:
  - uploaded CSV history
  - last goal
  - last discussed case
  - last command
  - analyst intent
- CSV upload follow-up memory for Discord uploads.
- OOF executive brief generation with a multi-role orchestration pattern:
  - risk summary agent
  - entity investigator agent
  - controls strategist agent
  - lead synthesis agent
- Deeper case explanations with row-level evidence and model contributors.
- Structured review-judge suggestions with:
  - suggested disposition
  - confidence
  - rationale
  - reviewer checks
- Per-case proactive reminders with escalation when a case stays untouched beyond the configured threshold.
- Optional real OpenClaw agent runtime synthesis for Discord replies when:
  - `OPENCLAW_USE_AGENT_RUNTIME=true`
  - the OpenClaw gateway is running
  - the default model is authenticated through `openai-codex`
- Image-analysis workflows for Discord attachments:
  - upload PNG/JPG/JPEG/WEBP and ask `analyze this`
  - `/analyze-image`
  - `/fraud-image-review`
  - `compare this to current flagged patterns`
  - `what should OOF do next based on this image?`
- Markdown and JSON image-review exports in `outputs/chatops/exports/image_reviews/`.

## State Files

- Discord workspace and case thread state:
  - `outputs/chatops/discord_bot_state.json`
- Alert dedupe state:
  - `outputs/chatops/alert_state.json`
- Published active fraud context:
  - `outputs/chatops/active_context/`
- Generated exported artifacts:
  - `outputs/chatops/exports/`
  - `outputs/chatops/exports/image_reviews/`
  - `outputs/chatops/uploads/images/`

## Useful Commands

Run the Discord bot:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 scripts/openclaw_discord_bot.py
```

Optional: start the real OpenClaw runtime first if you want Discord replies to be synthesized by `openclaw agent` instead of only the local fraud assistant:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
export OPENCLAW_USE_AGENT_RUNTIME=true
npx openclaw@latest --version
npx openclaw@latest health
npx openclaw@latest gateway run --allow-unconfigured --verbose
```

Important:

- `npx openclaw@latest` requires Node.js `v22.12+`.
- This Mac is currently on Node `v20.13.1`, so upgrade Node first or the bot will correctly fall back to the local grounded fraud assistant.

In another terminal:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
npx openclaw@latest models auth login --provider openai-codex --set-default
npx openclaw@latest models status
python3 scripts/openclaw_discord_bot.py
```

Run Streamlit separately:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 -m streamlit run fraud_pipeline/app/streamlit_app.py
```

Send a report manually:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 scripts/send_fraud_alerts.py --report-only
```

Send alerts manually:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 scripts/send_fraud_alerts.py --force
```

## Local Verification Commands

Syntax checks:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 -m py_compile \
  fraud_pipeline/scripts/openclaw_discord_bot.py \
  fraud_pipeline/src/chatops/openclaw_bridge.py \
  fraud_pipeline/src/chatops/openclaw_agent.py \
  fraud_pipeline/src/chatops/discord_state.py \
  fraud_pipeline/src/chatops/message_formatter.py \
  fraud_pipeline/src/chatops/query_service.py \
  fraud_pipeline/src/chatops/discord_upload_service.py \
  fraud_pipeline/src/ai_assistant.py \
  fraud_pipeline/src/review_judge.py \
  fraud_pipeline/src/tda_analysis.py \
  fraud_pipeline/src/anomaly_detection.py
```

Workflow smoke test without Discord:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source ../venv/bin/activate
OPENAI_API_KEY='' python3 - <<'PY'
from pathlib import Path
from src.chatops.query_service import run_command_workflow, answer_analyst_question, create_oof_brief
from src.chatops.context_loader import load_report_bundle

bundle = load_report_bundle()
for cmd in ['/top-accounts', '/pending-review', '/merchant M026', '/triage TX000275']:
    result = run_command_workflow(cmd, bundle=bundle)
    print(cmd, result.get('handled'), result.get('case_type'), result.get('case_id'))
    print(str(result.get('answer', ''))[:240])

brief_path = Path('outputs/chatops/exports/test-oof-brief.md')
brief = create_oof_brief(bundle=bundle, focus='OOF executive brief for limited resources', export_path=brief_path)
print('brief_path_exists', brief_path.exists(), brief_path)

q = answer_analyst_question('Why was TX000275 flagged?', bundle=bundle)
print(q.get('case_type'), q.get('case_id'))
print(str(q['answer'])[:240])
PY
```

Pipeline and artifact smoke test:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source ../venv/bin/activate
ENABLE_AI_REVIEW_JUDGE=false python3 run_pipeline.py --skip-streamlit
```

Image-analysis smoke test without Discord:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates
source venv/bin/activate
python3 scripts/test_image_chatops.py --all-samples --use-pipeline-outputs
```

Thread-routing dry run:

```bash
cd /Users/macbook/Hack/Deloitte-pi-rates/fraud_pipeline
source ../venv/bin/activate
python3 - <<'PY'
from src.chatops.discord_state import read_discord_state, write_discord_state, upsert_case_thread
from src.chatops.message_formatter import build_decision_update_message
from src.chatops.openclaw_bridge import deliver_message

state = read_discord_state()
upsert_case_thread(
    state,
    case_type='transaction',
    case_id='TX000275',
    thread_id='1234567890',
    channel_id='1482399372724539568',
    thread_name='Transaction TX000275',
)
write_discord_state(state)

message = build_decision_update_message(
    case_summary={
        'transactionid': 'TX000275',
        'accountid': 'AC00454',
        'merchantid': 'M074',
        'location': 'Kansas City',
        'channel': 'ATM',
        'composite_risk_score': 1.0,
    },
    decision='Needs Review',
    notes='thread routing smoke test',
    source_label='Pipeline outputs',
)

result = deliver_message(message, dry_run=True)
print(result.delivery_error, bool(result.payload_preview))
PY
```

## Demo Prompts For Discord

- `/triage TX000275`
- `/triage AC00454`
- `/merchant M026`
- `/pending-review`
- `/top-accounts`
- `/send-oof-brief`
- `Why was TX000899 flagged?`
- `What disposition would you suggest for TX000275?`
- `What should OOF prioritize if resources are limited?`
- `Give me the top control gaps in the current fraud context.`
- Upload `OOF_phishing_email_alert.png` with `analyze this`
- Upload `Payment_receipt_invoice.png` with `is this suspicious?`
- Upload `Transaction-table.png` with `compare this to current flagged patterns`
