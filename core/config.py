"""Central place for app-wide settings and limits."""

# Old-style .xls would need an extra library (xlrd); users can re-save as .xlsx.
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
MAX_FILE_MB = 25
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024

# Guard rails so a huge sheet can't freeze the app
MAX_ROWS = 500_000
MAX_COLUMNS = 300

# A text column is converted to numbers/dates only if at least this share
# of its non-empty values convert cleanly.
CONVERSION_THRESHOLD = 0.95

# Protects your OpenAI credit: max AI requests per browser session (refresh to reset).
MAX_AI_REQUESTS_PER_SESSION = 10

# Protects your OpenAI credit on a public deployment: max AI requests per day (UTC) across
# ALL visitors of one running app. Override with AI_DAILY_LIMIT in .env or Streamlit secrets.
# Results already generated for the same file are shared and don't count (see ui/state.py).
AI_DAILY_LIMIT = 2

# Cost controls for every OpenAI request (gpt-5-mini bills reasoning tokens as output tokens).
AI_REASONING_EFFORT = "low"          # less hidden reasoning = cheaper and faster
AI_MAX_INPUT_CHARS = 40_000          # ~11k tokens; larger requests are refused before sending
SUMMARY_MAX_OUTPUT_TOKENS = 4_000    # a full summary is ~1.3k visible tokens
QA_MAX_OUTPUT_TOKENS = 2_000         # a plan or an answer is ~300 visible tokens
