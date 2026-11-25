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
