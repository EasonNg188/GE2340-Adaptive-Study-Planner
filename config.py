"""Central configuration for the adaptive study planner."""

from pathlib import Path

APP_TITLE = "Adaptive Study Planner"
APP_ICON = "📚"

PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LLM_PROMPT_PATH = PROMPTS_DIR / "llm_scheduler_prompt.txt"
SAMPLE_OUTPUT_DIR = PROJECT_ROOT / "sample_outputs"
LLM_RUN_DIR = SAMPLE_OUTPUT_DIR / "llm_runs"

# Knowledge levels
WEAK = "Weak"
DEVELOPING = "Developing"
MASTERED = "Mastered"

WEAK_MAX_SCORE = 59
DEVELOPING_MAX_SCORE = 79
MIN_QUIZ_SCORE = 0
MAX_QUIZ_SCORE = 100

WEAK_MULTIPLIER = 1.5
DEVELOPING_MULTIPLIER = 1.0
MASTERED_MULTIPLIER = 0.5

# Priority formula
PRIORITY_WEAKNESS_WEIGHT = 0.5
PRIORITY_URGENCY_WEIGHT = 0.3
PRIORITY_TASK_WEIGHT = 0.2
DEADLINE_HORIZON_DAYS = 30
PRIORITY_MIN = 0.0
PRIORITY_MAX = 100.0

# Sessions
MIN_SESSION_MINUTES = 30
DEFAULT_SESSION_MINUTES = 60
MAX_SESSION_MINUTES = 90
ROUNDING_UNIT_MINUTES = 30

# CSV schemas
TOPIC_COLUMNS = [
    "course_name",
    "topic_name",
    "deadline",
    "estimated_hours",
    "quiz_score",
    "task_weight",
]
AVAILABILITY_COLUMNS = [
    "course_name",
    "topic_name",
    "available_date",
    "start_time",
    "duration_minutes",
]

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"

# Local Ollama baseline
OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
OLLAMA_BASE_URL_ENV = "OLLAMA_BASE_URL"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_OLLAMA_TEMPERATURE = 0.0
