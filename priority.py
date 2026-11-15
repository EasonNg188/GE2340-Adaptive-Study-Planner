"""Rule-based knowledge, priority, and study-time calculations."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from config import (
    DEADLINE_HORIZON_DAYS,
    DEVELOPING,
    DEVELOPING_MAX_SCORE,
    DEVELOPING_MULTIPLIER,
    MASTERED,
    MASTERED_MULTIPLIER,
    MAX_QUIZ_SCORE,
    MIN_QUIZ_SCORE,
    PRIORITY_MAX,
    PRIORITY_MIN,
    PRIORITY_TASK_WEIGHT,
    PRIORITY_URGENCY_WEIGHT,
    PRIORITY_WEAKNESS_WEIGHT,
    ROUNDING_UNIT_MINUTES,
    MIN_SESSION_MINUTES,
    WEAK,
    WEAK_MAX_SCORE,
    WEAK_MULTIPLIER,
)
from models import Topic


def _validate_score(score: int) -> None:
    if isinstance(score, bool) or not isinstance(score, int):
        raise ValueError("Quiz score must be an integer.")
    if not MIN_QUIZ_SCORE <= score <= MAX_QUIZ_SCORE:
        raise ValueError("Quiz score must be between 0 and 100.")


def knowledge_category(score: int) -> str:
    _validate_score(score)
    if score <= WEAK_MAX_SCORE:
        return WEAK
    if score <= DEVELOPING_MAX_SCORE:
        return DEVELOPING
    return MASTERED


def study_multiplier(score: int) -> float:
    category = knowledge_category(score)
    if category == WEAK:
        return WEAK_MULTIPLIER
    if category == DEVELOPING:
        return DEVELOPING_MULTIPLIER
    return MASTERED_MULTIPLIER


def days_to_deadline(deadline: date, today: date) -> int:
    return (deadline - today).days


def urgency_for_days(days: int) -> float:
    if days <= 0:
        return 1.0
    return 1.0 - min(days, DEADLINE_HORIZON_DAYS) / DEADLINE_HORIZON_DAYS


def weakness_for_score(score: int) -> float:
    _validate_score(score)
    return (MAX_QUIZ_SCORE - score) / MAX_QUIZ_SCORE


def calculate_priority(topic: Topic, today: date) -> float:
    weakness = weakness_for_score(topic.quiz_score)
    urgency = urgency_for_days(days_to_deadline(topic.deadline, today))
    normalized_task_weight = topic.task_weight / 100.0
    value = 100.0 * (
        PRIORITY_WEAKNESS_WEIGHT * weakness
        + PRIORITY_URGENCY_WEIGHT * urgency
        + PRIORITY_TASK_WEIGHT * normalized_task_weight
    )
    return max(PRIORITY_MIN, min(PRIORITY_MAX, value))


def priority_breakdown(topic: Topic, today: date) -> dict[str, float | int]:
    days = days_to_deadline(topic.deadline, today)
    return {
        "weakness": weakness_for_score(topic.quiz_score),
        "days_to_deadline": days,
        "urgency": urgency_for_days(days),
        "normalized_task_weight": topic.task_weight / 100.0,
        "priority": calculate_priority(topic, today),
    }


def round_priority(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    )


def format_priority(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{round_priority(value):.2f}"


def round_minutes_to_30(value: float | Decimal | int) -> int:
    """Round minutes to the nearest 30, with exact halfway values rounded up."""
    decimal_value = Decimal(str(value))
    if decimal_value <= 0:
        return 0
    blocks = (decimal_value / Decimal(ROUNDING_UNIT_MINUTES)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return max(MIN_SESSION_MINUTES, int(blocks) * ROUNDING_UNIT_MINUTES)


def adjusted_study_minutes(estimated_hours: float, quiz_score: int) -> int:
    if estimated_hours <= 0:
        return 0
    raw_minutes = (
        Decimal(str(estimated_hours)) * Decimal(60) * Decimal(str(study_multiplier(quiz_score)))
    )
    return round_minutes_to_30(raw_minutes)


def session_durations(total_minutes: int) -> list[int]:
    """Split required minutes into 60-minute sessions plus a 30-minute remainder."""
    if total_minutes <= 0:
        return []
    normalized = round_minutes_to_30(total_minutes)
    full_sessions, remainder = divmod(normalized, 60)
    durations = [60] * full_sessions
    if remainder:
        durations.append(remainder)
    return durations
