"""Minimal local-Ollama scheduling baseline with strict validation and logging."""

from __future__ import annotations

import json
import os
import re
import time as time_module
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TEMPERATURE,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    LLM_PROMPT_PATH,
    LLM_RUN_DIR,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MODEL_ENV,
    PROJECT_ROOT,
)
from models import AvailabilityWindow, StudySession, Topic, TopicKey, topic_key
from validation import format_errors, validate_dataset

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")
DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")


class LLMError(RuntimeError):
    """Base class for LLM baseline errors."""


class LLMInputError(LLMError):
    """Raised when planner input is invalid before any Ollama request."""


class LLMConfigurationError(LLMError):
    """Raised when required LLM configuration is missing."""


class OllamaUnavailableError(LLMError):
    """Raised when the local Ollama service cannot be reached or rejects a request."""


class OllamaProtocolError(LLMError):
    """Raised when Ollama returns a response without usable model content."""


@dataclass(frozen=True)
class OllamaConfig:
    model: str
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    temperature: float = DEFAULT_OLLAMA_TEMPERATURE
    timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls) -> "OllamaConfig":
        model = os.environ.get(OLLAMA_MODEL_ENV, "").strip()
        if not model:
            raise LLMConfigurationError(
                "OLLAMA_MODEL is not set. Set it to an installed local Ollama model. "
                "No model was selected automatically."
            )
        base_url = os.environ.get(OLLAMA_BASE_URL_ENV, "").strip()
        if not base_url:
            base_url = DEFAULT_OLLAMA_BASE_URL
        if not base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError(
                "OLLAMA_BASE_URL must start with http:// or https://."
            )
        return cls(model=model, base_url=base_url.rstrip("/"))

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def model_parameters(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "format": "json",
            "stream": False,
            "timeout_seconds": self.timeout_seconds,
            "endpoint": "/api/chat",
        }


def _llm_round_minutes_to_30(raw_minutes: Decimal) -> int:
    """Round raw minutes half-up to the nearest 30 without a minimum block."""
    if raw_minutes <= 0:
        return 0
    blocks = (raw_minutes / Decimal(30)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    return int(blocks) * 30


def llm_required_minutes(estimated_hours: float | int | Decimal) -> int:
    """Return original estimated minutes rounded half-up to the nearest 30."""
    raw_minutes = Decimal(str(estimated_hours)) * Decimal(60)
    return _llm_round_minutes_to_30(raw_minutes)


def _ceil_to_next_minute(value: datetime) -> datetime:
    floored = value.replace(second=0, microsecond=0)
    if value > floored:
        floored += timedelta(minutes=1)
    return floored


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_negative_int(value: Any) -> int | None:
    if not _is_int(value) or value < 0:
        return None
    return value


def _positive_int(value: Any) -> int | None:
    if not _is_int(value) or value <= 0:
        return None
    return value


def _valid_duration(value: Any) -> bool:
    return _is_int(value) and 30 <= value <= 90


@dataclass(frozen=True)
class TopicAccounting:
    course: str
    topic: str
    required_minutes: int
    completed_minutes: int
    remaining_required_minutes: int
    scheduled_minutes: int
    reported_unscheduled_minutes: int
    values_valid: bool = True

    @property
    def accounted_minutes(self) -> int:
        return self.scheduled_minutes + self.reported_unscheduled_minutes

    @property
    def is_complete(self) -> bool:
        return (
            self.values_valid
            and self.accounted_minutes == self.remaining_required_minutes
        )

    @property
    def accounting_status(self) -> str:
        if not self.values_valid:
            return "Invalid accounting input"
        if self.is_complete:
            return "Complete"
        if self.accounted_minutes < self.remaining_required_minutes:
            return "Under-accounted"
        return "Over-accounted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "course": self.course,
            "topic": self.topic,
            "required_minutes": self.required_minutes,
            "completed_minutes": self.completed_minutes,
            "remaining_required_minutes": self.remaining_required_minutes,
            "scheduled_minutes": self.scheduled_minutes,
            "reported_unscheduled_minutes": self.reported_unscheduled_minutes,
            "accounted_minutes": self.accounted_minutes,
            "accounting_status": self.accounting_status,
            "is_complete": self.is_complete,
            "values_valid": self.values_valid,
        }


@dataclass(frozen=True)
class AccountingSummary:
    topics: list[TopicAccounting]

    @property
    def total_required_minutes(self) -> int:
        return sum(row.required_minutes for row in self.topics)

    @property
    def total_completed_minutes(self) -> int:
        return sum(row.completed_minutes for row in self.topics)

    @property
    def total_remaining_required_minutes(self) -> int:
        return sum(row.remaining_required_minutes for row in self.topics)

    @property
    def total_scheduled_minutes(self) -> int:
        return sum(row.scheduled_minutes for row in self.topics)

    @property
    def total_reported_unscheduled_minutes(self) -> int:
        return sum(row.reported_unscheduled_minutes for row in self.topics)

    @property
    def total_accounted_minutes(self) -> int:
        return sum(row.accounted_minutes for row in self.topics)

    @property
    def has_invalid_values(self) -> bool:
        return any(not row.values_valid for row in self.topics)

    @property
    def coverage_percentage(self) -> float | None:
        if not self.topics or self.has_invalid_values:
            return None
        remaining = self.total_remaining_required_minutes
        accounted = self.total_accounted_minutes
        if remaining == 0:
            return 100.0 if accounted == 0 else 0.0
        percentage = Decimal(accounted) / Decimal(remaining) * Decimal(100)
        return float(percentage.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    @property
    def is_complete(self) -> bool:
        return bool(self.topics) and all(row.is_complete for row in self.topics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_minutes": self.total_required_minutes,
            "completed_minutes": self.total_completed_minutes,
            "required_remaining_minutes": self.total_remaining_required_minutes,
            "scheduled_minutes": self.total_scheduled_minutes,
            "reported_unscheduled_minutes": self.total_reported_unscheduled_minutes,
            "accounted_minutes": self.total_accounted_minutes,
            "coverage_percentage": self.coverage_percentage,
            "is_complete": self.is_complete,
            "has_invalid_values": self.has_invalid_values,
            "topics": [row.to_dict() for row in self.topics],
        }


def _structured_topic_map(
    structured_input: Mapping[str, Any],
) -> dict[TopicKey, Mapping[str, Any]]:
    topic_map: dict[TopicKey, Mapping[str, Any]] = {}
    if not isinstance(structured_input, dict):
        return topic_map
    topics_payload = structured_input.get("topics")
    if not isinstance(topics_payload, list):
        return topic_map
    for topic in topics_payload:
        if not isinstance(topic, dict):
            continue
        course = topic.get("course")
        name = topic.get("topic")
        if isinstance(course, str) and isinstance(name, str):
            topic_map[topic_key(course, name)] = topic
    return topic_map


def _entry_topic_key(entry: Mapping[str, Any]) -> TopicKey | None:
    course = entry.get("course")
    name = entry.get("topic")
    if not isinstance(course, str) or not isinstance(name, str):
        return None
    return topic_key(course, name)


def _accounting_scheduled_minutes(value: Any) -> int | None:
    if not _is_int(value) or value < 0:
        return None
    return value


def build_accounting_summary(
    structured_input: Mapping[str, Any],
    schedule: Mapping[str, Any] | None,
) -> AccountingSummary:
    """Build a diagnostic accounting summary without modifying the LLM output."""
    topic_map = _structured_topic_map(structured_input)
    scheduled_by_topic: dict[TopicKey, int] = {}
    reported_by_topic: dict[TopicKey, int] = {}
    invalid_topics: set[TopicKey] = set()

    if isinstance(schedule, dict):
        sessions = schedule.get("sessions")
        if isinstance(sessions, list):
            for entry in sessions:
                if not isinstance(entry, dict):
                    continue
                key = _entry_topic_key(entry)
                if key is None or key not in topic_map:
                    continue
                duration = entry.get("duration_minutes")
                minutes = _accounting_scheduled_minutes(duration)
                if minutes is None:
                    invalid_topics.add(key)
                    continue
                scheduled_by_topic[key] = scheduled_by_topic.get(key, 0) + minutes
                if not _valid_duration(duration):
                    invalid_topics.add(key)

        unscheduled = schedule.get("unscheduled_topics")
        if isinstance(unscheduled, list):
            seen: set[TopicKey] = set()
            for entry in unscheduled:
                if not isinstance(entry, dict):
                    continue
                key = _entry_topic_key(entry)
                if key is None or key not in topic_map:
                    continue
                minutes = _positive_int(entry.get("unscheduled_minutes"))
                if key in seen:
                    invalid_topics.add(key)
                    continue
                seen.add(key)
                if minutes is None:
                    invalid_topics.add(key)
                    continue
                reported_by_topic[key] = minutes

    rows: list[TopicAccounting] = []
    for key, topic in topic_map.items():
        required = _non_negative_int(topic.get("required_minutes"))
        completed = _non_negative_int(topic.get("completed_minutes"))
        remaining = _non_negative_int(topic.get("remaining_required_minutes"))
        values_valid = (
            required is not None
            and completed is not None
            and remaining is not None
            and remaining == max(required - completed, 0)
            and key not in invalid_topics
        )
        rows.append(
            TopicAccounting(
                course=str(topic.get("course")),
                topic=str(topic.get("topic")),
                required_minutes=required or 0,
                completed_minutes=completed or 0,
                remaining_required_minutes=remaining or 0,
                scheduled_minutes=scheduled_by_topic.get(key, 0),
                reported_unscheduled_minutes=reported_by_topic.get(key, 0),
                values_valid=values_valid,
            )
        )
    return AccountingSummary(topics=rows)


@dataclass
class LLMRunResult:
    run_id: str
    timestamp: str
    model_name: str | None
    base_url: str | None
    model_parameters: dict[str, Any]
    prompt: str | None
    repair_prompt: str | None
    structured_input: dict[str, Any]
    raw_initial_response: str | None = None
    raw_repair_response: str | None = None
    parsed_schedule: dict[str, Any] | None = None
    accounting_summary: AccountingSummary | None = None
    validation_errors_initial: list[str] = field(default_factory=list)
    validation_errors_final: list[str] = field(default_factory=list)
    repair_attempted: bool = False
    runtime_seconds: float = 0.0
    status: str = "failure"
    error: str | None = None
    log_path: str | None = None
    logging_error: str | None = None

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def validation_status(self) -> str:
        if self.success:
            return "valid"
        if self.raw_initial_response is None and self.raw_repair_response is None:
            return "not_available"
        return "invalid"

    @property
    def validation_errors(self) -> list[str]:
        if self.status == "success":
            return []
        return self.validation_errors_final or self.validation_errors_initial

    @property
    def raw_model_response(self) -> str | None:
        return self.raw_repair_response or self.raw_initial_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "model_parameters": self.model_parameters,
            "prompt": self.prompt,
            "repair_prompt": self.repair_prompt,
            "structured_input": self.structured_input,
            "raw_initial_response": self.raw_initial_response,
            "raw_repair_response": self.raw_repair_response,
            "raw_model_response": self.raw_model_response,
            "parsed_schedule": self.parsed_schedule,
            "accounting_summary": (
                self.accounting_summary.to_dict()
                if self.accounting_summary is not None
                else None
            ),
            "validation_errors_initial": self.validation_errors_initial,
            "validation_errors_final": self.validation_errors_final,
            "validation_errors": self.validation_errors,
            "validation_status": self.validation_status,
            "repair_attempted": self.repair_attempted,
            "runtime_seconds": self.runtime_seconds,
            "status": self.status,
            "success": self.success,
            "error": self.error,
            "log_path": self.log_path,
            "logging_error": self.logging_error,
        }


def build_structured_input(
    topics: Sequence[Topic],
    availability: Sequence[AvailabilityWindow],
    completed_sessions: Sequence[StudySession],
    current_date: date,
    current_datetime: datetime | None = None,
) -> dict[str, Any]:
    if current_datetime is None:
        current_datetime = datetime.combine(current_date, time.min)
    current_datetime = _ceil_to_next_minute(current_datetime)

    availability_by_topic: dict[TopicKey, list[AvailabilityWindow]] = {}
    for window in availability:
        availability_by_topic.setdefault(window.key, []).append(window)

    completed_by_topic: dict[TopicKey, int] = {}
    for session in completed_sessions:
        if session.completed:
            completed_by_topic[session.key] = (
                completed_by_topic.get(session.key, 0) + session.duration_minutes
            )

    topic_payload: list[dict[str, Any]] = []
    for topic in topics:
        windows = sorted(
            availability_by_topic.get(topic.key, []),
            key=lambda item: (item.available_date, item.start_time),
        )
        required_minutes = llm_required_minutes(topic.estimated_hours)
        completed_minutes = completed_by_topic.get(topic.key, 0)
        topic_payload.append(
            {
                "course": topic.course_name,
                "topic": topic.topic_name,
                "deadline": topic.deadline.isoformat(),
                "estimated_study_hours": topic.estimated_hours,
                "quiz_score": topic.quiz_score,
                "task_weight": topic.task_weight,
                "required_minutes": required_minutes,
                "completed_minutes": completed_minutes,
                "remaining_required_minutes": max(
                    required_minutes - completed_minutes, 0
                ),
                "availability": [
                    {
                        "available_date": window.available_date.isoformat(),
                        "start_time": window.start_time.strftime("%H:%M"),
                        "duration_minutes": window.duration_minutes,
                    }
                    for window in windows
                ],
            }
        )

    completed_payload = [
        {
            "course": session.course_name,
            "topic": session.topic_name,
            "date": session.session_date.isoformat(),
            "start_time": session.start_time.strftime("%H:%M"),
            "duration_minutes": session.duration_minutes,
            "status": "completed",
        }
        for session in sorted(
            (session for session in completed_sessions if session.completed),
            key=lambda item: (item.session_date, item.start_time),
        )
    ]

    return {
        "current_date": current_datetime.date().isoformat(),
        "current_datetime": current_datetime.isoformat(timespec="minutes"),
        "topics": topic_payload,
        "completed_sessions": completed_payload,
    }


def load_prompt_template(path: str | Path = LLM_PROMPT_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def render_initial_prompt(template: str, structured_input: Mapping[str, Any]) -> str:
    marker = "{{INPUT_JSON}}"
    if marker not in template:
        raise LLMConfigurationError(
            f"Prompt template is missing the {marker} placeholder."
        )
    payload = json.dumps(structured_input, indent=2, ensure_ascii=False)
    return template.replace(marker, payload)


def build_repair_prompt(
    initial_prompt: str,
    raw_response: str,
    validation_errors: Sequence[str],
) -> str:
    numbered_errors = "\n".join(
        f"{index}. {error}" for index, error in enumerate(validation_errors, start=1)
    )
    return (
        "Your previous response was rejected by strict schedule validation.\n\n"
        "Validation errors:\n"
        f"{numbered_errors}\n\n"
        "Original scheduling request:\n"
        f"{initial_prompt}\n\n"
        "Rejected response:\n"
        f"{raw_response}\n\n"
        "Return only corrected JSON using the required schema. Do not use markdown "
        "fences, prose, comments, or omit required fields. Do not introduce unknown "
        "courses, topics, or unavailable time slots. For every topic, "
        "generated_scheduled_minutes + unscheduled_minutes must equal "
        "remaining_required_minutes exactly."
    )


def call_ollama(config: OllamaConfig, prompt: str) -> str:
    body = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": config.temperature},
    }
    try:
        response = requests.post(
            config.endpoint,
            json=body,
            timeout=config.timeout_seconds,
        )
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {config.base_url}. "
            "Start Ollama and verify OLLAMA_BASE_URL. "
            f"Details: {exc}"
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:500].strip()
        raise OllamaUnavailableError(
            f"Ollama returned HTTP {response.status_code} for model "
            f"{config.model!r} at {config.base_url}."
            + (f" Response: {detail}" if detail else "")
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise OllamaProtocolError("Ollama returned a response that was not JSON.") from exc
    try:
        content = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise OllamaProtocolError(
            "Ollama response did not contain message.content."
        ) from exc
    if not isinstance(content, str):
        raise OllamaProtocolError("Ollama message content was not a string.")
    return content


def parse_llm_json(raw_response: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return None, [f"Response is not valid JSON: {exc.msg} at line {exc.lineno}."]
    if not isinstance(parsed, dict):
        return None, ["Response JSON must be an object."]
    return parsed, []


def _strict_date(value: Any) -> date | None:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _strict_time(value: Any) -> time | None:
    if not isinstance(value, str) or not TIME_RE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError:
        return None


def _strict_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not DATETIME_RE.fullmatch(value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError:
        return None


def validate_llm_schedule(
    schedule: Any, structured_input: Mapping[str, Any]
) -> list[str]:
    """Return all constraint violations without modifying the LLM schedule."""
    errors: list[str] = []
    if not isinstance(schedule, dict):
        return ["Response JSON must be an object."]

    sessions = schedule.get("sessions")
    if not isinstance(sessions, list):
        errors.append("'sessions' must be a list.")
        sessions = []

    unscheduled = schedule.get("unscheduled_topics")
    if not isinstance(unscheduled, list):
        errors.append("'unscheduled_topics' must be a list.")
        unscheduled = []

    topics_payload = structured_input.get("topics")
    if not isinstance(topics_payload, list):
        return errors + ["Structured input does not contain a topics list."]
    if not topics_payload:
        return errors + ["Structured input does not contain any topics."]

    topic_map = _structured_topic_map(structured_input)
    if not topic_map:
        return errors + ["Structured input does not contain any valid topics."]
    remaining_by_topic: dict[TopicKey, int] = {}
    for key, topic in topic_map.items():
        required = _non_negative_int(topic.get("required_minutes"))
        completed = _non_negative_int(topic.get("completed_minutes"))
        remaining = _non_negative_int(topic.get("remaining_required_minutes"))
        if required is None or completed is None or remaining is None:
            errors.append(
                f"Structured input for {key[0]} / {key[1]} is missing valid "
                "required_minutes, completed_minutes, or remaining_required_minutes."
            )
            continue
        expected_remaining = max(required - completed, 0)
        if remaining != expected_remaining:
            errors.append(
                f"Structured input for {key[0]} / {key[1]} has inconsistent "
                f"workload fields: remaining_required_minutes is {remaining}, "
                f"expected {expected_remaining}."
            )
            continue
        remaining_by_topic[key] = remaining

    current_date_value = structured_input.get("current_date")
    current_date = _strict_date(current_date_value)
    current_date_valid = current_date is not None
    if not current_date_valid:
        errors.append("Structured input current_date is invalid.")
        current_date = date.today()

    current_datetime_value = structured_input.get("current_datetime")
    current_datetime = _strict_datetime(current_datetime_value)
    if current_datetime_value is not None and current_datetime is None:
        errors.append("Structured input current_datetime is invalid.")
    if current_datetime is None:
        current_datetime = datetime.combine(current_date, time.min)
    elif current_date_valid and current_datetime.date() != current_date:
        errors.append(
            "Structured input current_datetime date does not match current_date."
        )

    completed_intervals: list[tuple[datetime, datetime, str]] = []
    completed_payload = structured_input.get("completed_sessions", [])
    if isinstance(completed_payload, list):
        for entry in completed_payload:
            if not isinstance(entry, dict):
                continue
            session_date = _strict_date(entry.get("date"))
            start_time = _strict_time(entry.get("start_time"))
            duration = entry.get("duration_minutes")
            if session_date is None or start_time is None or not _is_int(duration):
                continue
            start_dt = datetime.combine(session_date, start_time)
            end_dt = start_dt + timedelta(minutes=duration)
            completed_intervals.append(
                (
                    start_dt,
                    end_dt,
                    f"completed session for {entry.get('course')} / {entry.get('topic')}",
                )
            )

    window_usage: dict[tuple[TopicKey, int], int] = {}
    topic_usage: dict[TopicKey, int] = {}
    scheduled_intervals: list[tuple[datetime, datetime, str]] = []

    for index, entry in enumerate(sessions, start=1):
        label = f"Session {index}"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object.")
            continue

        missing = [
            field_name
            for field_name in (
                "course",
                "topic",
                "date",
                "start_time",
                "duration_minutes",
                "reason",
            )
            if field_name not in entry
        ]
        if missing:
            errors.append(f"{label} is missing fields: {', '.join(missing)}.")
            continue

        course = entry.get("course")
        topic_name = entry.get("topic")
        if not isinstance(course, str) or not course.strip():
            errors.append(f"{label} has an invalid course.")
            continue
        if not isinstance(topic_name, str) or not topic_name.strip():
            errors.append(f"{label} has an invalid topic.")
            continue
        key = topic_key(course, topic_name)
        topic_info = topic_map.get(key)
        if topic_info is None:
            errors.append(
                f"{label} introduces unknown course/topic: {course} / {topic_name}."
            )
        elif remaining_by_topic.get(key) == 0:
            errors.append(
                f"{label} schedules {course} / {topic_name}, but its remaining "
                "requirement is 0 minutes."
            )

        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label} must include a non-empty reason.")

        session_date = _strict_date(entry.get("date"))
        if session_date is None:
            errors.append(f"{label} has invalid date {entry.get('date')!r}; expected YYYY-MM-DD.")
        start_time = _strict_time(entry.get("start_time"))
        if start_time is None:
            errors.append(
                f"{label} has invalid start_time {entry.get('start_time')!r}; expected HH:MM."
            )
        duration = entry.get("duration_minutes")
        if not _valid_duration(duration):
            errors.append(f"{label} duration must be an integer from 30 to 90 minutes.")

        if session_date is None or start_time is None or not _valid_duration(duration):
            continue

        start_dt = datetime.combine(session_date, start_time)
        end_dt = start_dt + timedelta(minutes=duration)
        if start_dt < current_datetime:
            errors.append(
                f"{label} is scheduled before the current date/time "
                f"({current_datetime.isoformat(timespec='minutes')})."
            )

        if topic_info is None:
            scheduled_intervals.append((start_dt, end_dt, label))
            continue

        deadline = _strict_date(topic_info.get("deadline"))
        if deadline is not None:
            deadline_exclusive = datetime.combine(
                deadline + timedelta(days=1), time.min
            )
            if end_dt > deadline_exclusive:
                errors.append(
                    f"{label} is after the deadline for {course} / {topic_name} "
                    f"({deadline.isoformat()})."
                )

        availability = topic_info.get("availability")
        if not isinstance(availability, list):
            availability = []
        matched_window: tuple[int, Mapping[str, Any]] | None = None
        for window_index, window in enumerate(availability):
            if not isinstance(window, dict):
                continue
            window_date = _strict_date(window.get("available_date"))
            window_start_time = _strict_time(window.get("start_time"))
            window_duration = window.get("duration_minutes")
            if (
                window_date is None
                or window_start_time is None
                or not _is_int(window_duration)
                or window_duration <= 0
            ):
                continue
            window_start = datetime.combine(window_date, window_start_time)
            window_end = window_start + timedelta(minutes=window_duration)
            if start_dt >= window_start and end_dt <= window_end:
                matched_window = (window_index, window)
                break

        if matched_window is None:
            errors.append(
                f"{label} does not fit inside any available time slot for "
                f"{course} / {topic_name}."
            )
        else:
            window_index, _ = matched_window
            usage_key = (key, window_index)
            window_usage[usage_key] = window_usage.get(usage_key, 0) + duration
            topic_usage[key] = topic_usage.get(key, 0) + duration

        for other_start, other_end, other_label in (
            completed_intervals + scheduled_intervals
        ):
            if start_dt < other_end and other_start < end_dt:
                errors.append(f"{label} overlaps {other_label}.")
        scheduled_intervals.append((start_dt, end_dt, label))

    for (key, window_index), used in window_usage.items():
        topic_info = topic_map.get(key)
        if topic_info is None:
            continue
        availability = topic_info.get("availability", [])
        if not isinstance(availability, list) or window_index >= len(availability):
            continue
        window = availability[window_index]
        duration = window.get("duration_minutes") if isinstance(window, dict) else None
        if _is_int(duration) and used > duration:
            errors.append(
                f"Scheduled time for {key[0]} / {key[1]} on "
                f"{window.get('available_date')} exceeds the available duration."
            )

    for key, used in topic_usage.items():
        topic_info = topic_map.get(key)
        if topic_info is None:
            continue
        availability = topic_info.get("availability", [])
        if not isinstance(availability, list):
            availability = []
        total_available = sum(
            window.get("duration_minutes", 0)
            for window in availability
            if isinstance(window, dict) and _is_int(window.get("duration_minutes"))
        )
        if used > total_available:
            errors.append(
                f"Total scheduled time for {key[0]} / {key[1]} exceeds its total "
                "available duration."
            )

    unscheduled_seen: set[TopicKey] = set()
    for index, entry in enumerate(unscheduled, start=1):
        label = f"Unscheduled topic {index}"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object.")
            continue
        course = entry.get("course")
        topic_name = entry.get("topic")
        key: TopicKey | None = None
        if not isinstance(course, str) or not isinstance(topic_name, str):
            errors.append(f"{label} must include course and topic strings.")
        else:
            key = topic_key(course, topic_name)
            if key not in topic_map:
                errors.append(
                    f"{label} introduces unknown course/topic: {course} / {topic_name}."
                )
                key = None
            elif key in unscheduled_seen:
                errors.append(
                    f"{label} duplicates the unscheduled entry for "
                    f"{course} / {topic_name}."
                )
            else:
                unscheduled_seen.add(key)

        minutes = entry.get("unscheduled_minutes")
        if _positive_int(minutes) is None:
            errors.append(
                f"{label} must have a positive integer unscheduled_minutes."
            )

        reason = entry.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label} must include a non-empty reason.")

        if key is not None and remaining_by_topic.get(key) == 0:
            errors.append(
                f"{label} reports unscheduled minutes for {course} / {topic_name}, "
                "but its remaining requirement is 0 minutes."
            )

    accounting = build_accounting_summary(structured_input, schedule)
    for row in accounting.topics:
        key = topic_key(row.course, row.topic)
        remaining = remaining_by_topic.get(key)
        if remaining is None:
            continue
        if remaining > 0 and row.scheduled_minutes > remaining:
            errors.append(
                f"Scheduled time for {key[0]} / {key[1]} exceeds its remaining "
                f"requirement ({row.scheduled_minutes} scheduled > "
                f"{remaining} remaining)."
            )
        if row.accounted_minutes != remaining:
            errors.append(
                f"Workload mismatch for {key[0]} / {key[1]}: required remaining "
                f"{remaining} minutes, but the response accounts for "
                f"{row.accounted_minutes} minutes ({row.scheduled_minutes} scheduled "
                f"+ {row.reported_unscheduled_minutes} unscheduled)."
            )

    return errors


def _repository_relative_path(path: Path) -> str:
    """Return a non-absolute path for logs, preferring project-relative form."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        pass
    try:
        relative = os.path.relpath(resolved, PROJECT_ROOT)
    except ValueError:
        return path.name
    return Path(relative).as_posix()


def save_run_log(
    result: LLMRunResult, output_dir: str | Path | None = None
) -> Path:
    destination = Path(output_dir) if output_dir is not None else LLM_RUN_DIR
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{result.run_id}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def generate_llm_schedule(
    topics: Sequence[Topic],
    availability: Sequence[AvailabilityWindow],
    current_date: date | None = None,
    completed_sessions: Sequence[StudySession] | None = None,
    output_dir: str | Path | None = None,
    prompt_path: str | Path | None = None,
    config: OllamaConfig | None = None,
    current_datetime: datetime | None = None,
) -> LLMRunResult:
    """Call Ollama, validate, optionally repair once, and log the complete run."""
    started = time_module.perf_counter()
    if current_datetime is None:
        current_datetime = (
            datetime.combine(current_date, time.min)
            if current_date is not None
            else datetime.now()
        )
    current_datetime = _ceil_to_next_minute(current_datetime)
    structured_input: dict[str, Any] = {}

    configured_model = config.model if config is not None else (
        os.environ.get(OLLAMA_MODEL_ENV, "").strip() or None
    )
    configured_url = config.base_url if config is not None else (
        os.environ.get(OLLAMA_BASE_URL_ENV, "").strip() or DEFAULT_OLLAMA_BASE_URL
    )
    parameters = (
        config.model_parameters
        if config is not None
        else {
            "temperature": DEFAULT_OLLAMA_TEMPERATURE,
            "format": "json",
            "stream": False,
            "timeout_seconds": DEFAULT_OLLAMA_TIMEOUT_SECONDS,
            "endpoint": "/api/chat",
        }
    )

    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()
    initial_prompt: str | None = None
    repair_prompt: str | None = None
    raw_initial: str | None = None
    raw_repair: str | None = None
    parsed_schedule: dict[str, Any] | None = None
    initial_errors: list[str] = []
    final_errors: list[str] = []
    repair_attempted = False
    status = "failure"
    error: str | None = None

    try:
        input_errors = validate_dataset(topics, availability)
        if input_errors:
            raise LLMInputError(
                "Invalid planner input:\n" + format_errors(input_errors)
            )
        structured_input = build_structured_input(
            topics,
            availability,
            completed_sessions or [],
            current_date=current_datetime.date(),
            current_datetime=current_datetime,
        )

        active_config = config or OllamaConfig.from_env()
        configured_model = active_config.model
        configured_url = active_config.base_url
        parameters = active_config.model_parameters

        template = load_prompt_template(prompt_path or LLM_PROMPT_PATH)
        initial_prompt = render_initial_prompt(template, structured_input)
        raw_initial = call_ollama(active_config, initial_prompt)
        parsed_initial, parse_errors = parse_llm_json(raw_initial)
        parsed_schedule = parsed_initial
        initial_errors = parse_errors + validate_llm_schedule(
            parsed_initial, structured_input
        )
        final_errors = list(initial_errors)

        if not initial_errors:
            status = "success"
        else:
            repair_attempted = True
            repair_prompt = build_repair_prompt(
                initial_prompt, raw_initial, initial_errors
            )
            # The final result must reflect the repair attempt. If the repair call
            # or parsing fails, do not fall back to the initial parsed schedule.
            parsed_schedule = None
            raw_repair = call_ollama(active_config, repair_prompt)
            parsed_repair, repair_parse_errors = parse_llm_json(raw_repair)
            parsed_schedule = parsed_repair
            final_errors = repair_parse_errors + validate_llm_schedule(
                parsed_repair, structured_input
            )
            if not final_errors:
                status = "success"
                error = None
            else:
                error = "LLM response remained invalid after one repair attempt."
    except LLMInputError as exc:
        error = str(exc)
    except LLMConfigurationError as exc:
        error = str(exc)
    except OllamaUnavailableError as exc:
        error = str(exc)
    except OllamaProtocolError as exc:
        error = str(exc)
    except OSError as exc:
        error = f"Could not read the prompt template or write planner files: {exc}"
    except Exception as exc:  # Keep failures structured for the Streamlit page.
        error = f"Unexpected LLM baseline error ({type(exc).__name__}): {exc}"

    runtime_seconds = time_module.perf_counter() - started
    if status != "success" and error is None:
        error = "LLM run did not produce a valid schedule."

    accounting_summary = (
        build_accounting_summary(structured_input, parsed_schedule)
        if isinstance(parsed_schedule, dict)
        else None
    )

    result = LLMRunResult(
        run_id=run_id,
        timestamp=timestamp,
        model_name=configured_model,
        base_url=configured_url,
        model_parameters=parameters,
        prompt=initial_prompt,
        repair_prompt=repair_prompt,
        structured_input=structured_input,
        raw_initial_response=raw_initial,
        raw_repair_response=raw_repair,
        parsed_schedule=parsed_schedule,
        accounting_summary=accounting_summary,
        validation_errors_initial=initial_errors,
        validation_errors_final=final_errors,
        repair_attempted=repair_attempted,
        runtime_seconds=runtime_seconds,
        status=status,
        error=error,
    )

    destination = Path(output_dir) if output_dir is not None else LLM_RUN_DIR
    result.log_path = _repository_relative_path(destination / f"{run_id}.json")
    try:
        save_run_log(result, destination)
    except OSError as exc:
        result.logging_error = f"Could not save LLM run log: {exc}"
    return result
