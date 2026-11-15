"""Validation for planner inputs and CSV imports."""

from __future__ import annotations

import csv
import io
import math
import re
from datetime import date, datetime, time, timedelta
from typing import Iterable, Sequence

from config import (
    AVAILABILITY_COLUMNS,
    DATE_FORMAT,
    MAX_QUIZ_SCORE,
    MIN_QUIZ_SCORE,
    TIME_FORMAT,
    TOPIC_COLUMNS,
)
from models import AvailabilityWindow, Topic, topic_key

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_RE = re.compile(r"^\d{2}:\d{2}$")


def parse_date_strict(value: str) -> date:
    if not isinstance(value, str) or not DATE_RE.fullmatch(value):
        raise ValueError(f"Expected date in YYYY-MM-DD format, got {value!r}.")
    return datetime.strptime(value, DATE_FORMAT).date()


def parse_time_strict(value: str) -> time:
    if not isinstance(value, str) or not TIME_RE.fullmatch(value):
        raise ValueError(f"Expected 24-hour time in HH:MM format, got {value!r}.")
    return datetime.strptime(value, TIME_FORMAT).time()


def validate_topic(topic: Topic) -> list[str]:
    errors: list[str] = []
    if not isinstance(topic.course_name, str) or not topic.course_name.strip():
        errors.append("Course name is required.")
    if not isinstance(topic.topic_name, str) or not topic.topic_name.strip():
        errors.append("Topic name is required.")
    if not isinstance(topic.deadline, date) or isinstance(topic.deadline, datetime):
        errors.append("Deadline must be a date.")
    if (
        isinstance(topic.estimated_hours, bool)
        or not isinstance(topic.estimated_hours, (int, float))
        or not math.isfinite(topic.estimated_hours)
        or topic.estimated_hours < 0
    ):
        errors.append("Estimated study hours must be a finite number greater than or equal to 0.")
    if (
        isinstance(topic.quiz_score, bool)
        or not isinstance(topic.quiz_score, int)
        or not MIN_QUIZ_SCORE <= topic.quiz_score <= MAX_QUIZ_SCORE
    ):
        errors.append("Quiz score must be an integer between 0 and 100.")
    if (
        isinstance(topic.task_weight, bool)
        or not isinstance(topic.task_weight, (int, float))
        or not math.isfinite(topic.task_weight)
        or not 0 <= topic.task_weight <= 100
    ):
        errors.append("Task weight must be a finite number between 0 and 100.")
    return errors


def validate_availability(
    window: AvailabilityWindow, topics: Sequence[Topic]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(window.course_name, str) or not window.course_name.strip():
        errors.append("Course name is required.")
    if not isinstance(window.topic_name, str) or not window.topic_name.strip():
        errors.append("Topic name is required.")
    known_keys = {topic.key for topic in topics}
    if window.key not in known_keys:
        errors.append(
            f"Availability refers to an unknown topic: {window.course_name} / {window.topic_name}."
        )
    if not isinstance(window.available_date, date) or isinstance(window.available_date, datetime):
        errors.append("Available date must be a date.")
    if not isinstance(window.start_time, time):
        errors.append("Start time must be a time.")
    if (
        isinstance(window.duration_minutes, bool)
        or not isinstance(window.duration_minutes, int)
        or window.duration_minutes <= 0
    ):
        errors.append("Available duration must be a positive whole number of minutes.")
    if (
        isinstance(window.available_date, date)
        and not isinstance(window.available_date, datetime)
        and isinstance(window.start_time, time)
        and isinstance(window.duration_minutes, int)
        and not isinstance(window.duration_minutes, bool)
        and window.duration_minutes > 0
    ):
        start = datetime.combine(window.available_date, window.start_time)
        end = start + timedelta(minutes=window.duration_minutes)
        if end.date() != window.available_date:
            errors.append("Availability windows may not cross midnight.")
    return errors


def validate_dataset(
    topics: Sequence[Topic], availability: Sequence[AvailabilityWindow]
) -> list[str]:
    errors: list[str] = []
    if not topics:
        errors.append("At least one topic is required.")

    seen: set[tuple[str, str]] = set()
    for index, topic in enumerate(topics, start=1):
        for message in validate_topic(topic):
            errors.append(f"Topic {index} ({topic.course_name} / {topic.topic_name}): {message}")
        if topic.key in seen:
            errors.append(
                f"Duplicate topic: {topic.course_name} / {topic.topic_name}. "
                "Course and topic names must be unique together."
            )
        seen.add(topic.key)

    for index, window in enumerate(availability, start=1):
        for message in validate_availability(window, topics):
            errors.append(
                f"Availability {index} "
                f"({window.course_name} / {window.topic_name}): {message}"
            )
    return errors


def _decode_csv(content: str | bytes) -> str:
    if isinstance(content, bytes):
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("CSV must be UTF-8 encoded.") from exc
    else:
        text = content.lstrip("\ufeff")
    return text


def _read_rows(
    content: str | bytes, required_columns: Sequence[str]
) -> tuple[list[dict[str, str]], list[str]]:
    text = _decode_csv(content)
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        return [], ["CSV has no header row."]
    normalized_headers = [header.strip() for header in reader.fieldnames]
    missing = [column for column in required_columns if column not in normalized_headers]
    if missing:
        return [], [f"CSV is missing required columns: {', '.join(missing)}."]

    rows: list[dict[str, str]] = []
    for raw_row in reader:
        row = {
            (key or "").strip(): (value or "").strip()
            for key, value in raw_row.items()
            if key is not None
        }
        if any(value for value in row.values()):
            rows.append(row)
    return rows, []


def load_topics_csv(content: str | bytes) -> tuple[list[Topic], list[str]]:
    rows, errors = _read_rows(content, TOPIC_COLUMNS)
    if errors:
        return [], errors
    if not rows:
        return [], ["CSV contains no topic rows."]

    topics: list[Topic] = []
    for index, row in enumerate(rows, start=2):
        prefix = f"Row {index}"
        try:
            deadline = parse_date_strict(row["deadline"])
            estimated_hours = float(row["estimated_hours"])
            quiz_score = int(row["quiz_score"])
            task_weight = float(row["task_weight"])
            topic = Topic(
                course_name=row["course_name"],
                topic_name=row["topic_name"],
                deadline=deadline,
                estimated_hours=estimated_hours,
                quiz_score=quiz_score,
                task_weight=task_weight,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        row_errors = validate_topic(topic)
        if row_errors:
            for message in row_errors:
                errors.append(f"{prefix}: {message}")
            continue
        topics.append(topic)

    seen: set[tuple[str, str]] = set()
    for topic in topics:
        if topic.key in seen:
            errors.append(
                f"Duplicate topic: {topic.course_name} / {topic.topic_name}."
            )
        seen.add(topic.key)
    return topics, errors


def load_availability_csv(
    content: str | bytes, topics: Sequence[Topic]
) -> tuple[list[AvailabilityWindow], list[str]]:
    rows, errors = _read_rows(content, AVAILABILITY_COLUMNS)
    if errors:
        return [], errors
    if not rows:
        return [], ["CSV contains no availability rows."]

    windows: list[AvailabilityWindow] = []
    for index, row in enumerate(rows, start=2):
        prefix = f"Row {index}"
        try:
            available_date = parse_date_strict(row["available_date"])
            start_time = parse_time_strict(row["start_time"])
            duration_minutes = int(row["duration_minutes"])
            window = AvailabilityWindow(
                course_name=row["course_name"],
                topic_name=row["topic_name"],
                available_date=available_date,
                start_time=start_time,
                duration_minutes=duration_minutes,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"{prefix}: {exc}")
            continue
        row_errors = validate_availability(window, topics)
        if row_errors:
            for message in row_errors:
                errors.append(f"{prefix}: {message}")
            continue
        windows.append(window)
    return windows, errors


def format_errors(errors: Iterable[str]) -> str:
    return "\n".join(f"- {error}" for error in errors)
