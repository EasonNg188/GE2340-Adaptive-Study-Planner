from datetime import date, time

from config import PROJECT_ROOT
from models import AvailabilityWindow, Topic
from validation import (
    load_availability_csv,
    load_topics_csv,
    validate_availability,
    validate_dataset,
    validate_topic,
)


def make_topic(**overrides):
    values = dict(
        course_name="CS101",
        topic_name="Recursion",
        deadline=date(2026, 12, 5),
        estimated_hours=2.0,
        quiz_score=50,
        task_weight=70.0,
    )
    values.update(overrides)
    return Topic(**values)


def test_valid_topic():
    assert validate_topic(make_topic()) == []


def test_invalid_topic_fields():
    errors = validate_topic(
        make_topic(quiz_score=101, estimated_hours=-1, task_weight=101)
    )
    assert len(errors) == 3


def test_availability_unknown_topic_and_cross_midnight():
    topic = make_topic()
    unknown = AvailabilityWindow(
        course_name="UNKNOWN",
        topic_name="None",
        available_date=date(2026, 11, 16),
        start_time=time(18, 0),
        duration_minutes=60,
    )
    crossing = AvailabilityWindow(
        course_name="CS101",
        topic_name="Recursion",
        available_date=date(2026, 11, 16),
        start_time=time(23, 30),
        duration_minutes=60,
    )
    assert any("unknown topic" in error for error in validate_availability(unknown, [topic]))
    assert any("cross midnight" in error for error in validate_availability(crossing, [topic]))


def test_dataset_duplicate_topics():
    topic = make_topic()
    errors = validate_dataset([topic, topic], [])
    assert any("Duplicate topic" in error for error in errors)


def test_load_topics_and_availability_sample():
    topics, errors = load_topics_csv(
        (PROJECT_ROOT / "sample_topics.csv").read_text(encoding="utf-8")
    )
    assert not errors
    assert len(topics) == 5
    windows, errors = load_availability_csv(
        (PROJECT_ROOT / "sample_availability.csv").read_text(encoding="utf-8"),
        topics,
    )
    assert not errors
    assert len(windows) == 15


def test_load_topics_csv_reports_row_errors():
    text = (
        "course_name,topic_name,deadline,estimated_hours,quiz_score,task_weight\n"
        "CS101,Test,not-a-date,2,50,70\n"
    )
    topics, errors = load_topics_csv(text)
    assert topics == []
    assert errors


def test_load_availability_csv_requires_known_topic():
    topics = [make_topic()]
    text = (
        "course_name,topic_name,available_date,start_time,duration_minutes\n"
        "CS101,Recursion,2026-11-16,18:00,120\n"
        "NOPE,Nope,2026-11-16,18:00,120\n"
    )
    windows, errors = load_availability_csv(text, topics)
    assert len(windows) == 1
    assert any("unknown topic" in error for error in errors)
