from datetime import date, timedelta

import pytest

from models import Topic
from priority import (
    adjusted_study_minutes,
    calculate_priority,
    format_priority,
    knowledge_category,
    round_minutes_to_30,
    round_priority,
    session_durations,
    study_multiplier,
    urgency_for_days,
)


@pytest.mark.parametrize(
    ("score", "category", "multiplier"),
    [
        (0, "Weak", 1.5),
        (59, "Weak", 1.5),
        (60, "Developing", 1.0),
        (79, "Developing", 1.0),
        (80, "Mastered", 0.5),
        (100, "Mastered", 0.5),
    ],
)
def test_knowledge_boundaries(score, category, multiplier):
    assert knowledge_category(score) == category
    assert study_multiplier(score) == multiplier


def test_invalid_score_rejected():
    with pytest.raises(ValueError):
        knowledge_category(101)
    with pytest.raises(ValueError):
        knowledge_category(-1)


def test_urgency_boundaries():
    assert urgency_for_days(-2) == 1.0
    assert urgency_for_days(0) == 1.0
    assert urgency_for_days(15) == 0.5
    assert urgency_for_days(30) == 0.0
    assert urgency_for_days(60) == 0.0


def test_priority_formula_exact():
    topic = Topic(
        course_name="CS",
        topic_name="Recursion",
        deadline=date(2026, 1, 1),
        estimated_hours=2,
        quiz_score=50,
        task_weight=50,
    )
    # weakness=.5, urgency=1, weight=.5 -> 100*(.25+.3+.1)=65
    assert calculate_priority(topic, date(2026, 1, 1)) == pytest.approx(65.0)


def test_priority_display_two_decimals():
    assert round_priority(33.333333) == 33.33
    assert format_priority(33.333333) == "33.33"
    assert format_priority(1.005) == "1.01"
    assert format_priority(None) == "—"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, 0),
        (1, 30),
        (15, 30),
        (29, 30),
        (44, 30),
        (45, 60),
        (59, 60),
        (60, 60),
        (74, 60),
        (75, 90),
        (120, 120),
    ],
)
def test_round_minutes_to_30(raw, expected):
    assert round_minutes_to_30(raw) == expected


def test_adjusted_study_minutes_by_level():
    assert adjusted_study_minutes(2.0, 50) == 180
    assert adjusted_study_minutes(2.0, 60) == 120
    assert adjusted_study_minutes(2.0, 90) == 60
    assert adjusted_study_minutes(0.0, 50) == 0


@pytest.mark.parametrize(
    ("total", "expected"),
    [
        (0, []),
        (30, [30]),
        (60, [60]),
        (90, [60, 30]),
        (120, [60, 60]),
        (150, [60, 60, 30]),
        (210, [60, 60, 60, 30]),
    ],
)
def test_session_durations(total, expected):
    assert session_durations(total) == expected


def test_priority_clamped_to_range():
    topic = Topic(
        course_name="CS",
        topic_name="Test",
        deadline=date.today() + timedelta(days=0),
        estimated_hours=1,
        quiz_score=0,
        task_weight=100,
    )
    result = calculate_priority(topic, date.today())
    assert 0 <= result <= 100
