from datetime import datetime, time, timedelta

from baseline import (
    baseline_required_minutes,
    generate_baseline_schedule,
    sort_topics_by_deadline,
)
from models import AvailabilityWindow, Topic

NOW = datetime(2026, 11, 15, 8, 0)


def make_topic(name, *, score=50, weight=50, hours=1.0, deadline_days=10):
    return Topic(
        course_name="CS101",
        topic_name=name,
        deadline=NOW.date() + timedelta(days=deadline_days),
        estimated_hours=hours,
        quiz_score=score,
        task_weight=weight,
    )


def window(topic, *, days=1, hour=18):
    return AvailabilityWindow(
        course_name=topic.course_name,
        topic_name=topic.topic_name,
        available_date=NOW.date() + timedelta(days=days),
        start_time=time(hour, 0),
        duration_minutes=120,
    )


def test_baseline_uses_original_hours_not_quiz_multiplier():
    weak = make_topic("Weak", score=0, hours=2.0)
    mastered = make_topic("Mastered", score=100, hours=2.0)
    assert baseline_required_minutes(weak) == 120
    assert baseline_required_minutes(mastered) == 120


def test_baseline_sorts_deadline_only_and_input_order():
    early = make_topic("Early", score=100, weight=0, deadline_days=2)
    late = make_topic("Late", score=0, weight=100, deadline_days=10)
    ordered = sort_topics_by_deadline([late, early])
    assert [topic.topic_name for topic in ordered] == ["Early", "Late"]

    first = make_topic("First", score=100, weight=0, deadline_days=5)
    second = make_topic("Second", score=0, weight=100, deadline_days=5)
    ordered = sort_topics_by_deadline([first, second])
    assert [topic.topic_name for topic in ordered] == ["First", "Second"]


def test_baseline_schedule_does_not_use_completion_or_priority():
    first = make_topic("First", score=100, weight=0, deadline_days=5)
    second = make_topic("Second", score=0, weight=100, deadline_days=5)
    windows = [window(first), window(second)]
    result = generate_baseline_schedule([first, second], windows, now=NOW)
    assert result.mode == "baseline"
    assert len(result.sessions) == 2
    assert result.sessions[0].topic_name == "First"
    assert all(plan.priority is None for plan in result.topic_plans)
    assert all(plan.category is None for plan in result.topic_plans)
    assert result.total_completed_minutes == 0


def test_baseline_uses_shared_deadline_and_session_rules():
    topic = make_topic("Topic", score=90, hours=1.0, deadline_days=2)
    result = generate_baseline_schedule([topic], [window(topic, days=3)], now=NOW)
    assert result.sessions == []
    assert result.total_unscheduled_minutes == 60
    assert any("after the deadline" in warning for warning in result.warnings)
