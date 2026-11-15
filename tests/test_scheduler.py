from datetime import date, datetime, time, timedelta

from models import AvailabilityWindow, StudySession, Topic
from scheduler import generate_adaptive_schedule, sort_topics_by_priority

NOW = datetime(2026, 11, 15, 8, 0)


def make_topic(
    name="Recursion",
    *,
    course="CS101",
    score=50,
    weight=70.0,
    hours=1.0,
    deadline_days=10,
):
    return Topic(
        course_name=course,
        topic_name=name,
        deadline=NOW.date() + timedelta(days=deadline_days),
        estimated_hours=hours,
        quiz_score=score,
        task_weight=weight,
    )


def make_window(topic, *, days=1, hour=18, minute=0, duration=120):
    return AvailabilityWindow(
        course_name=topic.course_name,
        topic_name=topic.topic_name,
        available_date=NOW.date() + timedelta(days=days),
        start_time=time(hour, minute),
        duration_minutes=duration,
    )


def test_sort_priority_then_deadline_then_score():
    later = make_topic("Later", deadline_days=50)
    earlier = make_topic("Earlier", deadline_days=40)
    ordered = sort_topics_by_priority([later, earlier], NOW.date())
    assert [topic.topic_name for topic in ordered] == ["Earlier", "Later"]

    weak = make_topic("Weak", score=50, weight=60, deadline_days=10)
    developing = make_topic("Developing", score=60, weight=85, deadline_days=10)
    ordered = sort_topics_by_priority([developing, weak], NOW.date())
    assert [topic.topic_name for topic in ordered] == ["Weak", "Developing"]


def test_normal_one_session_per_topic_per_day():
    topic = make_topic(hours=2.0, score=60)
    windows = [make_window(topic, days=1), make_window(topic, days=2)]
    result = generate_adaptive_schedule([topic], windows, now=NOW)
    assert result.total_scheduled_minutes == 120
    assert result.total_unscheduled_minutes == 0
    assert len(result.sessions) == 2
    assert {session.session_date for session in result.sessions} == {
        NOW.date() + timedelta(days=1),
        NOW.date() + timedelta(days=2),
    }


def test_multiple_sessions_when_necessary():
    topic = make_topic(hours=2.0, score=60)
    windows = [make_window(topic, days=1, duration=240)]
    result = generate_adaptive_schedule([topic], windows, now=NOW)
    assert len(result.sessions) == 2
    assert len({session.session_date for session in result.sessions}) == 1
    assert any("More than one session" in warning for warning in result.warnings)
    assert result.total_unscheduled_minutes == 0


def test_earliest_availability_and_deadline_day():
    topic = make_topic(hours=1.0, score=60, deadline_days=2)
    windows = [
        make_window(topic, days=2),
        make_window(topic, days=1),
    ]
    result = generate_adaptive_schedule([topic], windows, now=NOW)
    assert result.sessions[0].session_date == NOW.date() + timedelta(days=1)


def test_session_after_deadline_is_unscheduled():
    topic = make_topic(hours=1.0, score=60, deadline_days=2)
    windows = [make_window(topic, days=3)]
    result = generate_adaptive_schedule([topic], windows, now=NOW)
    assert result.sessions == []
    assert result.total_unscheduled_minutes == 60
    assert any("after the deadline" in warning for warning in result.warnings)


def test_overdue_deadline_warning():
    topic = make_topic(hours=1.0, score=60, deadline_days=-1)
    result = generate_adaptive_schedule([topic], [], now=NOW)
    assert any("Overdue deadline" in warning for warning in result.warnings)
    assert result.total_unscheduled_minutes == 60


def test_past_availability_is_ignored():
    topic = make_topic(hours=1.0, score=60)
    windows = [make_window(topic, days=-1)]
    result = generate_adaptive_schedule([topic], windows, now=NOW)
    assert result.sessions == []
    assert result.total_unscheduled_minutes == 60


def test_no_overlap_between_topics_sharing_a_slot():
    first = make_topic("High", score=0)
    second = make_topic("Low", score=100)
    windows = [
        make_window(first, days=1),
        make_window(second, days=1),
    ]
    result = generate_adaptive_schedule([first, second], windows, now=NOW)
    intervals = sorted(
        (session.start_datetime, session.end_datetime) for session in result.sessions
    )
    assert len(intervals) == 3
    for earlier, later in zip(intervals, intervals[1:]):
        assert earlier[1] <= later[0]


def test_adaptive_update_preserves_completed_and_regenerates():
    topic = make_topic(hours=2.0, score=60)
    windows = [make_window(topic, days=1), make_window(topic, days=2)]
    initial = generate_adaptive_schedule([topic], windows, now=NOW)
    completed = initial.sessions[0]
    completed.completed = True
    updated = generate_adaptive_schedule(
        [topic], windows, now=NOW, preserved_sessions=initial.sessions
    )
    assert completed.session_id in {session.session_id for session in updated.sessions}
    assert updated.plan_for(topic.key).completed_minutes == 60
    assert updated.plan_for(topic.key).remaining_minutes == 60
    assert updated.total_scheduled_minutes == 60


def test_adaptive_update_clamps_completed_over_requirement():
    topic = make_topic(hours=2.0, score=80)
    windows = [make_window(topic, days=1), make_window(topic, days=2)]
    completed = [
        StudySession(
            session_id="a",
            course_name=topic.course_name,
            topic_name=topic.topic_name,
            session_date=NOW.date() + timedelta(days=1),
            start_time=time(18, 0),
            duration_minutes=60,
            completed=True,
        ),
        StudySession(
            session_id="b",
            course_name=topic.course_name,
            topic_name=topic.topic_name,
            session_date=NOW.date() + timedelta(days=2),
            start_time=time(18, 0),
            duration_minutes=60,
            completed=True,
        ),
    ]
    result = generate_adaptive_schedule(
        [topic], windows, now=NOW, preserved_sessions=completed
    )
    plan = result.plan_for(topic.key)
    assert plan.completed_minutes == 120
    assert plan.required_minutes == 60
    assert plan.remaining_minutes == 0
    assert result.total_scheduled_minutes == 0
    assert any("clamped" in warning for warning in result.warnings)


def test_future_incomplete_is_regenerated_and_past_incomplete_preserved():
    topic = make_topic(hours=1.0, score=80)
    past = StudySession(
        session_id="past",
        course_name=topic.course_name,
        topic_name=topic.topic_name,
        session_date=NOW.date() - timedelta(days=1),
        start_time=time(18, 0),
        duration_minutes=60,
        completed=False,
    )
    future = StudySession(
        session_id="future",
        course_name=topic.course_name,
        topic_name=topic.topic_name,
        session_date=NOW.date() + timedelta(days=1),
        start_time=time(18, 0),
        duration_minutes=60,
        completed=False,
    )
    windows = [make_window(topic, days=2)]
    result = generate_adaptive_schedule(
        [topic], windows, now=NOW, preserved_sessions=[past, future]
    )
    ids = {session.session_id for session in result.sessions}
    assert "past" in ids
    assert "future" not in ids
    assert result.plan_for(topic.key).completed_minutes == 0


def test_unscheduled_minutes_reported():
    topic = make_topic(hours=1.0, score=60)
    result = generate_adaptive_schedule([topic], [], now=NOW)
    assert result.total_unscheduled_minutes == 60
    assert any("Unscheduled" in warning for warning in result.warnings)
