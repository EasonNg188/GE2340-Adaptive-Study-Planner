"""Rule-based adaptive scheduling and shared allocation logic."""

from __future__ import annotations

import copy
import uuid
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Sequence

from models import (
    AvailabilityWindow,
    ScheduleResult,
    StudySession,
    Topic,
    TopicKey,
    TopicPlan,
)
from priority import (
    adjusted_study_minutes,
    calculate_priority,
    knowledge_category,
    session_durations,
    study_multiplier,
)


def sort_topics_by_priority(topics: Sequence[Topic], today: date) -> list[Topic]:
    indexed = list(enumerate(topics))
    indexed.sort(
        key=lambda item: (
            -calculate_priority(item[1], today),
            item[1].deadline,
            item[1].quiz_score,
            item[0],
        )
    )
    return [topic for _, topic in indexed]


def _ceil_to_next_minute(value: datetime) -> datetime:
    floored = value.replace(second=0, microsecond=0)
    if value > floored:
        floored += timedelta(minutes=1)
    return floored


def build_adaptive_plans(
    topics: Sequence[Topic],
    now: datetime,
    completed_sessions: Sequence[StudySession] = (),
) -> list[TopicPlan]:
    completed_by_topic: dict[TopicKey, int] = defaultdict(int)
    for session in completed_sessions:
        if session.completed:
            completed_by_topic[session.key] += session.duration_minutes

    plans: list[TopicPlan] = []
    for topic in topics:
        required = adjusted_study_minutes(topic.estimated_hours, topic.quiz_score)
        completed = completed_by_topic.get(topic.key, 0)
        plans.append(
            TopicPlan(
                topic=topic,
                required_minutes=required,
                completed_minutes=completed,
                remaining_minutes=max(0, required - completed),
                priority=calculate_priority(topic, now.date()),
                category=knowledge_category(topic.quiz_score),
                multiplier=study_multiplier(topic.quiz_score),
            )
        )

    input_order = {topic.key: index for index, topic in enumerate(topics)}
    plans.sort(
        key=lambda plan: (
            -float(plan.priority),
            plan.topic.deadline,
            plan.topic.quiz_score,
            input_order.get(plan.topic.key, 0),
        )
    )
    return plans


def split_preserved_sessions(
    sessions: Sequence[StudySession], now: datetime
) -> tuple[list[StudySession], list[StudySession]]:
    """Return completed sessions and historical incomplete sessions.

    Future/current incomplete sessions are intentionally omitted so they can be
    regenerated. Past incomplete sessions remain visible as history.
    """
    completed: list[StudySession] = []
    historical_incomplete: list[StudySession] = []
    for session in sessions:
        copied = copy.copy(session)
        if copied.completed:
            completed.append(copied)
        elif copied.end_datetime < now:
            historical_incomplete.append(copied)
    return completed, historical_incomplete


def generate_adaptive_schedule(
    topics: Sequence[Topic],
    availability: Sequence[AvailabilityWindow],
    now: datetime | None = None,
    preserved_sessions: Sequence[StudySession] | None = None,
) -> ScheduleResult:
    now = now or datetime.now()
    completed, historical = split_preserved_sessions(preserved_sessions or [], now)
    plans = build_adaptive_plans(topics, now, completed)

    warnings: list[str] = []
    for plan in plans:
        if plan.completed_minutes > plan.required_minutes:
            warnings.append(
                f"Completed time exceeds the adjusted requirement for "
                f"{plan.topic.label}; remaining time was clamped to 0 minutes."
            )

    result = allocate_plans(
        plans=plans,
        availability=availability,
        now=now,
        completed_sessions=completed,
        historical_sessions=historical,
        mode="adaptive",
    )
    result.warnings = list(dict.fromkeys(warnings + result.warnings))
    return result


def _deadline_exclusive(deadline: date) -> datetime:
    return datetime.combine(deadline + timedelta(days=1), time.min)


def _usable_windows(
    plan: TopicPlan,
    availability: Sequence[AvailabilityWindow],
    now: datetime,
) -> list[tuple[AvailabilityWindow, datetime, datetime]]:
    now_ceil = _ceil_to_next_minute(now)
    deadline_exclusive = _deadline_exclusive(plan.topic.deadline)
    usable: list[tuple[AvailabilityWindow, datetime, datetime]] = []

    for window in availability:
        if window.key != plan.topic.key:
            continue
        window_start = window.start_datetime
        window_end = window.end_datetime
        effective_start = max(window_start, now_ceil)
        effective_end = min(window_end, deadline_exclusive)
        if effective_start < effective_end:
            usable.append((window, effective_start, effective_end))

    usable.sort(key=lambda item: item[1])
    return usable


def _find_slot(
    plan: TopicPlan,
    duration: int,
    usable_windows: Sequence[tuple[AvailabilityWindow, datetime, datetime]],
    busy: Sequence[tuple[datetime, datetime]],
    daily_counts: dict[tuple[TopicKey, date], int],
    one_session_per_day: bool,
) -> tuple[AvailabilityWindow, datetime] | None:
    sorted_busy = sorted(busy, key=lambda interval: interval[0])
    for window, window_start, window_end in usable_windows:
        candidate = window_start
        if one_session_per_day and daily_counts.get(
            (plan.topic.key, candidate.date()), 0
        ) >= 1:
            continue

        fits = False
        for busy_start, busy_end in sorted_busy:
            if busy_end <= candidate:
                continue
            if busy_start >= candidate + timedelta(minutes=duration):
                fits = True
                break
            if busy_start < candidate + timedelta(minutes=duration):
                candidate = busy_end
                if candidate + timedelta(minutes=duration) > window_end:
                    break

        if candidate + timedelta(minutes=duration) <= window_end:
            fits = True

        if not fits:
            continue
        if one_session_per_day and daily_counts.get(
            (plan.topic.key, candidate.date()), 0
        ) >= 1:
            continue
        return window, candidate
    return None


def _book_session(
    plan: TopicPlan,
    start: datetime,
    duration: int,
    mode: str,
    new_sessions: list[StudySession],
    busy: list[tuple[datetime, datetime]],
    daily_counts: dict[tuple[TopicKey, date], int],
) -> StudySession:
    session = StudySession(
        session_id=uuid.uuid4().hex,
        course_name=plan.topic.course_name,
        topic_name=plan.topic.topic_name,
        session_date=start.date(),
        start_time=start.time().replace(second=0, microsecond=0),
        duration_minutes=duration,
        completed=False,
        source=mode,
        reason="Allocated to the earliest available slot before the deadline.",
    )
    new_sessions.append(session)
    busy.append((session.start_datetime, session.end_datetime))
    daily_counts[(plan.topic.key, session.session_date)] += 1
    plan.scheduled_minutes += duration
    return session


def allocate_plans(
    plans: Sequence[TopicPlan],
    availability: Sequence[AvailabilityWindow],
    now: datetime,
    completed_sessions: Sequence[StudySession] | None = None,
    historical_sessions: Sequence[StudySession] | None = None,
    mode: str = "adaptive",
) -> ScheduleResult:
    """Allocate ordered plans using the shared session-size and deadline rules."""
    completed_sessions = list(completed_sessions or [])
    historical_sessions = list(historical_sessions or [])
    working_plans: list[TopicPlan] = []
    for original in plans:
        plan = copy.copy(original)
        plan.scheduled_minutes = 0
        plan.unscheduled_minutes = 0
        working_plans.append(plan)

    busy: list[tuple[datetime, datetime]] = [
        (session.start_datetime, session.end_datetime)
        for session in completed_sessions
    ]
    daily_counts: dict[tuple[TopicKey, date], int] = defaultdict(int)
    for session in completed_sessions:
        daily_counts[(session.key, session.session_date)] += 1

    warnings: list[str] = []
    for plan in working_plans:
        topic_windows = [window for window in availability if window.key == plan.topic.key]
        if plan.topic.deadline < now.date():
            warnings.append(
                f"Overdue deadline: {plan.topic.label} was due on "
                f"{plan.topic.deadline.isoformat()}."
            )
        if plan.remaining_minutes > 0:
            if not topic_windows:
                warnings.append(
                    f"No availability is configured for {plan.topic.label}."
                )
            elif all(window.available_date > plan.topic.deadline for window in topic_windows):
                warnings.append(
                    f"All availability for {plan.topic.label} is after the deadline."
                )
            else:
                usable = _usable_windows(plan, availability, now)
                if not usable:
                    warnings.append(
                        f"No future availability remains on or before the deadline "
                        f"for {plan.topic.label}."
                    )

    new_sessions: list[StudySession] = []
    pending: list[tuple[TopicPlan, int]] = []

    # Pass 1: normal rule, at most one session per topic per day.
    for plan in working_plans:
        durations = session_durations(plan.remaining_minutes)
        usable = _usable_windows(plan, availability, now)
        for duration in durations:
            slot = _find_slot(
                plan=plan,
                duration=duration,
                usable_windows=usable,
                busy=busy,
                daily_counts=daily_counts,
                one_session_per_day=True,
            )
            if slot is None:
                pending.append((plan, duration))
                continue
            _, start = slot
            _book_session(
                plan, start, duration, mode, new_sessions, busy, daily_counts
            )

    # Pass 2: relax the daily cap only for work that could not otherwise fit.
    forced_days: dict[TopicKey, set[date]] = defaultdict(set)
    for plan, duration in pending:
        usable = _usable_windows(plan, availability, now)
        slot = _find_slot(
            plan=plan,
            duration=duration,
            usable_windows=usable,
            busy=busy,
            daily_counts=daily_counts,
            one_session_per_day=False,
        )
        if slot is None:
            plan.unscheduled_minutes += duration
            continue
        _, start = slot
        if daily_counts.get((plan.topic.key, start.date()), 0) >= 1:
            forced_days[plan.topic.key].add(start.date())
        _book_session(plan, start, duration, mode, new_sessions, busy, daily_counts)

    for plan in working_plans:
        if plan.unscheduled_minutes > 0:
            warnings.append(
                f"Unscheduled time for {plan.topic.label}: "
                f"{plan.unscheduled_minutes} minutes."
            )
    for key, dates in forced_days.items():
        label = next(
            (plan.topic.label for plan in working_plans if plan.topic.key == key),
            f"{key[0]} / {key[1]}",
        )
        formatted_dates = ", ".join(sorted(day.isoformat() for day in dates))
        warnings.append(
            f"More than one session was necessary for {label} on {formatted_dates} "
            "to fit the remaining work before the deadline."
        )

    all_sessions = completed_sessions + historical_sessions + new_sessions
    all_sessions.sort(
        key=lambda session: (
            session.session_date,
            session.start_time,
            session.course_name,
            session.topic_name,
            session.session_id,
        )
    )

    return ScheduleResult(
        mode=mode,
        generated_at=now,
        topic_plans=working_plans,
        sessions=all_sessions,
        warnings=list(dict.fromkeys(warnings)),
    )
