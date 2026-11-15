"""Deadline-only, non-adaptive scheduling baseline."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from models import AvailabilityWindow, ScheduleResult, Topic, TopicPlan
from priority import round_minutes_to_30
from scheduler import allocate_plans


def baseline_required_minutes(topic: Topic) -> int:
    """Use original estimated study hours and ignore quiz-score adaptation."""
    return round_minutes_to_30(float(topic.estimated_hours) * 60.0)


def sort_topics_by_deadline(topics: Sequence[Topic]) -> list[Topic]:
    indexed = list(enumerate(topics))
    indexed.sort(key=lambda item: (item[1].deadline, item[0]))
    return [topic for _, topic in indexed]


def build_baseline_plans(topics: Sequence[Topic]) -> list[TopicPlan]:
    plans: list[TopicPlan] = []
    for topic in sort_topics_by_deadline(topics):
        required = baseline_required_minutes(topic)
        plans.append(
            TopicPlan(
                topic=topic,
                required_minutes=required,
                completed_minutes=0,
                remaining_minutes=required,
                priority=None,
                category=None,
                multiplier=None,
            )
        )
    return plans


def generate_baseline_schedule(
    topics: Sequence[Topic],
    availability: Sequence[AvailabilityWindow],
    now: datetime | None = None,
) -> ScheduleResult:
    """Generate the deadline-only baseline.

    Quiz scores, task weights, knowledge categories, adaptive multipliers, and
    completed session status are deliberately ignored.
    """
    now = now or datetime.now()
    return allocate_plans(
        plans=build_baseline_plans(topics),
        availability=availability,
        now=now,
        completed_sessions=[],
        historical_sessions=[],
        mode="baseline",
    )
