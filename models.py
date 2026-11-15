"""Domain models shared by the rule-based and LLM planners."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP

TopicKey = tuple[str, str]


def topic_key(course_name: str, topic_name: str) -> TopicKey:
    return (course_name.strip(), topic_name.strip())


@dataclass(frozen=True)
class Topic:
    course_name: str
    topic_name: str
    deadline: date
    estimated_hours: float
    quiz_score: int
    task_weight: float

    @property
    def key(self) -> TopicKey:
        return topic_key(self.course_name, self.topic_name)

    @property
    def label(self) -> str:
        return f"{self.course_name} / {self.topic_name}"


@dataclass(frozen=True)
class AvailabilityWindow:
    course_name: str
    topic_name: str
    available_date: date
    start_time: time
    duration_minutes: int

    @property
    def key(self) -> TopicKey:
        return topic_key(self.course_name, self.topic_name)

    @property
    def start_datetime(self) -> datetime:
        return datetime.combine(self.available_date, self.start_time)

    @property
    def end_datetime(self) -> datetime:
        return self.start_datetime + timedelta(minutes=self.duration_minutes)


@dataclass
class StudySession:
    session_id: str
    course_name: str
    topic_name: str
    session_date: date
    start_time: time
    duration_minutes: int
    completed: bool = False
    source: str = "adaptive"
    reason: str = ""

    @property
    def key(self) -> TopicKey:
        return topic_key(self.course_name, self.topic_name)

    @property
    def start_datetime(self) -> datetime:
        return datetime.combine(self.session_date, self.start_time)

    @property
    def end_datetime(self) -> datetime:
        return self.start_datetime + timedelta(minutes=self.duration_minutes)

    @property
    def status_label(self) -> str:
        return "Completed" if self.completed else "Incomplete"


@dataclass
class TopicPlan:
    topic: Topic
    required_minutes: int
    completed_minutes: int = 0
    remaining_minutes: int = 0
    scheduled_minutes: int = 0
    unscheduled_minutes: int = 0
    priority: float | None = None
    category: str | None = None
    multiplier: float | None = None

    @property
    def priority_display(self) -> float | None:
        if self.priority is None:
            return None
        return float(
            Decimal(str(self.priority)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )


@dataclass
class ScheduleResult:
    mode: str
    generated_at: datetime
    topic_plans: list[TopicPlan] = field(default_factory=list)
    sessions: list[StudySession] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_scheduled_minutes(self) -> int:
        return sum(plan.scheduled_minutes for plan in self.topic_plans)

    @property
    def total_unscheduled_minutes(self) -> int:
        return sum(plan.unscheduled_minutes for plan in self.topic_plans)

    @property
    def total_completed_minutes(self) -> int:
        return sum(plan.completed_minutes for plan in self.topic_plans)

    @property
    def total_required_minutes(self) -> int:
        return sum(plan.required_minutes for plan in self.topic_plans)

    @property
    def total_remaining_minutes(self) -> int:
        return sum(plan.remaining_minutes for plan in self.topic_plans)

    def plan_for(self, key: TopicKey) -> TopicPlan | None:
        for plan in self.topic_plans:
            if plan.topic.key == key:
                return plan
        return None
