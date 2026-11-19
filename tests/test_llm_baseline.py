import copy
import json
from datetime import date, datetime, time
from pathlib import Path

import pytest
import requests

import llm_baseline
from llm_baseline import (
    OllamaConfig,
    OllamaProtocolError,
    build_accounting_summary,
    build_structured_input,
    call_ollama,
    generate_llm_schedule,
    llm_required_minutes,
    parse_llm_json,
    validate_llm_schedule,
)
from models import AvailabilityWindow, StudySession, Topic

CURRENT_DATE = date(2026, 11, 15)
TOPIC = Topic(
    course_name="CS101",
    topic_name="Recursion",
    deadline=date(2026, 11, 20),
    estimated_hours=2.0,
    quiz_score=50,
    task_weight=50.0,
)
WINDOW = AvailabilityWindow(
    course_name="CS101",
    topic_name="Recursion",
    available_date=date(2026, 11, 16),
    start_time=time(18, 0),
    duration_minutes=120,
)
PAYLOAD = build_structured_input([TOPIC], [WINDOW], [], CURRENT_DATE)
CONFIG = OllamaConfig(model="test-model:latest", base_url="http://localhost:11434")

VALID_SCHEDULE = {
    "sessions": [
        {
            "course": "CS101",
            "topic": "Recursion",
            "date": "2026-11-16",
            "start_time": "18:00",
            "duration_minutes": 60,
            "reason": "Use the earliest available slot.",
        },
        {
            "course": "CS101",
            "topic": "Recursion",
            "date": "2026-11-16",
            "start_time": "19:00",
            "duration_minutes": 60,
            "reason": "Finish the remaining required minutes.",
        },
    ],
    "unscheduled_topics": [],
}


def make_topic(
    *,
    name: str = "Recursion",
    course: str = "CS101",
    hours: float = 3.0,
    score: int = 50,
) -> Topic:
    return Topic(
        course_name=course,
        topic_name=name,
        deadline=date(2026, 11, 20),
        estimated_hours=hours,
        quiz_score=score,
        task_weight=50.0,
    )


def make_window(
    topic: Topic,
    *,
    minutes: int = 180,
    day: int = 16,
    hour: int = 18,
) -> AvailabilityWindow:
    return AvailabilityWindow(
        course_name=topic.course_name,
        topic_name=topic.topic_name,
        available_date=date(2026, 11, day),
        start_time=time(hour, 0),
        duration_minutes=minutes,
    )


def make_session(
    *,
    course: str = "CS101",
    name: str = "Recursion",
    day: int = 16,
    hour: int = 18,
    duration: int = 60,
) -> dict:
    return {
        "course": course,
        "topic": name,
        "date": f"2026-11-{day:02d}",
        "start_time": f"{hour:02d}:00",
        "duration_minutes": duration,
        "reason": "Scheduled within an available window.",
    }


def make_unscheduled(
    *,
    course: str = "CS101",
    name: str = "Recursion",
    minutes=120,
) -> dict:
    return {
        "course": course,
        "topic": name,
        "unscheduled_minutes": minutes,
        "reason": "No further suitable availability.",
    }


class FakeResponse:
    def __init__(self, content, status_code=200):
        self._content = content
        self.status_code = status_code
        self.text = "" if status_code < 400 else content

    def json(self):
        return {"message": {"content": self._content}}


def install_fake_post(monkeypatch, responses):
    calls = []

    def fake_post(url, **kwargs):
        if len(calls) >= len(responses):
            raise AssertionError("Unexpected extra Ollama request")
        item = responses[len(calls)]
        calls.append({"url": url, "kwargs": kwargs})
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    monkeypatch.setattr(llm_baseline.requests, "post", fake_post)
    return calls


def as_json(payload):
    return json.dumps(payload)


def test_valid_json_output(tmp_path, monkeypatch):
    calls = install_fake_post(monkeypatch, [as_json(VALID_SCHEDULE)])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert result.success
    assert result.validation_status == "valid"
    assert result.repair_attempted is False
    assert result.model_name == "test-model:latest"
    assert result.model_parameters["temperature"] == 0.0
    assert len(calls) == 1
    assert calls[0]["url"].endswith("/api/chat")
    assert calls[0]["kwargs"]["json"]["model"] == "test-model:latest"
    assert calls[0]["kwargs"]["json"]["options"]["temperature"] == 0.0
    assert "Recursion" in result.prompt
    assert "generated_scheduled_minutes" in result.prompt
    assert "remaining_required_minutes" in result.prompt
    log_path = tmp_path / f"{result.run_id}.json"
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["run_id"] == result.run_id
    assert log["model_name"] == "test-model:latest"
    assert log["prompt"] == result.prompt
    assert log["structured_input"]["topics"][0]["topic"] == "Recursion"
    assert log["structured_input"]["topics"][0]["required_minutes"] == 120
    assert log["structured_input"]["topics"][0]["completed_minutes"] == 0
    assert log["structured_input"]["topics"][0]["remaining_required_minutes"] == 120
    assert log["raw_initial_response"] == as_json(VALID_SCHEDULE)
    assert log["status"] == "success"
    assert log["validation_errors"] == []
    assert log["accounting_summary"]["required_remaining_minutes"] == 120
    assert log["accounting_summary"]["scheduled_minutes"] == 120
    assert log["accounting_summary"]["reported_unscheduled_minutes"] == 0
    assert log["accounting_summary"]["accounted_minutes"] == 120
    assert log["accounting_summary"]["coverage_percentage"] == 100.0
    assert log["accounting_summary"]["is_complete"] is True
    topic_row = log["accounting_summary"]["topics"][0]
    assert topic_row["course"] == "CS101"
    assert topic_row["topic"] == "Recursion"
    assert topic_row["accounting_status"] == "Complete"
    assert not Path(log["log_path"]).is_absolute()
    assert log["log_path"].endswith(f"{result.run_id}.json")


def test_malformed_json_fails_after_repair(tmp_path, monkeypatch):
    calls = install_fake_post(monkeypatch, ["not json", "still not json"])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.validation_status == "invalid"
    assert result.repair_attempted
    assert len(calls) == 2
    assert result.validation_errors
    assert any("not valid JSON" in error for error in result.validation_errors)
    assert "repair" in result.error.lower()
    assert result.accounting_summary is None


def test_hallucinated_topic_is_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["sessions"][0]["topic"] = "Invented Topic"
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("unknown course/topic" in error for error in errors)


def test_session_after_deadline_is_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["sessions"][0]["date"] = "2026-11-21"
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("after the deadline" in error for error in errors)


def test_session_before_current_date_is_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["sessions"][0]["date"] = "2026-11-14"
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("before the current date" in error for error in errors)


def test_session_earlier_on_current_day_is_rejected():
    current = datetime(2026, 11, 16, 18, 0)
    window = AvailabilityWindow(
        course_name="CS101",
        topic_name="Recursion",
        available_date=date(2026, 11, 16),
        start_time=time(15, 0),
        duration_minutes=300,
    )
    payload = build_structured_input(
        [TOPIC], [window], [], current.date(), current_datetime=current
    )
    schedule = {
        "sessions": [make_session(hour=17, duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=60)],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any("before the current date/time" in error for error in errors)


def test_session_at_exact_current_datetime_is_allowed():
    current = datetime(2026, 11, 16, 18, 0)
    payload = build_structured_input(
        [TOPIC], [WINDOW], [], current.date(), current_datetime=current
    )
    schedule = {
        "sessions": [make_session(hour=18, duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=60)],
    }
    assert validate_llm_schedule(schedule, payload) == []


def test_session_later_on_current_day_is_allowed():
    current = datetime(2026, 11, 16, 18, 0)
    payload = build_structured_input(
        [TOPIC], [WINDOW], [], current.date(), current_datetime=current
    )
    schedule = {
        "sessions": [make_session(hour=19, duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=60)],
    }
    assert validate_llm_schedule(schedule, payload) == []


def test_completed_session_before_current_datetime_is_not_time_checked():
    current = datetime(2026, 11, 16, 18, 0)
    completed = StudySession(
        session_id="completed-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 16),
        start_time=time(9, 0),
        duration_minutes=60,
        completed=True,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [completed], current.date(), current_datetime=current
    )
    schedule = {
        "sessions": [make_session(hour=18, duration=60)],
        "unscheduled_topics": [],
    }
    assert validate_llm_schedule(schedule, payload) == []


def test_structured_input_current_datetime_is_rounded_up_to_the_minute():
    payload = build_structured_input(
        [TOPIC],
        [WINDOW],
        [],
        date(2026, 11, 16),
        current_datetime=datetime(2026, 11, 16, 17, 59, 30),
    )
    assert payload["current_date"] == "2026-11-16"
    assert payload["current_datetime"] == "2026-11-16T18:00"

    payload = build_structured_input(
        [TOPIC],
        [WINDOW],
        [],
        date(2026, 11, 16),
        current_datetime=datetime(2026, 11, 16, 17, 59, 0),
    )
    assert payload["current_datetime"] == "2026-11-16T17:59"

    assert PAYLOAD["current_datetime"] == "2026-11-15T00:00"


def test_current_datetime_date_must_match_current_date():
    payload = copy.deepcopy(PAYLOAD)
    payload["current_date"] = "2026-11-16"
    errors = validate_llm_schedule(
        {"sessions": [], "unscheduled_topics": []}, payload
    )
    assert any("does not match current_date" in error for error in errors)


def test_overlapping_sessions_are_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    second = copy.deepcopy(schedule["sessions"][0])
    second["reason"] = "Overlapping duplicate."
    schedule["sessions"].append(second)
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("overlaps" in error for error in errors)


def test_unavailable_time_slot_is_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["sessions"][0]["date"] = "2026-11-17"
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("does not fit inside any available time slot" in error for error in errors)


def test_failed_connection(tmp_path, monkeypatch):
    calls = install_fake_post(monkeypatch, [requests.ConnectionError("refused")])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.validation_status == "not_available"
    assert result.repair_attempted is False
    assert len(calls) == 1
    assert "Could not reach Ollama" in result.error
    assert (tmp_path / f"{result.run_id}.json").exists()


def test_http_error_is_reported_and_logged(tmp_path, monkeypatch):
    class ErrorResponse:
        status_code = 500
        text = "server exploded"

    monkeypatch.setattr(
        llm_baseline.requests, "post", lambda *args, **kwargs: ErrorResponse()
    )
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.validation_status == "not_available"
    assert result.repair_attempted is False
    assert "HTTP 500" in result.error
    assert (tmp_path / f"{result.run_id}.json").exists()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (ValueError("not json"), "was not JSON"),
        ({"message": {}}, "did not contain message.content"),
        ({"message": {"content": 123}}, "content was not a string"),
    ],
)
def test_ollama_protocol_errors_are_reported(monkeypatch, payload, expected):
    class ProtocolErrorResponse:
        status_code = 200
        text = ""

        def json(self):
            if isinstance(payload, Exception):
                raise payload
            return payload

    monkeypatch.setattr(
        llm_baseline.requests,
        "post",
        lambda *args, **kwargs: ProtocolErrorResponse(),
    )
    with pytest.raises(OllamaProtocolError) as exc_info:
        call_ollama(CONFIG, "prompt")
    assert expected in str(exc_info.value)


def test_successful_repair_response(tmp_path, monkeypatch):
    invalid = copy.deepcopy(VALID_SCHEDULE)
    invalid["sessions"][0]["course"] = "UNKNOWN"
    calls = install_fake_post(monkeypatch, [as_json(invalid), as_json(VALID_SCHEDULE)])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert result.success
    assert result.repair_attempted
    assert len(calls) == 2
    assert result.validation_errors_initial
    assert result.validation_errors_final == []
    assert result.validation_errors == []
    log = json.loads((tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert log["validation_errors_initial"]
    assert log["validation_errors_final"] == []
    assert log["validation_errors"] == []
    repair_prompt = calls[1]["kwargs"]["json"]["messages"][0]["content"]
    assert "Validation errors" in repair_prompt
    assert "unknown course/topic" in repair_prompt


def test_failed_repair_response(tmp_path, monkeypatch):
    invalid = copy.deepcopy(VALID_SCHEDULE)
    invalid["sessions"][0]["date"] = "2026-11-21"
    calls = install_fake_post(monkeypatch, [as_json(invalid), "not json"])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.repair_attempted
    assert len(calls) == 2
    assert result.validation_errors
    assert "not valid JSON" in " ".join(result.validation_errors)
    assert "repair" in result.error.lower()
    assert result.raw_initial_response == as_json(invalid)
    assert result.raw_repair_response == "not json"
    assert result.parsed_schedule is None
    assert result.accounting_summary is None
    log = json.loads((tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert log["raw_initial_response"] == as_json(invalid)
    assert log["raw_repair_response"] == "not json"
    assert log["parsed_schedule"] is None
    assert log["accounting_summary"] is None
    assert log["validation_errors_initial"]
    assert log["validation_errors_final"]


def test_repair_connection_failure_clears_final_schedule(tmp_path, monkeypatch):
    invalid = copy.deepcopy(VALID_SCHEDULE)
    invalid["sessions"][0]["date"] = "2026-11-21"
    calls = install_fake_post(
        monkeypatch,
        [as_json(invalid), requests.exceptions.ConnectionError("simulated failure")],
    )
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.repair_attempted
    assert len(calls) == 2
    assert "Could not reach Ollama" in result.error
    assert result.raw_initial_response == as_json(invalid)
    assert result.raw_repair_response is None
    assert result.parsed_schedule is None
    assert result.accounting_summary is None
    assert result.validation_errors_initial
    log = json.loads((tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8"))
    assert log["raw_initial_response"] == as_json(invalid)
    assert log["raw_repair_response"] is None
    assert log["parsed_schedule"] is None
    assert log["accounting_summary"] is None
    assert log["validation_errors_initial"]


def test_missing_model_does_not_call_ollama(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    calls = install_fake_post(monkeypatch, [])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
    )
    assert not result.success
    assert len(calls) == 0
    assert result.model_name is None
    assert "OLLAMA_MODEL" in result.error
    assert (tmp_path / f"{result.run_id}.json").exists()


def test_invalid_planner_input_skips_ollama_and_is_logged(tmp_path, monkeypatch):
    calls = install_fake_post(monkeypatch, [])
    nan_topic = make_topic(hours=float("nan"))
    result = generate_llm_schedule(
        [nan_topic],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.validation_status == "not_available"
    assert result.repair_attempted is False
    assert len(calls) == 0
    assert "finite number" in result.error
    assert result.structured_input == {}
    log = json.loads(
        (tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8")
    )
    assert log["structured_input"] == {}
    assert log["validation_status"] == "not_available"
    assert "finite number" in log["error"]


def test_environment_model_and_default_url_are_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "configured-model:7b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/")
    calls = install_fake_post(monkeypatch, [as_json(VALID_SCHEDULE)])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
    )
    assert result.model_name == "configured-model:7b"
    assert result.base_url == "http://localhost:11434"
    assert calls[0]["kwargs"]["json"]["model"] == "configured-model:7b"
    assert calls[0]["url"] == "http://localhost:11434/api/chat"


def test_url_defaults_when_environment_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "configured-model:7b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    calls = install_fake_post(monkeypatch, [as_json(VALID_SCHEDULE)])
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
    )
    assert result.base_url == "http://localhost:11434"
    assert calls[0]["url"] == "http://localhost:11434/api/chat"


@pytest.mark.parametrize("duration", [29, 91, 60.0, True])
def test_invalid_session_duration_is_rejected(duration):
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["sessions"][0]["duration_minutes"] = duration
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("duration must be an integer from 30 to 90" in error for error in errors)


def test_overlap_with_completed_session_is_rejected():
    completed = StudySession(
        session_id="completed-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 16),
        start_time=time(18, 30),
        duration_minutes=30,
        completed=True,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [completed], CURRENT_DATE
    )
    errors = validate_llm_schedule(VALID_SCHEDULE, payload)
    assert any("completed session" in error for error in errors)


def test_unknown_unscheduled_topic_is_rejected():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["unscheduled_topics"] = [
        {
            "course": "NOPE",
            "topic": "Unknown",
            "unscheduled_minutes": 30,
            "reason": "Cannot fit.",
        }
    ]
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("unknown course/topic" in error for error in errors)


def test_json_fences_are_not_silently_stripped():
    parsed, errors = parse_llm_json("```json\n" + as_json(VALID_SCHEDULE) + "\n```")
    assert parsed is None
    assert errors


def test_llm_required_minutes_rounds_half_up_and_ignores_multiplier():
    assert llm_required_minutes(0.01) == 0
    assert llm_required_minutes(0.1) == 0
    assert llm_required_minutes(0.24) == 0
    assert llm_required_minutes(0.25) == 30
    assert llm_required_minutes(0.5) == 30
    assert llm_required_minutes(2.0) == 120
    assert llm_required_minutes(0.75) == 60
    assert llm_required_minutes(1.25) == 90
    assert llm_required_minutes(1.75) == 120
    assert isinstance(llm_required_minutes(1.25), int)

    weak = make_topic(name="Weak", hours=1.25, score=0)
    mastered = make_topic(name="Mastered", hours=1.25, score=100)
    payload = build_structured_input(
        [weak, mastered],
        [make_window(weak), make_window(mastered)],
        [],
        CURRENT_DATE,
    )
    assert [topic["required_minutes"] for topic in payload["topics"]] == [90, 90]


def test_exact_accounting_is_valid():
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=120)],
    }
    assert validate_llm_schedule(schedule, payload) == []


def test_under_accounting_is_invalid():
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=30)],
    }
    errors = validate_llm_schedule(schedule, payload)
    mismatch = next(
        error for error in errors if error.startswith("Workload mismatch")
    )
    assert "CS101 / Recursion" in mismatch
    assert "required remaining 180 minutes" in mismatch
    assert "accounts for 90 minutes" in mismatch
    assert "60 scheduled + 30 unscheduled" in mismatch


def test_over_accounting_is_invalid():
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [
            make_session(hour=18, duration=60),
            make_session(hour=19, duration=60),
        ],
        "unscheduled_topics": [make_unscheduled(minutes=90)],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any(
        "Workload mismatch" in error
        and "accounts for 210 minutes" in error
        and "required remaining 180 minutes" in error
        for error in errors
    )
    summary = build_accounting_summary(payload, schedule)
    assert summary.total_accounted_minutes == 210
    assert summary.coverage_percentage == 116.67
    assert summary.is_complete is False


def test_positive_unscheduled_entry_when_fully_scheduled_is_invalid():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    schedule["unscheduled_topics"] = [make_unscheduled(minutes=30)]
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any(
        "Workload mismatch" in error
        and "accounts for 150 minutes" in error
        and "120 scheduled + 30 unscheduled" in error
        for error in errors
    )


def test_scheduled_time_exceeding_remaining_requirement_is_invalid():
    topic = make_topic(hours=2.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [
            make_session(hour=18, duration=60),
            make_session(hour=19, duration=60),
            make_session(hour=20, duration=60),
        ],
        "unscheduled_topics": [],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any("exceeds its remaining requirement" in error for error in errors)
    assert any("Workload mismatch" in error for error in errors)


def test_missing_unscheduled_entry_is_invalid():
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [
            make_session(hour=18, duration=60),
            make_session(hour=19, duration=60),
        ],
        "unscheduled_topics": [],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any(
        "Workload mismatch" in error
        and "accounts for 120 minutes" in error
        and "0 unscheduled" in error
        for error in errors
    )


def test_duplicate_unscheduled_entries_are_invalid():
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [
            make_unscheduled(minutes=120),
            make_unscheduled(minutes=120),
        ],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any("duplicates the unscheduled entry" in error for error in errors)


@pytest.mark.parametrize(
    "minutes", [0, -30, True, False, "30", 30.0, 30.5, None]
)
def test_invalid_unscheduled_minutes_are_rejected(minutes):
    topic = make_topic(hours=3.0)
    payload = build_structured_input(
        [topic], [make_window(topic, minutes=180)], [], CURRENT_DATE
    )
    schedule = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [make_unscheduled(minutes=minutes)],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any(
        "positive integer unscheduled_minutes" in error for error in errors
    )


def test_fully_scheduled_topic_has_no_unscheduled_entry():
    assert validate_llm_schedule(copy.deepcopy(VALID_SCHEDULE), PAYLOAD) == []
    summary = build_accounting_summary(PAYLOAD, VALID_SCHEDULE)
    assert summary.is_complete
    assert summary.total_scheduled_minutes == 120
    assert summary.total_reported_unscheduled_minutes == 0
    assert summary.coverage_percentage == 100.0


def test_completed_sessions_reduce_remaining_work():
    completed = StudySession(
        session_id="completed-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 15),
        start_time=time(9, 0),
        duration_minutes=60,
        completed=True,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [completed], CURRENT_DATE
    )
    fields = payload["topics"][0]
    assert fields["required_minutes"] == 120
    assert fields["completed_minutes"] == 60
    assert fields["remaining_required_minutes"] == 60
    schedule = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [],
    }
    assert validate_llm_schedule(schedule, payload) == []


def test_incomplete_sessions_do_not_reduce_remaining_work():
    incomplete = StudySession(
        session_id="incomplete-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 15),
        start_time=time(9, 0),
        duration_minutes=60,
        completed=False,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [incomplete], CURRENT_DATE
    )
    fields = payload["topics"][0]
    assert fields["completed_minutes"] == 0
    assert fields["remaining_required_minutes"] == 120


def test_completed_minutes_clamp_remaining_to_zero():
    completed = StudySession(
        session_id="completed-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 15),
        start_time=time(9, 0),
        duration_minutes=180,
        completed=True,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [completed], CURRENT_DATE
    )
    fields = payload["topics"][0]
    assert fields["required_minutes"] == 120
    assert fields["completed_minutes"] == 180
    assert fields["remaining_required_minutes"] == 0
    assert validate_llm_schedule({"sessions": [], "unscheduled_topics": []}, payload) == []

    summary = build_accounting_summary(
        payload, {"sessions": [], "unscheduled_topics": []}
    )
    assert summary.is_complete
    assert summary.coverage_percentage == 100.0


def test_zero_remaining_topic_rejects_sessions_and_unscheduled_minutes():
    completed = StudySession(
        session_id="completed-1",
        course_name="CS101",
        topic_name="Recursion",
        session_date=date(2026, 11, 15),
        start_time=time(9, 0),
        duration_minutes=120,
        completed=True,
    )
    payload = build_structured_input(
        [TOPIC], [WINDOW], [completed], CURRENT_DATE
    )
    schedule = {
        "sessions": [make_session(duration=30)],
        "unscheduled_topics": [make_unscheduled(minutes=30)],
    }
    errors = validate_llm_schedule(schedule, payload)
    assert any("remaining requirement is 0 minutes" in error for error in errors)
    assert any(
        "Unscheduled topic" in error and "remaining requirement is 0 minutes" in error
        for error in errors
    )


def test_workload_error_triggers_one_repair_request(tmp_path, monkeypatch):
    under_accounted = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [],
    }
    calls = install_fake_post(
        monkeypatch, [as_json(under_accounted), as_json(VALID_SCHEDULE)]
    )
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert result.repair_attempted
    assert len(calls) == 2
    assert any(
        "Workload mismatch" in error for error in result.validation_errors_initial
    )
    repair_prompt = calls[1]["kwargs"]["json"]["messages"][0]["content"]
    assert "Workload mismatch" in repair_prompt
    assert (
        "generated_scheduled_minutes + unscheduled_minutes" in repair_prompt
    )


def test_repaired_exact_accounting_is_valid(tmp_path, monkeypatch):
    under_accounted = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [],
    }
    calls = install_fake_post(
        monkeypatch, [as_json(under_accounted), as_json(VALID_SCHEDULE)]
    )
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert result.success
    assert result.repair_attempted
    assert len(calls) == 2
    assert result.validation_errors_initial
    assert result.validation_errors_final == []


def test_repaired_another_accounting_mismatch_remains_failure(
    tmp_path, monkeypatch
):
    under_accounted = {
        "sessions": [make_session(duration=60)],
        "unscheduled_topics": [],
    }
    over_accounted = {
        "sessions": [
            make_session(hour=18, duration=60),
            make_session(hour=19, duration=60),
        ],
        "unscheduled_topics": [make_unscheduled(minutes=60)],
    }
    calls = install_fake_post(
        monkeypatch, [as_json(under_accounted), as_json(over_accounted)]
    )
    result = generate_llm_schedule(
        [TOPIC],
        [WINDOW],
        current_date=CURRENT_DATE,
        output_dir=tmp_path,
        config=CONFIG,
    )
    assert not result.success
    assert result.repair_attempted
    assert len(calls) == 2
    assert any(
        "Workload mismatch" in error for error in result.validation_errors_final
    )
    log = json.loads(
        (tmp_path / f"{result.run_id}.json").read_text(encoding="utf-8")
    )
    assert log["raw_initial_response"] == as_json(under_accounted)
    assert log["raw_repair_response"] == as_json(over_accounted)
    assert log["validation_errors_initial"]
    assert log["validation_errors_final"]
    assert log["accounting_summary"]["is_complete"] is False
    assert log["accounting_summary"]["accounted_minutes"] == 180


def test_accounting_summary_handles_malformed_output_without_crashing():
    schedule = {
        "sessions": [
            "not an object",
            {"course": "CS101", "topic": "Recursion", "duration_minutes": "60"},
            {"course": "CS101", "topic": "Recursion", "duration_minutes": True},
            {"course": "CS101", "topic": "Recursion", "duration_minutes": 60.0},
            {"course": "CS101", "topic": "Recursion"},
            {"course": "NOPE", "topic": "Unknown", "duration_minutes": 60},
        ],
        "unscheduled_topics": [
            "not an object",
            {
                "course": "CS101",
                "topic": "Recursion",
                "unscheduled_minutes": "30",
                "reason": "invalid type",
            },
            {
                "course": "CS101",
                "topic": "Recursion",
                "unscheduled_minutes": True,
                "reason": "invalid type",
            },
            {
                "course": "CS101",
                "topic": "Recursion",
                "unscheduled_minutes": 30.0,
                "reason": "invalid type",
            },
            {
                "course": "NOPE",
                "topic": "Unknown",
                "unscheduled_minutes": 30,
                "reason": "unknown",
            },
            {
                "course": "CS101",
                "topic": "Recursion",
                "reason": "missing minutes",
            },
            {
                "course": "CS101",
                "topic": "Recursion",
                "unscheduled_minutes": "30",
                "reason": "duplicate invalid value",
            },
        ],
    }
    summary = build_accounting_summary(PAYLOAD, schedule)
    assert summary.total_scheduled_minutes == 0
    assert summary.total_reported_unscheduled_minutes == 0
    assert summary.total_accounted_minutes == 0
    assert summary.is_complete is False
    assert summary.has_invalid_values is True
    assert summary.coverage_percentage is None

    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert errors


def test_invalid_structured_workload_fields_are_handled_safely():
    payload = copy.deepcopy(PAYLOAD)
    payload["topics"][0]["required_minutes"] = "120"
    payload["topics"][0]["remaining_required_minutes"] = 120.0
    summary = build_accounting_summary(
        payload, {"sessions": [], "unscheduled_topics": []}
    )
    assert summary.has_invalid_values is True
    assert summary.is_complete is False
    assert summary.coverage_percentage is None
    assert summary.to_dict()["coverage_percentage"] is None
    assert summary.topics[0].is_complete is False
    assert summary.topics[0].accounting_status == "Invalid accounting input"
    errors = validate_llm_schedule(
        {"sessions": [], "unscheduled_topics": []}, payload
    )
    assert any("missing valid" in error for error in errors)


def test_same_topic_name_in_different_courses_is_accounted_separately():
    first = make_topic(course="CS101", name="Recursion", hours=1.0)
    second = make_topic(course="CS102", name="Recursion", hours=1.0)
    payload = build_structured_input(
        [first, second],
        [make_window(first), make_window(second)],
        [],
        CURRENT_DATE,
    )
    schedule = {
        "sessions": [
            make_session(
                course="CS101", name="Recursion", hour=18, duration=60
            ),
            make_session(
                course="CS102", name="Recursion", hour=20, duration=60
            ),
        ],
        "unscheduled_topics": [],
    }
    assert validate_llm_schedule(schedule, payload) == []
    summary = build_accounting_summary(payload, schedule)
    assert summary.total_scheduled_minutes == 120
    assert [row.course for row in summary.topics] == ["CS101", "CS102"]


def test_accounting_summary_does_not_override_validation():
    schedule = copy.deepcopy(VALID_SCHEDULE)
    for session in schedule["sessions"]:
        session["date"] = "2026-11-21"
    summary = build_accounting_summary(PAYLOAD, schedule)
    assert summary.is_complete is True
    errors = validate_llm_schedule(schedule, PAYLOAD)
    assert any("after the deadline" in error for error in errors)


def test_empty_topics_are_rejected_and_not_complete():
    errors = validate_llm_schedule(
        {"sessions": [], "unscheduled_topics": []},
        {"current_date": "2026-11-17", "topics": []},
    )
    assert errors
    assert any("does not contain any topics" in error for error in errors)

    summary = build_accounting_summary(
        {"topics": []}, {"sessions": [], "unscheduled_topics": []}
    )
    assert summary.is_complete is False
    assert summary.coverage_percentage is None
    assert summary.to_dict()["coverage_percentage"] is None


def test_topics_without_any_valid_entry_are_rejected():
    payload = {
        "current_date": "2026-11-17",
        "topics": [{"course": 1, "topic": 2}],
    }
    errors = validate_llm_schedule(
        {"sessions": [], "unscheduled_topics": []}, payload
    )
    assert errors
    assert any("does not contain any valid topics" in error for error in errors)

    summary = build_accounting_summary(
        payload, {"sessions": [], "unscheduled_topics": []}
    )
    assert summary.is_complete is False
    assert summary.coverage_percentage is None
