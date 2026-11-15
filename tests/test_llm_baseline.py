import copy
import json
from datetime import date, time

import pytest
import requests

import llm_baseline
from llm_baseline import (
    OllamaConfig,
    build_structured_input,
    generate_llm_schedule,
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
        }
    ],
    "unscheduled_topics": [],
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
    log_path = tmp_path / f"{result.run_id}.json"
    assert log_path.exists()
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert log["run_id"] == result.run_id
    assert log["model_name"] == "test-model:latest"
    assert log["prompt"] == result.prompt
    assert log["structured_input"]["topics"][0]["topic"] == "Recursion"
    assert log["raw_initial_response"] == as_json(VALID_SCHEDULE)
    assert log["status"] == "success"
    assert log["validation_errors"] == []


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
