"""Streamlit application for the adaptive planner and LLM baseline."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from datetime import date, datetime, time, timedelta

import pandas as pd
import streamlit as st

from baseline import generate_baseline_schedule
from config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_OLLAMA_BASE_URL,
    LLM_RUN_DIR,
    OLLAMA_BASE_URL_ENV,
    OLLAMA_MODEL_ENV,
    PROJECT_ROOT,
)
from llm_baseline import generate_llm_schedule
from models import AvailabilityWindow, ScheduleResult, StudySession, Topic
from priority import (
    adjusted_study_minutes,
    calculate_priority,
    format_priority,
    knowledge_category,
    study_multiplier,
)
from scheduler import generate_adaptive_schedule
from validation import (
    format_errors,
    load_availability_csv,
    load_topics_csv,
    validate_availability,
    validate_dataset,
    validate_topic,
)

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def _init_state() -> None:
    defaults = {
        "topics": [],
        "availability": [],
        "sessions": [],
        "adaptive_result": None,
        "baseline_result": None,
        "before_snapshot": None,
        "after_snapshot": None,
        "llm_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _invalidate_schedule(clear_llm: bool = True) -> None:
    st.session_state.sessions = []
    st.session_state.adaptive_result = None
    st.session_state.baseline_result = None
    st.session_state.before_snapshot = None
    st.session_state.after_snapshot = None
    if clear_llm:
        st.session_state.llm_result = None


def _load_sample_data() -> None:
    topics_text = (PROJECT_ROOT / "sample_topics.csv").read_text(encoding="utf-8")
    availability_text = (PROJECT_ROOT / "sample_availability.csv").read_text(
        encoding="utf-8"
    )
    topics, topic_errors = load_topics_csv(topics_text)
    availability, availability_errors = load_availability_csv(
        availability_text, topics
    )
    errors = topic_errors + availability_errors
    if errors:
        raise ValueError(format_errors(errors))
    st.session_state.topics = topics
    st.session_state.availability = availability
    _invalidate_schedule()


def _topic_options(topics: list[Topic]) -> dict[str, Topic]:
    return {topic.label: topic for topic in topics}


def _topic_overview(topics: list[Topic]) -> pd.DataFrame:
    today = date.today()
    records = []
    for topic in topics:
        records.append(
            {
                "Course": topic.course_name,
                "Topic": topic.topic_name,
                "Deadline": topic.deadline.isoformat(),
                "Est. hours": topic.estimated_hours,
                "Quiz": topic.quiz_score,
                "Weight": topic.task_weight,
                "Category": knowledge_category(topic.quiz_score),
                "Multiplier": study_multiplier(topic.quiz_score),
                "Adjusted min": adjusted_study_minutes(
                    topic.estimated_hours, topic.quiz_score
                ),
                "Priority": format_priority(
                    _priority_for_topic(topic, today)
                ),
            }
        )
    return pd.DataFrame(records)


def _priority_for_topic(topic: Topic, today: date) -> float:
    return calculate_priority(topic, today)


def _availability_frame(availability: list[AvailabilityWindow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Course": window.course_name,
                "Topic": window.topic_name,
                "Available date": window.available_date.isoformat(),
                "Start": window.start_time.strftime("%H:%M"),
                "Minutes": window.duration_minutes,
            }
            for window in availability
        ]
    )


def _is_past_incomplete(session: StudySession, now: datetime) -> bool:
    return not session.completed and session.end_datetime < now


def _session_rows(
    result: ScheduleResult, now: datetime, include_id: bool = False
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for session in result.sessions:
        status = session.status_label
        if _is_past_incomplete(session, now):
            status = "Past incomplete"
        row: dict[str, object] = {}
        if include_id:
            row["session_id"] = session.session_id
        row.update(
            {
                "Completed": session.completed,
                "Course": session.course_name,
                "Topic": session.topic_name,
                "Date": session.session_date.isoformat(),
                "Start": session.start_time.strftime("%H:%M"),
                "End": session.end_datetime.strftime("%H:%M"),
                "Minutes": session.duration_minutes,
                "Status": status,
                "Source": session.source.title(),
                "Reason": session.reason,
            }
        )
        rows.append(row)
    return rows


def _session_frame(result: ScheduleResult, now: datetime) -> pd.DataFrame:
    return pd.DataFrame(_session_rows(result, now))


def _editable_session_frame(result: ScheduleResult, now: datetime) -> pd.DataFrame:
    return pd.DataFrame(_session_rows(result, now, include_id=True))


def _plan_frame(result: ScheduleResult) -> pd.DataFrame:
    rows = []
    for plan in result.topic_plans:
        rows.append(
            {
                "Course": plan.topic.course_name,
                "Topic": plan.topic.topic_name,
                "Deadline": plan.topic.deadline.isoformat(),
                "Category": plan.category or "—",
                "Priority": format_priority(plan.priority),
                "Adjusted min": plan.required_minutes,
                "Completed min": plan.completed_minutes,
                "Remaining min": plan.remaining_minutes,
                "Scheduled min": plan.scheduled_minutes,
                "Unscheduled min": plan.unscheduled_minutes,
            }
        )
    return pd.DataFrame(rows)


def _show_errors(errors: list[str]) -> None:
    if errors:
        st.error("Please fix these issues:")
        st.code(format_errors(errors), language=None)


def _show_warnings(result: ScheduleResult) -> None:
    if result.warnings:
        with st.expander("Schedule warnings and unscheduled time", expanded=True):
            for warning in result.warnings:
                st.warning(warning)


def _run_adaptive() -> bool:
    errors = validate_dataset(st.session_state.topics, st.session_state.availability)
    if errors:
        _show_errors(errors)
        return False
    result = generate_adaptive_schedule(
        topics=st.session_state.topics,
        availability=st.session_state.availability,
        now=datetime.now(),
        preserved_sessions=st.session_state.sessions,
    )
    st.session_state.adaptive_result = result
    st.session_state.sessions = result.sessions
    st.session_state.baseline_result = None
    st.session_state.before_snapshot = None
    st.session_state.after_snapshot = None
    st.session_state.llm_result = None
    return True


def _apply_completion_updates(edited: pd.DataFrame) -> bool:
    result = st.session_state.adaptive_result
    if result is None:
        return False
    sessions_by_id = {session.session_id: session for session in result.sessions}
    updated_sessions: list[StudySession] = []
    for record in edited.to_dict("records"):
        session = sessions_by_id.get(str(record.get("session_id")))
        if session is None:
            continue
        session.completed = bool(record.get("Completed"))
        updated_sessions.append(session)
    if not updated_sessions:
        return False

    errors = validate_dataset(st.session_state.topics, st.session_state.availability)
    if errors:
        _show_errors(errors)
        return False
    new_result = generate_adaptive_schedule(
        topics=st.session_state.topics,
        availability=st.session_state.availability,
        now=datetime.now(),
        preserved_sessions=updated_sessions,
    )
    st.session_state.adaptive_result = new_result
    st.session_state.sessions = new_result.sessions
    st.session_state.baseline_result = None
    st.session_state.before_snapshot = None
    st.session_state.after_snapshot = None
    st.session_state.llm_result = None
    return True


def _render_schedule_section(result: ScheduleResult) -> None:
    now = datetime.now()
    st.subheader("Schedule")
    st.caption(
        "Completed sessions are preserved. Future incomplete sessions may be "
        "regenerated when quiz scores or completion status change."
    )
    schedule_df = _editable_session_frame(result, now)
    if schedule_df.empty:
        st.info("No sessions are currently scheduled.")
        return
    edited = st.data_editor(
        schedule_df,
        hide_index=True,
        use_container_width=True,
        key="completion_editor",
        disabled=[
            "session_id",
            "Course",
            "Topic",
            "Date",
            "Start",
            "End",
            "Minutes",
            "Status",
            "Source",
            "Reason",
        ],
        column_config={
            "session_id": st.column_config.TextColumn("Session ID", width="small"),
            "Completed": st.column_config.CheckboxColumn("Completed"),
        },
    )
    if st.button("Save completion status and regenerate", key="save_completion"):
        if _apply_completion_updates(edited):
            st.success(
                "Completion status saved. Completed sessions were preserved and "
                "future incomplete sessions were regenerated."
            )
            result = st.session_state.adaptive_result
            st.rerun()

    download_frame = _session_frame(result, now)
    st.download_button(
        "Download current schedule CSV",
        data=download_frame.to_csv(index=False).encode("utf-8"),
        file_name="adaptive_schedule.csv",
        mime="text/csv",
        key="download_adaptive_csv",
    )


def _comparison_frame(before: ScheduleResult, after: ScheduleResult) -> pd.DataFrame:
    before_plans = {plan.topic.key: plan for plan in before.topic_plans}
    after_plans = {plan.topic.key: plan for plan in after.topic_plans}
    keys = list(after_plans) + [key for key in before_plans if key not in after_plans]
    rows = []
    for key in keys:
        before_plan = before_plans.get(key)
        after_plan = after_plans.get(key)
        topic = (after_plan or before_plan).topic
        rows.append(
            {
                "Course": topic.course_name,
                "Topic": topic.topic_name,
                "Score before": before_plan.topic.quiz_score if before_plan else "",
                "Score after": after_plan.topic.quiz_score if after_plan else "",
                "Category before": before_plan.category if before_plan else "",
                "Category after": after_plan.category if after_plan else "",
                "Adjusted before": before_plan.required_minutes if before_plan else "",
                "Adjusted after": after_plan.required_minutes if after_plan else "",
                "Priority before": format_priority(before_plan.priority) if before_plan else "",
                "Priority after": format_priority(after_plan.priority) if after_plan else "",
                "Remaining before": before_plan.remaining_minutes if before_plan else "",
                "Remaining after": after_plan.remaining_minutes if after_plan else "",
            }
        )
    return pd.DataFrame(rows)


def _render_llm_result() -> None:
    result = st.session_state.llm_result
    if result is None:
        return

    if result.success:
        st.success("Valid LLM-generated schedule. All required constraints passed.")
    else:
        st.error(
            "LLM run failed. The output is not a valid schedule and must not be "
            "treated as one."
        )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Validation", result.validation_status)
    metric_cols[1].metric("Repair attempted", "Yes" if result.repair_attempted else "No")
    metric_cols[2].metric("Model", result.model_name or "not configured")
    metric_cols[3].metric("Runtime", f"{result.runtime_seconds:.2f}s")
    metric_cols[4].metric("Status", result.status)

    if result.error:
        st.error(result.error)

    if result.repair_attempted:
        st.info(
            "The first response failed validation, so exactly one repair request "
            "was sent. The repaired response was validated again."
        )
        if result.validation_errors_initial:
            with st.expander("Initial validation errors", expanded=False):
                for error in result.validation_errors_initial:
                    st.write(f"- {error}")

    if result.validation_errors_final:
        st.subheader("Constraint violations")
        for error in result.validation_errors_final:
            st.error(error)
    elif result.success:
        st.caption("Raw validation status: valid JSON and all constraints satisfied.")

    if result.success and result.parsed_schedule is not None:
        sessions = result.parsed_schedule.get("sessions", [])
        if sessions:
            st.subheader("LLM-generated sessions")
            st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
        unscheduled = result.parsed_schedule.get("unscheduled_topics", [])
        if unscheduled:
            st.subheader("LLM-reported unscheduled topics")
            st.dataframe(pd.DataFrame(unscheduled), use_container_width=True, hide_index=True)
    else:
        st.warning(
            "Raw model output is shown below for debugging only. It is not displayed "
            "as a valid schedule."
        )

    if result.raw_initial_response is not None:
        with st.expander("Raw initial model response", expanded=False):
            st.code(result.raw_initial_response, language="json")
    if result.raw_repair_response is not None:
        with st.expander("Raw repair model response", expanded=False):
            st.code(result.raw_repair_response, language="json")

    st.download_button(
        "Download LLM run result as JSON",
        data=json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"llm_run_{result.run_id}.json",
        mime="application/json",
        key="download_llm_result",
    )
    if result.log_path:
        st.caption(f"Run log: {result.log_path}")
    if result.logging_error:
        st.error(result.logging_error)


_init_state()

st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "Local, rule-based adaptive planning with a strict local-Ollama LLM baseline. "
    "Use synthetic data only."
)

setup_tab, adaptive_tab, update_tab, baseline_tab, llm_tab = st.tabs(
    [
        "Setup",
        "Adaptive Schedule",
        "Update Quiz",
        "Baseline Comparison",
        "LLM Baseline",
    ]
)

with setup_tab:
    st.subheader("Synthetic data")
    action_cols = st.columns(4)
    if action_cols[0].button("Load bundled sample data", key="load_sample"):
        try:
            _load_sample_data()
            st.success("Sample topics and availability loaded.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if action_cols[1].button("Clear all data", key="clear_all"):
        st.session_state.topics = []
        st.session_state.availability = []
        _invalidate_schedule()
        st.success("All data cleared.")
        st.rerun()

    sample_topics_bytes = (PROJECT_ROOT / "sample_topics.csv").read_bytes()
    sample_availability_bytes = (PROJECT_ROOT / "sample_availability.csv").read_bytes()
    action_cols[2].download_button(
        "Download topic template",
        data=sample_topics_bytes,
        file_name="sample_topics.csv",
        mime="text/csv",
        key="download_topic_template",
    )
    action_cols[3].download_button(
        "Download availability template",
        data=sample_availability_bytes,
        file_name="sample_availability.csv",
        mime="text/csv",
        key="download_availability_template",
    )

    st.markdown("#### Add a topic")
    with st.form("add_topic_form"):
        col1, col2, col3 = st.columns(3)
        course_name = col1.text_input("Course name", key="course_name_input")
        topic_name = col2.text_input("Topic name", key="topic_name_input")
        deadline = col3.date_input(
            "Deadline", value=date.today() + timedelta(days=7), key="deadline_input"
        )
        col4, col5, col6 = st.columns(3)
        estimated_hours = col4.number_input(
            "Estimated study hours",
            min_value=0.0,
            max_value=1000.0,
            value=2.0,
            step=0.5,
            key="hours_input",
        )
        quiz_score = col5.number_input(
            "Quiz score (0-100)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key="quiz_input",
        )
        task_weight = col6.number_input(
            "Task weight (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            key="weight_input",
        )
        add_topic = st.form_submit_button("Add topic")
    if add_topic:
        topic = Topic(
            course_name=course_name,
            topic_name=topic_name,
            deadline=deadline,
            estimated_hours=float(estimated_hours),
            quiz_score=int(quiz_score),
            task_weight=float(task_weight),
        )
        errors = validate_topic(topic)
        if topic.key in {existing.key for existing in st.session_state.topics}:
            errors.append(f"Duplicate topic: {topic.course_name} / {topic.topic_name}.")
        if errors:
            _show_errors(errors)
        else:
            st.session_state.topics.append(topic)
            _invalidate_schedule()
            st.success(f"Added {topic.label}.")
            st.rerun()

    st.markdown("#### Add an availability window")
    topic_lookup = _topic_options(st.session_state.topics)
    if not topic_lookup:
        st.info("Add at least one topic before adding availability.")
    else:
        with st.form("add_availability_form"):
            selected_label = st.selectbox(
                "Topic", options=list(topic_lookup), key="availability_topic_input"
            )
            col1, col2, col3 = st.columns(3)
            available_date = col1.date_input(
                "Available date",
                value=date.today() + timedelta(days=1),
                key="availability_date_input",
            )
            start_time = col2.time_input(
                "Start time", value=time(18, 0), key="availability_start_input"
            )
            duration_minutes = col3.number_input(
                "Available duration (minutes)",
                min_value=30,
                max_value=720,
                value=120,
                step=30,
                key="availability_duration_input",
            )
            add_window = st.form_submit_button("Add availability")
        if add_window:
            selected = topic_lookup[selected_label]
            window = AvailabilityWindow(
                course_name=selected.course_name,
                topic_name=selected.topic_name,
                available_date=available_date,
                start_time=start_time,
                duration_minutes=int(duration_minutes),
            )
            errors = validate_availability(window, st.session_state.topics)
            if errors:
                _show_errors(errors)
            else:
                st.session_state.availability.append(window)
                _invalidate_schedule()
                st.success("Availability window added.")
                st.rerun()

    with st.expander("Import CSV files", expanded=False):
        st.caption(
            "Topic columns: course_name, topic_name, deadline, estimated_hours, "
            "quiz_score, task_weight. Availability columns: course_name, topic_name, "
            "available_date, start_time, duration_minutes."
        )
        topics_file = st.file_uploader(
            "Topics CSV", type=["csv"], key="topics_csv_upload"
        )
        if st.button("Import topics CSV", key="import_topics"):
            if topics_file is None:
                st.error("Choose a topics CSV first.")
            else:
                topics, errors = load_topics_csv(topics_file.getvalue())
                if errors:
                    _show_errors(errors)
                else:
                    st.session_state.topics = topics
                    st.session_state.availability = []
                    _invalidate_schedule()
                    st.success(f"Imported {len(topics)} topics.")
                    st.rerun()

        availability_file = st.file_uploader(
            "Availability CSV", type=["csv"], key="availability_csv_upload"
        )
        if st.button("Import availability CSV", key="import_availability"):
            if availability_file is None:
                st.error("Choose an availability CSV first.")
            elif not st.session_state.topics:
                st.error("Import or add topics before importing availability.")
            else:
                windows, errors = load_availability_csv(
                    availability_file.getvalue(), st.session_state.topics
                )
                if errors:
                    _show_errors(errors)
                else:
                    st.session_state.availability = windows
                    _invalidate_schedule()
                    st.success(f"Imported {len(windows)} availability windows.")
                    st.rerun()

    st.subheader("Current topics")
    if st.session_state.topics:
        st.dataframe(
            _topic_overview(st.session_state.topics),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No topics loaded.")
    st.subheader("Current availability")
    if st.session_state.availability:
        st.dataframe(
            _availability_frame(st.session_state.availability),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No availability windows loaded.")

with adaptive_tab:
    st.subheader("Generate the adaptive schedule")
    st.caption(
        "Topics are sorted by descending priority, then earlier deadline, then lower "
        "quiz score. Sessions go to the earliest available slot before the deadline."
    )
    if st.button("Generate adaptive schedule", type="primary", key="generate_adaptive"):
        if _run_adaptive():
            st.success("Adaptive schedule generated.")
    result = st.session_state.adaptive_result
    if result is not None:
        metric_cols = st.columns(4)
        metric_cols[0].metric("New minutes scheduled", result.total_scheduled_minutes)
        metric_cols[1].metric("Unscheduled minutes", result.total_unscheduled_minutes)
        metric_cols[2].metric("Completed minutes", result.total_completed_minutes)
        metric_cols[3].metric("Sessions in view", len(result.sessions))
        _show_warnings(result)
        st.subheader("Topic plans")
        st.dataframe(_plan_frame(result), use_container_width=True, hide_index=True)
        _render_schedule_section(result)
    else:
        st.info("Load some synthetic data and generate a schedule.")

with update_tab:
    st.subheader("Adaptive quiz-score update")
    if st.session_state.adaptive_result is None:
        st.info("Generate an adaptive schedule before updating a quiz score.")
    elif not st.session_state.topics:
        st.info("No topics are available.")
    else:
        topic_lookup = _topic_options(st.session_state.topics)
        selected_label = st.selectbox(
            "Topic to update", options=list(topic_lookup), key="update_topic_select"
        )
        selected_topic = topic_lookup[selected_label]
        st.caption(
            f"Current score: {selected_topic.quiz_score} · "
            f"Category: {knowledge_category(selected_topic.quiz_score)} · "
            f"Multiplier: {study_multiplier(selected_topic.quiz_score)}"
        )
        new_score = st.number_input(
            "New quiz score (0-100)",
            min_value=0,
            max_value=100,
            value=selected_topic.quiz_score,
            step=1,
            key=f"new_score_{selected_topic.course_name}_{selected_topic.topic_name}",
        )
        if st.button("Update score and regenerate future sessions", key="update_score"):
            before_result = copy.deepcopy(st.session_state.adaptive_result)
            before_topics = copy.deepcopy(st.session_state.topics)
            new_topics = [
                replace(topic, quiz_score=int(new_score))
                if topic.key == selected_topic.key
                else topic
                for topic in st.session_state.topics
            ]
            new_result = generate_adaptive_schedule(
                topics=new_topics,
                availability=st.session_state.availability,
                now=datetime.now(),
                preserved_sessions=st.session_state.sessions,
            )
            st.session_state.before_snapshot = {
                "topics": before_topics,
                "result": before_result,
            }
            st.session_state.after_snapshot = {
                "topics": copy.deepcopy(new_topics),
                "result": new_result,
            }
            st.session_state.topics = new_topics
            st.session_state.adaptive_result = new_result
            st.session_state.sessions = new_result.sessions
            st.session_state.baseline_result = None
            st.session_state.llm_result = None
            st.success(
                "Quiz score updated. Completed sessions were preserved and future "
                "incomplete sessions were regenerated."
            )

        before_snapshot = st.session_state.before_snapshot
        after_snapshot = st.session_state.after_snapshot
        if before_snapshot and after_snapshot:
            st.markdown("#### Before and after")
            st.dataframe(
                _comparison_frame(
                    before_snapshot["result"], after_snapshot["result"]
                ),
                use_container_width=True,
                hide_index=True,
            )
            before_col, after_col = st.columns(2)
            with before_col:
                st.markdown("**Before**")
                st.caption(
                    f"Scheduled {before_snapshot['result'].total_scheduled_minutes} min · "
                    f"Unscheduled {before_snapshot['result'].total_unscheduled_minutes} min"
                )
                st.dataframe(
                    _session_frame(before_snapshot["result"], datetime.now()),
                    use_container_width=True,
                    hide_index=True,
                )
            with after_col:
                st.markdown("**After**")
                st.caption(
                    f"Scheduled {after_snapshot['result'].total_scheduled_minutes} min · "
                    f"Unscheduled {after_snapshot['result'].total_unscheduled_minutes} min"
                )
                st.dataframe(
                    _session_frame(after_snapshot["result"], datetime.now()),
                    use_container_width=True,
                    hide_index=True,
                )

with baseline_tab:
    st.subheader("Deadline-only non-adaptive baseline")
    st.caption(
        "The baseline ignores quiz scores, task weights, knowledge categories, "
        "adaptive multipliers, and completed status. It uses original estimated "
        "study hours and sorts only by deadline."
    )
    if st.button("Generate baseline schedule", key="generate_baseline"):
        if not st.session_state.topics:
            st.error("Load topics before generating the baseline.")
        else:
            errors = validate_dataset(
                st.session_state.topics, st.session_state.availability
            )
            if errors:
                _show_errors(errors)
            else:
                st.session_state.baseline_result = generate_baseline_schedule(
                    topics=st.session_state.topics,
                    availability=st.session_state.availability,
                    now=datetime.now(),
                )
                st.success("Baseline schedule generated.")

    adaptive = st.session_state.adaptive_result
    baseline = st.session_state.baseline_result
    if adaptive is None:
        st.info("Generate an adaptive schedule first to compare against the baseline.")
    elif baseline is None:
        st.info("Generate the baseline schedule to compare the two approaches.")
    else:
        st.markdown("#### Comparison metrics")
        comparison = pd.DataFrame(
            [
                {
                    "Metric": "Scheduled minutes",
                    "Adaptive": adaptive.total_scheduled_minutes,
                    "Baseline": baseline.total_scheduled_minutes,
                },
                {
                    "Metric": "Unscheduled minutes",
                    "Adaptive": adaptive.total_unscheduled_minutes,
                    "Baseline": baseline.total_unscheduled_minutes,
                },
                {
                    "Metric": "Completed minutes",
                    "Adaptive": adaptive.total_completed_minutes,
                    "Baseline": 0,
                },
                {
                    "Metric": "Session rows",
                    "Adaptive": len(adaptive.sessions),
                    "Baseline": len(baseline.sessions),
                },
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        adaptive_col, baseline_col = st.columns(2)
        with adaptive_col:
            st.markdown("**Adaptive**")
            if adaptive.warnings:
                st.caption("Warnings: " + " | ".join(adaptive.warnings))
            st.dataframe(
                _session_frame(adaptive, datetime.now()),
                use_container_width=True,
                hide_index=True,
            )
        with baseline_col:
            st.markdown("**Deadline-only baseline**")
            if baseline.warnings:
                st.caption("Warnings: " + " | ".join(baseline.warnings))
            st.dataframe(
                _session_frame(baseline, datetime.now()),
                use_container_width=True,
                hide_index=True,
            )

with llm_tab:
    st.subheader("Local Ollama LLM baseline")
    st.warning(
        "Use synthetic data only. This page sends the serialized planner input to "
        "the local Ollama URL configured below. It does not use an external service."
    )
    model_name = os.environ.get(OLLAMA_MODEL_ENV, "").strip()
    base_url = (
        os.environ.get(OLLAMA_BASE_URL_ENV, "").strip() or DEFAULT_OLLAMA_BASE_URL
    )
    config_cols = st.columns(3)
    config_cols[0].metric("OLLAMA_MODEL", model_name or "not set")
    config_cols[1].metric("OLLAMA_BASE_URL", base_url)
    config_cols[2].metric("Temperature", "0")
    if not model_name:
        st.error(
            "OLLAMA_MODEL is not set. Set it to an installed local Ollama model. "
            "The app will not choose a model automatically."
        )

    if not st.session_state.topics:
        st.info("No planner input is loaded. Load the bundled synthetic sample data.")
        if st.button("Load sample data for LLM baseline", key="load_sample_llm"):
            try:
                _load_sample_data()
                st.success("Sample data loaded.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.caption(
        "The LLM receives the same synthetic topics, availability, current date, and "
        "completed sessions as the adaptive planner. Invalid output receives one "
        "repair request and is never replaced by a rule-based schedule."
    )
    if st.button(
        "Generate LLM schedule",
        type="primary",
        key="generate_llm",
        disabled=not st.session_state.topics,
    ):
        with st.spinner("Calling local Ollama..."):
            st.session_state.llm_result = generate_llm_schedule(
                topics=st.session_state.topics,
                availability=st.session_state.availability,
                current_date=date.today(),
                completed_sessions=[
                    session
                    for session in st.session_state.sessions
                    if session.completed
                ],
                output_dir=LLM_RUN_DIR,
            )
    _render_llm_result()

st.divider()
st.caption(
    "No external API, authentication, database, calendar integration, or external "
    "LLM is used. LLM runs are stored under sample_outputs/llm_runs."
)
