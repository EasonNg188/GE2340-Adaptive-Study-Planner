# Adaptive Study Planner with Local LLM Baseline

A local Python/Streamlit study-planning project with:

- a deterministic rule-based adaptive planner,
- a deadline-only non-adaptive baseline,
- a strict LLM scheduling baseline that calls a local Ollama model at runtime,
- CSV import/export and synthetic sample data,
- unit tests that do not require a real language model.

No external API, authentication, database, calendar integration, or external LLM
service is used. All sample data is synthetic.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface for setup, adaptive scheduling, quiz updates, baseline comparison, and the LLM Baseline page |
| `config.py` | Shared constants, thresholds, session limits, CSV columns, and Ollama defaults |
| `models.py` | Dataclasses for topics, availability windows, study sessions, topic plans, and schedule results |
| `validation.py` | Input and CSV validation |
| `priority.py` | Knowledge level, multiplier, urgency, priority, rounding, and session decomposition |
| `scheduler.py` | Rule-based priority ordering and availability allocation |
| `baseline.py` | Deadline-only non-adaptive baseline |
| `llm_baseline.py` | Ollama HTTP client, workload accounting, prompt construction, strict validation, one repair request, and run logging |
| `prompts/llm_scheduler_prompt.txt` | LLM scheduling prompt with strict JSON schema and workload-accounting rules |
| `sample_topics.csv` | Synthetic topics (2027 sample) |
| `sample_availability.csv` | Synthetic availability windows (2027 sample) |
| `sample_data_2026/` | Archived 2026 sample topics and availability windows |
| `sample_outputs/` | Runtime LLM logs, plus an explanation of the log format |
| `tests/` | Unit, scheduling, baseline, validation, LLM-mock, and app smoke tests |
| `requirements.txt` | Python dependencies |

## Setup

Python 3.12 is the tested version.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands in this README call `.\.venv\Scripts\python.exe` directly instead
of activating the environment. This avoids the Windows PowerShell
execution-policy error `Activate.ps1 cannot be loaded because running scripts is
disabled on this system`. If you prefer activation, allow it for the current
window only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open the local URL Streamlit prints. The **Setup** tab can load the bundled
synthetic sample data.

The sample deadlines are in November/December 2027. If you run the project after
those dates, update the sample CSVs or use future dates so availability and
deadlines are in the future.

## Rule-based planner

Knowledge levels:

- Weak: quiz score 0–59, study-time multiplier 1.5
- Developing: quiz score 60–79, multiplier 1.0
- Mastered: quiz score 80–100, multiplier 0.5

Priority:

```text
weakness = (100 - quiz_score) / 100
urgency = 1 if days_to_deadline <= 0
          otherwise 1 - min(days_to_deadline, 30) / 30
normalized_task_weight = task_weight / 100

priority = 100 * (
    0.5 * weakness
    + 0.3 * urgency
    + 0.2 * normalized_task_weight
)
```

The displayed priority is rounded half-up to two decimal places. Sorting uses the
unrounded priority, then earlier deadline, then lower quiz score, then input order.

Session rules:

- minimum 30 minutes
- default 60 minutes
- maximum 90 minutes
- adjusted time is rounded half-up to the nearest 30 minutes
- 60-minute sessions are created first, with one 30-minute remainder
- a 90-minute requirement becomes 60 + 30
- zero estimated time produces no sessions

Scheduling:

- descending priority
- earliest feasible availability before the deadline
- one session per topic per day normally
- more than one session per topic per day only for work that cannot otherwise fit
- availability windows are explicit and do not recur automatically
- a session must fit entirely within one availability window
- unscheduled minutes and overdue deadlines are reported
- deadline day is allowed through end of day

Completed sessions are preserved. Updating a quiz score recalculates category,
multiplier, adjusted time, priority, and remaining time. Future incomplete sessions
are regenerated. Past incomplete sessions remain visible as history but do not
satisfy remaining time. Remaining time is clamped at zero.

The deadline-only baseline ignores quiz scores, task weights, knowledge categories,
adaptive multipliers, and completion status. It uses original estimated study hours
and sorts by deadline only, with input order for ties.

## Data formats

`topics.csv`:

```csv
course_name,topic_name,deadline,estimated_hours,quiz_score,task_weight
CS101,Recursion,2026-12-05,3.0,45,80
```

`availability.csv`:

```csv
course_name,topic_name,available_date,start_time,duration_minutes
CS101,Recursion,2026-11-16,18:00,120
```

A topic is identified by the combination of course and topic name. A topic may have
multiple availability windows. Availability windows may not cross midnight.

## Local Ollama LLM baseline

The LLM baseline is an experimental comparison. It does **not** replace the
rule-based scheduler and does not fall back to rule-based output when the model or
validation fails.

The LLM baseline does **not** use the adaptive quiz-score study-time multipliers.
Its required workload is based on the original estimated study hours for each
topic.

### 1. Install and start Ollama

Install Ollama from the official download page: <https://ollama.com/download>.

To install the program and models outside the default user profile on Windows:

```powershell
OllamaSetup.exe /VERYSILENT /NORESTART /SUPPRESSMSGBOXES /DIR="D:\Ollama"
setx OLLAMA_MODELS "D:\Ollama\models"
```

`setx` applies to new terminals; set `$env:OLLAMA_MODELS` in the current window
if you start Ollama immediately.

On Windows the installer starts the local service automatically. Verify it:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If that fails, start the service manually:

```powershell
ollama serve
```

Do not run `ollama serve` while the Windows app is already running; it fails
with a port-in-use error (`bind: Only one usage of each socket address`). In
another terminal, verify the model is installed:

```powershell
ollama list
```

### 2. Pull and configure a model

Download a local model, for example:

```powershell
ollama pull llama3.2:3b
```

The code does not choose a model automatically. Set the exact model name/tag:

```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

On macOS/Linux:

```bash
export OLLAMA_MODEL="llama3.2:3b"
export OLLAMA_BASE_URL="http://localhost:11434"
```

`OLLAMA_BASE_URL` defaults to `http://localhost:11434` if unset.
`OLLAMA_MODEL` has no default and is required.

Run Streamlit from the same terminal/environment so it can see the variables:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The model request uses `POST /api/chat`, `temperature: 0` where supported, and
`format: "json"`. The exact model name, URL, and parameters are recorded in each
run log.

### 3. Generate a real baseline run

1. Start Ollama and ensure the configured model appears in `ollama list`.
2. Load the bundled synthetic topics and availability in the app.
3. Open the **LLM Baseline** tab.
4. Confirm the displayed `OLLAMA_MODEL` is the intended exact model.
5. Click **Generate LLM schedule**.
6. Review the validation status, constraint violations, and raw responses.
7. Download the run JSON and keep it with your results.

Do not put personal, student, or sensitive data into the app. Use synthetic data
only. Keep `OLLAMA_BASE_URL` pointed at a local Ollama service; if you change it,
you are responsible for where data is sent.

## LLM workload accounting and validation

For the LLM baseline, each topic's workload is calculated from its original
estimated study hours:

```text
required_minutes = estimated_study_hours * 60
```

The result is rounded half-up to the nearest 30 minutes with a Decimal-based
helper. There is no 30-minute minimum for the LLM baseline, so values below 15
minutes round to 0. The adaptive quiz-score multiplier is not applied.

Completed sessions reduce the workload before generation:

```text
completed_minutes =
    sum of completed session durations for that course/topic

remaining_required_minutes =
    max(required_minutes - completed_minutes, 0)
```

Only sessions explicitly marked `completed=True` reduce the requirement. A completed
session is never counted again as a generated LLM session.

These fields are supplied explicitly to the model for every topic:

- `required_minutes`
- `completed_minutes`
- `remaining_required_minutes`

The model is not asked to infer them. Its output must satisfy, for every topic:

```text
generated_scheduled_minutes
+ unscheduled_minutes
= remaining_required_minutes
```

The model must return JSON only. The validator checks:

- JSON is parseable and has the required structure
- the structured input contains at least one valid course/topic entry
- every course/topic exists in the input
- no unknown courses/topics are introduced
- each session fits inside an availability window
- sessions do not overlap each other or completed sessions
- duration is an integer from 30 to 90 minutes
- sessions are not before the supplied current date/time or after the deadline
- a session may start exactly at the current date/time but not earlier on the same day
- completed sessions are exempt from the current date/time check
- scheduled time does not exceed window or topic availability
- generated sessions plus reported unscheduled minutes exactly equal the topic's
  remaining required minutes
- `unscheduled_minutes` is a positive integer, with at most one entry per topic
- zero-remaining topics receive no new session and no positive unscheduled entry
- fully scheduled topics do not appear in `unscheduled_topics`

An output may satisfy every structural and temporal constraint but still fail
workload accounting. Temporal validity alone is not sufficient. The validator does
not silently shift sessions, change durations, infer natural-language reason
correctness, or substitute a rule-based schedule.

If the first response is invalid, exactly one repair request is sent with the
validation errors. The repaired response is validated again. If it remains invalid,
cannot be parsed, or the repair request fails, the run is recorded as a failure.
The final parsed schedule and accounting summary are then `null` in the log; raw
responses and validation errors are always retained.

Each run is logged to:

```text
sample_outputs/llm_runs/<run_id>.json
```

Newly generated logs record a repository-relative `log_path`, not an absolute local
drive path. Logs include the run ID, timestamp, exact model name, model parameters,
exact initial/repair prompts, structured input with the workload fields, raw
responses, parsed schedule, initial and final validation errors, repair-attempt flag,
runtime, final status, and an `accounting_summary` with per-topic and total accounting
values.

A failed LLM result may still be downloaded for debugging. It is explicitly labeled
as invalid and is not displayed as a valid schedule.

The accounting summary is diagnostic only. A result is valid only when JSON/schema,
topic, duration, availability, overlap, deadline, and workload-accounting checks all
pass. `is_complete` reflects exact workload equality for valid values and a
non-empty topic list; it is not by itself proof that the whole schedule is valid.
`is_complete` is false and coverage is unavailable when the topic list is empty or
any accounting value is malformed.

Coverage is reported as `accounted_minutes / remaining_required_minutes * 100` and is
not capped, so over-accounting is visible above 100%. The app displays `N/A` and the
log stores `null` when the topic list is empty or any accounting value is malformed.
If the total remaining requirement is zero and no additional minutes are scheduled
or reported, coverage is 100% and the accounting summary is complete. Malformed or
unsupported values are not coerced into integers; they are reported as validation
errors and flagged in the accounting summary instead of crashing it.

## Tests

Run the complete suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The LLM tests mock `requests.post` and **do not require Ollama**. They cover:

- valid JSON
- malformed JSON
- hallucinated topic
- session after deadline
- session earlier on the current day
- session exactly at the current time
- completed sessions exempt from the current time check
- `current_datetime` minute rounding and date consistency
- invalid planner input rejected without an Ollama request
- overlapping sessions
- unavailable time slot
- failed connection
- successful repair
- failed repair
- missing model configuration
- exact model name and temperature recording
- run log creation and repository-relative log paths
- strict JSON handling
- exact workload accounting
- under-accounting and over-accounting
- missing, duplicate, unknown, zero, negative, boolean, float, and string
  unscheduled values
- fully scheduled and zero-remaining topics
- completed sessions reducing or clamping the remaining requirement
- accounting errors triggering exactly one repair
- repaired exact accounting and repaired mismatches
- malformed-output safety for accounting summaries
- same topic names in different courses
- accounting equality not overriding temporal validation

The tests do not prove that a particular real Ollama model will produce a valid
schedule. Mocked tests only exercise deterministic code and stubbed HTTP responses.
Real-model performance requires the manual integration test below.

### Manual integration test

This test requires a real local Ollama service and an installed model.

1. Install/start Ollama and pull a model.
2. Set `OLLAMA_MODEL` to that exact model tag in the terminal that will run the app.
3. Run `.\.venv\Scripts\python.exe -m streamlit run app.py`.
4. Load the synthetic sample data.
5. In **LLM Baseline**, click **Generate LLM schedule**.
6. Verify the app shows the exact model, validation status, and either a valid
   schedule or constraint violations.
7. Download the JSON run log.
8. Record the reproducibility information below.

The automated test suite mocks `requests.post` and does not call a real model.
Manual real-model runs were performed on 2026-11-16, before strict workload
accounting was implemented. They are historical records only and were not
revalidated under the current validator.

## Historical real-run record (2026-11-16)

Environment: Ollama 0.40.16 on Windows 10, model `llama3.2:3b`, base URL
`http://127.0.0.1:11434`, temperature 0, `format: "json"`, Python 3.12.

| Run ID | Trigger | Result under the validator used at the time | Runtime |
|---|---|---|---|
| `82e0d8b64d5049daba5d265fd6420b8f` | first call after service start | failure: 120 s read timeout during cold model load | 120.07 s |
| `252db55f1256491abe0d63416015cb4c` | direct `generate_llm_schedule` call, model warm | Passed the temporal/structural validator used at the time, but would fail the new workload-accounting check because 300 scheduled + 120 unscheduled minutes accounted for only 420 of 780 required minutes. | 24.53 s |
| `132ffc417b464cd09c34a72189b5e61f` | Streamlit **LLM Baseline** page button | Passed the temporal/structural validator used at the time, but would fail the new workload-accounting check because 300 scheduled + 120 unscheduled minutes accounted for only 420 of 780 required minutes. | 25.23 s |

Another historical success artifact, `fccd64a17c4042f9a3f0b7816bc0e2c7`, also
predates strict workload accounting. It is not evidence of passing the current
validator.

Each run is written to `sample_outputs/llm_runs/<run_id>.json` and is excluded
from version control by `.gitignore`. Historical JSON logs are not rewritten.

The first call can exceed the 120-second default while the model loads. Warm up
before generating:

```powershell
ollama run llama3.2:3b "OK"
```

Alternatively, raise `DEFAULT_OLLAMA_TIMEOUT_SECONDS` in `config.py`.

## Reproducibility information

For any real LLM baseline result, report:

- exact model name and tag (`OLLAMA_MODEL`)
- Ollama version
- `OLLAMA_BASE_URL`
- temperature and other model parameters
- run ID and timestamp
- exact prompt or prompt file version
- structured input/sample CSV contents or hashes
- raw initial and repair responses
- parsed schedule
- validation errors
- whether repair was attempted
- runtime
- final status
- Python version and operating system

The per-run JSON log is designed to contain these details.

## Assumptions and limitations

- Dates and times are local, timezone-naive values.
- The deadline is the end of that calendar day.
- Availability is explicit; there is no automatic recurrence.
- A session cannot span multiple availability windows or cross midnight.
- Today's availability is clipped to the current time; past availability is ignored.
- Priority sorting uses unrounded priority; only display is rounded.
- Past incomplete sessions are retained as history and do not count as completed.
- Streamlit state is in memory and is not durable across app restarts.
- The LLM baseline is not automatically integrated into the adaptive optimizer.
- The LLM baseline uses original estimated study time, not the adaptive quiz-score
  multipliers.
- The LLM may allocate sessions differently from the rule-based planner. It cannot
  leave required minutes unaccounted for and still be valid, but validity does not
  mean the schedules are equivalent.
- Natural-language reasons are only checked for being non-empty. Their factual truth
  is not inferred.
- Unused schedulable time is a schedule-quality concern, not something the validator
  silently corrects.
- LLM validation rejects sessions before the supplied current date/time, including
  earlier times on the same day; a session may start exactly at the current minute.
  Completed sessions are exempt because they are historical records. The structured
  input is not trimmed to future availability, so the model may still propose slots
  that the validator must reject.
- Mocked tests do not prove real-model performance.
- Run logs are runtime artifacts ignored by `.gitignore` and are not committed.
  Historical logs from 2026-11-16 predate strict workload accounting and were not
  rewritten or revalidated; see the historical run record above.
