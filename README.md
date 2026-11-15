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
| `llm_baseline.py` | Ollama HTTP client, prompt construction, strict validation, one repair request, and run logging |
| `prompts/llm_scheduler_prompt.txt` | LLM scheduling prompt with strict JSON schema |
| `sample_topics.csv` | Synthetic topics |
| `sample_availability.csv` | Synthetic availability windows |
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

The sample deadlines are in November/December 2026. If you run the project after
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

### 1. Install and start Ollama

Install Ollama from the official download page: <https://ollama.com/download>.

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

## LLM validation and repair

The model must return JSON only. The validator checks:

- JSON is parseable and has the required structure
- every course/topic exists in the input
- no unknown courses/topics are introduced
- each session fits inside an availability window
- sessions do not overlap each other or completed sessions
- duration is an integer from 30 to 90 minutes
- sessions are not before the current date or after the deadline
- scheduled time does not exceed window or topic availability

If the first response is invalid, exactly one repair request is sent with the
validation errors. The repaired response is validated again. If it remains invalid,
the run is recorded as a failure. The app never silently shifts sessions, changes
durations, or substitutes a rule-based schedule.

Each run is logged to:

```text
sample_outputs/llm_runs/<run_id>.json
```

Logs include the run ID, timestamp, exact model name, model parameters, exact
initial/repair prompts, structured input, raw responses, parsed schedule, validation
errors, repair-attempt flag, runtime, and final success/failure status.

A failed LLM result may still be downloaded for debugging. It is explicitly labeled
as invalid and is not displayed as a valid schedule.

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
- overlapping sessions
- unavailable time slot
- failed connection
- successful repair
- failed repair
- missing model configuration
- exact model name and temperature recording
- run log creation
- strict JSON handling

The tests do not prove that a particular real Ollama model will produce a valid
schedule. That requires the manual integration test below.

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

A real run has not been performed or claimed as part of this repository's automated
tests.

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
- The LLM may produce a valid schedule with different allocations or less study time
  than the rule-based planner; validity does not mean the schedules are equivalent.
- No real LLM run is included in the repository.
