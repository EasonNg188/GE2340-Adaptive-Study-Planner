# LLM run outputs

This directory is populated at runtime when the **LLM Baseline** page calls a
local Ollama model. Each run is written as:

```text
sample_outputs/llm_runs/<run_id>.json
```

Newly generated logs use a repository-relative `log_path`. The JSON log contains
the run ID, timestamp, exact configured model name, model parameters, exact
prompts, synthetic structured input, raw model responses, parsed schedule,
initial and final validation errors, repair-attempt flag, runtime, and final
status.

The structured input includes, for every topic:

```text
required_minutes
completed_minutes
remaining_required_minutes
```

`required_minutes` is based on the topic's original estimated study hours. The
LLM baseline does not apply the adaptive quiz-score study-time multipliers.
Completed minutes are subtracted before generation. For a generated schedule to
be valid, every topic must satisfy:

```text
generated_scheduled_minutes
+ unscheduled_minutes
= remaining_required_minutes
```

Logs also include an optional `accounting_summary` with per-topic and total
accounting values, coverage percentage, and `is_complete`. `coverage_percentage`
is `null` when the topic list is empty or any accounting value is malformed; the
summary is then not complete. The summary is diagnostic only. A result is valid
only when all structural, temporal, and workload-accounting checks pass.

If the final response cannot be parsed, or the repair request fails,
`parsed_schedule` and `accounting_summary` are `null`. The raw initial/repair
responses and the initial/final validation errors are still recorded.

No fabricated successful LLM run is included in this repository. A log only
appears after a real call to Ollama. Unit tests write to temporary directories
and do not create sample runs here.

Historical real-model logs from 2026-11-16 are present on the development machine
and are ignored by `.gitignore`. They passed only the temporal/structural
validator used at the time. They predate strict workload accounting, would fail
the new accounting check where required minutes were unaccounted for, and have
not been rewritten or revalidated under the current validator.

The historical success logs include:

- `252db55f1256491abe0d63416015cb4c`
- `132ffc417b464cd09c34a72189b5e61f`
- `fccd64a17c4042f9a3f0b7816bc0e2c7`

These are historical artifacts, not claims of current-validator success.

Other historical failure logs are `82e0d8b64d5049daba5d265fd6420b8f` (cold-load
timeout) and `e69d425b27204bb183801d9ea4375c77` (`OLLAMA_MODEL` was not set).
