# LLM run outputs

This directory is populated at runtime when the **LLM Baseline** page calls a
local Ollama model. Each run is written as:

```text
sample_outputs/llm_runs/<run_id>.json
```

The JSON log contains the run ID, timestamp, exact configured model name, model
parameters, exact prompts, synthetic structured input, raw model responses,
parsed schedule, validation errors, repair-attempt flag, runtime, and final
status.

No fabricated successful LLM run is included in this repository. A log only
appears after a real call to Ollama. Unit tests write to temporary directories
and do not create sample runs here.

Real-model runs have been performed on the development machine (see the
"Real run record" section in the top-level README). Their JSON logs are ignored
by `.gitignore` and are not committed.
