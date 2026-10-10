# Research Harness

Research harness for decidability, complexity, and strategy synthesis in distributed games and automata-theoretic synthesis.

This is a local CLI-first system for LICS-style research frontiers across distributed synthesis, asynchronous games, control games, Petri games, games on graphs, imperfect-information games, trace theory, Zielonka/asynchronous automata, MSO/logical characterizations, and parity/safety/reachability/liveness objectives. ATS/CDM/2DM games are supported as one seed model family and experiment layer, not as the boundary of the project.

## Philosophy

The autonomous runner creates one finished research unit per successful step. Model-generated findings remain draft or unverified; experimental observations are not proofs. Source-reported claims may be used as working premises, but their provenance and uncertainty stay visible. No general mathematical proof verifier is implemented.

The older extraction and planning helpers still exist internally, but the GUI and CLI no longer expose pending-review approval controls.

## Install

Use Python 3.11 or newer.

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## PostgreSQL Setup

PostgreSQL is the only supported persistence backend. The application never installs or
controls an operating-system database service during normal runtime. Provision it explicitly:

```bash
# Debian/Ubuntu only; runs with administrator privileges.
sudo ./scripts/install_postgres_debian.sh
```

For an existing PostgreSQL server, an administrator can create the restricted application
role and database with the supplied bootstrap script. Keep credentials only in your shell or
secret manager; never commit them:

```bash
export POSTGRES_ADMIN_URL="postgresql://admin/postgres"
export RESEARCH_HARNESS_DB_PASSWORD="choose-a-secret"
python scripts/bootstrap_postgres.py
export DATABASE_URL="postgresql://research_harness:choose-a-secret/research_harness"
python -m src.cli init-db
```

For self-hosted development, Docker Compose provisions PostgreSQL with a persistent volume:

```bash
export POSTGRES_PASSWORD="choose-a-secret"
docker compose up -d postgres
export DATABASE_URL="postgresql://research_harness:choose-a-secret/research_harness"
python -m src.cli init-db
```

Tests use isolated, automatically removed PostgreSQL schemas. Set `TEST_DATABASE_URL` to a non-production database before running them; if unset, tests use `DATABASE_URL`.

## Environment

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

The LLM settings are:

```text
LLM_PROVIDER=placeholder
LLM_API_KEY=
LLM_MODEL=placeholder-model
LLM_BASE_URL=https://api.openai.com/v1
```

The default `LLM_PROVIDER=placeholder` keeps extraction deterministic and offline. To opt into
an OpenAI-compatible provider, set `LLM_PROVIDER=openai` (or `openai-compatible`), provide
`LLM_API_KEY`, select `LLM_MODEL`, and use `--llm` with extraction. Set `LLM_BASE_URL` to an
OpenAI-compatible `/v1` endpoint for DeepSeek or another compatible service. API keys are never
printed or written to the database. OpenAlex can be queried without a key for small
searches; set the optional `OPENALEX_API_KEY` for a higher discovery rate limit.

## Local Dashboard

Install the project dependencies, initialize or migrate the database, then start the local
human-facing dashboard:

```powershell
python -m pip install -r requirements.txt
python -m src.cli init-db
streamlit run src/gui/app.py
```

The GUI opens on Research Tasks. Create a fixed objective, choose a task, and run a
user-defined number of autonomous steps. Optionally focus the first step on any existing
unit, or force a full-text literature survey from that unit. Each successful step creates a
finished child unit. The database explorer and experiment history remain read-only views.
Review/approve/reject controls are absent. Existing API keys are never prefilled into widgets.

Open **Settings** to configure one active OpenAI-compatible API connection, enter
a masked API key, change the database connection, and choose models by activity.
Settings are session-only: they survive page navigation but not a new browser
session or server restart. They do not overwrite `.env`, write credentials to the
database, or change other users' sessions. Reset restores the environment defaults.
Use this credential-entry interface locally; it is not an authenticated public service.

**Load models from API** makes an explicit metadata request to the active endpoint,
not a completion. Choose returned model IDs from dropdowns, or enter a custom ID
if the provider does not offer a compatible model list. No API calls are made just
by opening Settings. An available model is not necessarily free or suitable for
structured responses. “Inherit” uses the research model for ordinary activities and
the literature model for literature substeps; both fall back to the default.
If a selected activity has a different specialist model, that model develops the
activity before it is stored. A model ID must be supported by your endpoint;
entering a name does not enable a provider's web-search or deep-research tools.

The compact, height-limited unit graph uses numbered circles and thin left-to-right
connections, with dashed edges for inputs from other branches. Click a circle (or
focus it and press Enter/Space) to inspect that unit. Colours and the legend identify
activity types, not verification status. Full titles
remain in the inspector and the collapsed all-units table. The unit inspector shows the recorded outcome, linked object IDs,
and models used. A finished activity is not a verified scientific result.

## Initialize The Database

```powershell
python -m src.cli init-db
```

This initializes the PostgreSQL schema and shared ontology concepts. It creates no research tasks; specify a task objective explicitly before starting work.

## Research Tasks And Units

A research task has a fixed name and objective; it is not edited as work progresses.
A research unit is one activity within that task, such as a literature review, hypothesis
investigation, proof attempt, or experiment. Units are added as questions arise; a
`parent_unit_id` records a follow-up branch, while the task timeline remains chronological.
Unit status describes work progress; a finished unit does not establish a theorem.
Combining branches adds `uses` links to earlier research units in the same task.
Those dependencies form an acyclic graph without a separate Git repository or
new database tables. Research objects remain reusable across tasks.

```bash
python -m src.cli add-research-task --name "Global safety in ATS games" --description "Investigate finite-memory strategies for global safety in three-process ATS games."
python -m src.cli list-research-tasks
python -m src.cli add-research-unit --task-id 1 --kind literature_review --title "Survey global safety" --purpose "Find source-backed bounds and assumptions" --status ready
python -m src.cli next-research-unit --task-id 1
python -m src.cli list-research-units --task-id 1
python -m src.cli show-research-unit --unit-id 1
```

For literature intake, attach a local PDF to a paper record or let an autonomous
literature step discover openly accessible source text. It reads PDFs or HTML
article/main text and stores source-attributed claims with exact quotes; it does
not store PDF blobs in PostgreSQL. HTML locations are text sections, not PDF pages.

```bash
python -m src.cli add-paper --task-id 1 --title "Example Paper" --authors "A. Author" --year 2026 --pdf-path papers/example.pdf
python -m src.cli research-loop --task-id 1 --unit-id 1 --literature-from-unit --steps 1 --llm
```

A partial result is stored as a draft `DerivedResult`. It can spawn a proposed
follow-up unit without implying that the underlying question is solved.

```bash
python -m src.cli record-unit-result --unit-id 1 --title "Small case" --statement "Observation from the checked fragment" --followup-title "Check larger cases" --followup-purpose "Test whether the observation extends"
python -m src.cli update-research-unit --unit-id 1 --status finished --outcome-note "Partial result; larger cases remain."
```

Use `link-research-unit` to attach an existing paper, conjecture, open problem,
proof attempt, experiment run, or derived result with a relation such as
`investigates`, `uses`, or `produces`. A unit can reference objects from another
task as inputs; produced task-scoped objects must belong to its own task.
The runner can select a promising existing unit or build from `--unit-id`. Literature
steps search new open-access sources as well as unprocessed local PDFs.

## Papers

```powershell
python -m src.cli add-paper --title "Example Paper" --authors "A. Author;B. Writer" --year 2026 --venue "Draft" --task-id 1
python -m src.cli list-papers
```

## Shared Concepts

```powershell
python -m src.cli list-concepts
python -m src.cli add-concept --name "Observation equivalence" --type logic --aliases "obs-eq"
python -m src.cli link-concepts --source 1 --target 2 --relation related_to
```

Concepts form a shared ontology; they are not automatically created as research tasks.

## Research Planning

Use a configured LLM to turn a free-form objective into a small, reviewable plan grounded in the stored research state:

```powershell
python -m src.cli plan-research --goal "Investigate whether finite-memory strategies suffice for global safety in three-process ATS games." --task-id 1 --llm
```

The command only displays proposed directions. It does not create tasks, persist a plan,
approve claims, or run experiments.

Without `--llm` and a configured remote provider, the command writes nothing and reports that planning is unavailable.

## Autonomous Research Steps

`research-loop` runs from a fixed task objective. Set `--steps` to the number of
successful research units to create. The first step may be forced from `--unit-id`;
later steps select a promising existing unit, so progress may branch. A failed
provider call, invalid proposal, inaccessible source, or failed transaction stops
the run and reports the units already completed. A failed literature retrieval or
quote check also creates a blocked unit with a diagnostic ID, but no claims. It is
reported separately and does not count as a completed step. Select that unit on a
later run to retry or develop another direction.

Continuation includes the selected unit's outcome and up to 20 explicitly linked
research records, with bounded statement, proof-note, evidence and result text,
in addition to the task snapshot. Running research sends this context to your
configured LLM endpoint. Credentials and arbitrary local files are not context.
Each successful unit records contributing provider/model/usage metadata in its
timeline event, without storing raw prompts or API keys.

```bash
python -m src.cli research-loop --task-id 1 --steps 10 --llm
python -m src.cli research-loop --task-id 1 --unit-id 3 --steps 2 --llm
python -m src.cli research-loop --task-id 1 --unit-id 3 --literature-from-unit --steps 1 --llm
```

The LLM uses the shared `src/llm.py` provider. It may record analysis, draft
partial results, provisional conjectures or open problems, draft proof attempts,
an open-access full-text survey, or the existing tiny bounded ATS/CDM/2DM
experiment. Only tested checker artifacts may be used for that experiment.
No model-generated statement is marked verified. Source-reported results are
working premises with citations and uncertainty; significant conclusions
depending on them remain draft until a real verification mechanism exists.

Literature discovery uses OpenAlex metadata across publishers and repositories,
then reads directly available public HTTPS PDFs, with open HTML article/main text
as a fallback. Every fallback paper undergoes relevance selection. It also reads
unprocessed local PDFs already attached to a task. PDF bytes are not stored in
PostgreSQL. The extractor requires an exact quote from an extracted page or HTML
text section before inserting a
source-reported theorem, conjecture, or open problem. It does not bypass
paywalls, perform OCR, or verify mathematical correctness. Limits are 20 MB,
100 pages, and 300,000 extracted characters per PDF. HTML downloads are limited to
2 MB and 300,000 extracted characters. Some publisher pages expose only abstracts
or require JavaScript; this is not universal full-text access, and quotations prove
source attribution, not the truth or completeness of a claim. Failed downloads
include safe host/error details in the GUI and private local diagnostic log.

### Failure diagnostics

Failed research runs display a diagnostic ID and details in **Last run → Error
details**, also returned by `research-loop`. The private, owner-only log is
`data/research_errors.jsonl`. Each entry includes the UTC time, task/step,
research stage, parent unit, and stack locations. LLM failures also include
provider, selected model/role, endpoint host, HTTP status, recognized error code,
request ID and retry interval when supplied, attempt history and elapsed time.
Known quota errors are distinguished from rate limiting; an unexplained 429 is
explicitly labeled ambiguous. Validation errors retain field names and schema
name, not model output. Literature failures retain their reason and safe download
details. If writing the local log fails, the GUI says no entry was written.

Raw provider error messages/bodies, prompts, responses, authorization headers,
full endpoint URLs and API keys are not logged. Only recognized error codes and
types are retained, so an unknown provider reason may remain unspecified. GUI
session keys are protected as well as environment keys. These diagnostics do not
change retry policy or research behavior, and cannot recover details discarded
by older log entries.

## Legacy Extraction

`extract-from-text` and `extract-from-pdf` remain available for compatibility.
They write candidates to the old pending table, not to the autonomous task
memory. The review commands and GUI approval view are no longer exposed.
Use autonomous full-text literature steps for the new workflow.

## Literature Research Demo

Run the local vertical-slice literature workflow from approved seed artifacts:

```powershell
python -m src.cli research-demo --dry-run
python -m src.cli query-literature --topic-id 1 --question "What is known about global safety in CDM or ATS games?"
python -m src.cli research-memo --topic-id 1 --question "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?"
python -m src.cli research-memo --topic-id 1 --question "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?" --llm
python -m src.cli quality-check-literature --topic-id 1
python -m src.cli generate-verification-tasks --topic-id 1
```

The demo creates a research topic for causally ordered two-decision-maker ATS/CDM safety synthesis, loads local approved seed JSON files from `results/approved/`, stores literature notes and summaries in PostgreSQL, links evidence spans, and writes `results/literature/demo_literature_map.md`.

`research-memo` writes `results/literature/topic_<id>_research_memo.md` from stored evidence only. New conjectures are labelled as conjectures, and unsupported points are labelled `needs verification`.

`quality-check-literature` writes a local quality report for the notes, summaries, evidence spans, map, memo, and query robustness checks. `generate-verification-tasks` writes literature/theory checking tasks; it does not suggest experiments.

No new theorem-like claim is inserted into approved theorem tables by the demo. The report uses stored seed material and marks unresolved points as needing verification. See [docs/literature_workflow.md](docs/literature_workflow.md).

## Research Map Queries

```powershell
python -m src.cli theorems-by-task 1
python -m src.cli theorems-by-model "ATS games"
python -m src.cli theorems-by-objective safety
python -m src.cli open-problems-by-task 1
python -m src.cli show-research-map
```

`show-research-map` prints research tasks, key papers, key theorems, known upper/lower bounds, open gaps, and candidate conjectures.

## Conjectures

```powershell
python -m src.cli add-conjecture --statement "Every bounded-memory controller for this fragment has a finite-state normal form." --task-id 1 --attack-plan "Search tiny counterexamples first."
python -m src.cli list-conjectures
python -m src.cli show-conjecture 1
python -m src.cli update-conjecture-status 1 --status paused
```

## Experiments

The current experiment plugin supports tiny ATS/CDM/2DM safety games. The layout is intentionally modular:

```text
src/experiments/
  ats_models.py
  ats_generator.py
  ats_brute_solver.py
```

Later experiment plugins can add Petri game toy generation, graph game solvers, parity game solvers, or trace automata experiments.

Generate a tiny game:

```powershell
python -m src.cli generate-game --kind ATS --processes 3 --states 2 --seed 7 --output data/tiny_game.json
```

Run the bounded brute checker:

```powershell
python -m src.cli brute-check --input data/tiny_game.json --depth 5
```

## Code And Experiment Management

Research claims live in the database. Code lives in Git. Experiment metadata links them.

Reusable implementation code should live under `src/libraries/`, `src/experiments/`, or ordinary Git-tracked experiment scripts. PostgreSQL stores artifact metadata, command lines, result summaries, input/output paths, and git commit hashes. It does not store source-code blobs.

The repository includes these long-term code and output areas:

```text
src/libraries/
  ats/
  graph_games/
  reductions/
  search/
experiments/
  restricted_2dm/
  reachability_gap/
  logical_characterization/
generated/
  games/
  counterexamples/
results/
```

Register the current ATS brute checker as a code artifact:

```powershell
python -m src.cli register-code-artifact --name "ATS brute checker" --path "src/experiments/ats_brute_solver.py" --artifact-type checker --tests-path "tests/test_brute_solver.py" --related-concepts "ATS games;safety objective"
python -m src.cli list-code-artifacts
python -m src.cli show-code-artifact --artifact-id 1
```

Generate a tiny input game and run the checker as a recorded experiment:

```powershell
python -m src.cli generate-game --kind ATS --processes 2 --states 2 --seed 1 --output generated/games/tiny_ats.json
python -m src.cli run-experiment --artifact-id 1 --command "python -m src.cli brute-check --input generated/games/tiny_ats.json --depth 5" --input-path generated/games/tiny_ats.json --experiment-type ats_bounded_safety --conjecture-id 1
python -m src.cli list-experiment-runs
python -m src.cli show-experiment-run --run-id 1
```

Stdout and stderr are written to a timestamped file under `results/`. Previous result files are never deleted by the runner.

## Week-One Seed Assets

The repository includes seed prompt files under `prompts/` and approval JSON templates under `results/approved/` for an initial LICS-style literature spine:

- restricted ATS/CDM/2DM synthesis
- global objectives in causal-memory games
- acyclic architectures and automata-theoretic transfers

The ordered-2DM conjecture loop has a small driver at:

```text
experiments/restricted_2dm/run_tiny_ordered_2dm_search.py
```

Run it through the experiment manager:

```powershell
python -m src.cli run-experiment --artifact-id 1 --task-id 1 --conjecture-id 1 --command "python experiments/restricted_2dm/run_tiny_ordered_2dm_search.py --instances 200 --max-processes 3 --max-local-states 3 --objective safety --output results/ordered_2dm_tiny_search.json"
```

Run one bounded pipeline step:

```powershell
python -m src.cli run-pipeline --task-id 1 --mode literature
python -m src.cli run-pipeline --task-id 1 --mode experiments
```

## Tests

Tests use a real PostgreSQL server with temporary schemas, but fake LLM and
discovery responses; they do not require paid provider calls. Export
`TEST_DATABASE_URL` for a non-production database. On Linux/macOS, disable loading
your live `.env` so it cannot override offline test assumptions:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q
```
