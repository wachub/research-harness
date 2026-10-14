# Research Harness

A local, task-driven research workspace with a Streamlit GUI, a Python CLI,
PostgreSQL research memory, and an opt-in LLM-assisted research loop. The current
examples and deterministic experiments concern distributed games and synthesis
(ATS/CDM/2DM); the system is not a general theorem prover.

## How it works today

1. Create a **research task**: a fixed name and research objective.
2. Start a bounded run of 1–100 steps. The model selects a promising existing
   **research unit**, or starts from the task when there are no units.
3. Load task state and the selected unit's linked findings and evidence.
4. Ask the shared LLM client for one structured activity; validate with Pydantic
   and check every referenced database ID.
5. Execute that activity within the implemented limits and store a finished
   unit, its linked outputs, and a timeline event in one transaction.

A unit is an activity, not a precomputed milestone: literature review, analysis,
conjecture, open problem, partial result, proof attempt, or a tiny trusted
experiment. Parent links form branches; `uses` links can combine earlier units
in the same task. Other research objects may be reused as inputs across tasks.
The graph is stored in PostgreSQL, **not Git**.

The human can let the system choose, force continuation from a particular unit,
or force a literature survey from that unit. The first step follows that choice;
subsequent steps choose again. Runs stop on an error and report completed units.
A failed literature retrieval or quotation check can leave a **blocked unit**,
without claims, that can be revisited. Provider/schema failures may stop a run
before any unit is created.

**Finished activity does not mean established result.** Model outputs are drafts
or unverified. Source-reported claims retain citations and exact evidence quotes.
They may be working premises, but no general proof verifier exists and the LLM
cannot mark them verified. The GUI and CLI have no approve/reject review controls.

## Quick start

Python 3.11+ and PostgreSQL are required. Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit the ignored `.env` to set `DATABASE_URL` for your local application database.
Do not overwrite an existing `.env` when updating the repository.
On Windows use `python -m venv .venv`, activate with
`.\.venv\Scripts\Activate.ps1`, and copy the example with
`Copy-Item .env.example .env`.

After provisioning PostgreSQL (below):

```bash
python -m src.cli init-db
python -m streamlit run src/gui/app.py --server.address=127.0.0.1 --server.port=8501
```

Open http://localhost:8501. Go to **Settings**, enter your provider name and API
key, and save. Then open **Research Tasks**, create an objective, and try **one**
research step. Larger step counts can make many provider requests.

The GUI is a local tool, not an authenticated public service. Do not expose it
or its credential-entry forms directly to the Internet.

### PostgreSQL provisioning

PostgreSQL is the only database backend; there is no SQLite mode. Installation
and privileged provisioning are explicit setup actions, not runtime behavior.

For Debian/Ubuntu without PostgreSQL:

```bash
sudo ./scripts/install_postgres_debian.sh
```

For an existing server, an administrator can use:

```bash
export POSTGRES_ADMIN_URL="postgresql://admin/postgres"
export RESEARCH_HARNESS_DB_PASSWORD="choose-a-secret"
python scripts/bootstrap_postgres.py
```

The bootstrap creates a missing application role/database, normally both named
`research_harness`. It does not reset an existing role's password. Set
`DATABASE_URL` to the application's connection URL, then run `init-db`.
The schema initializes shared ontology concepts, not seed research tasks.

Alternatively, Docker Compose runs **only PostgreSQL**, not the GUI or LLM:

```bash
export POSTGRES_PASSWORD="choose-a-secret"
docker compose up -d postgres
```

Its default database/user are `research_harness`, accessible on localhost:5432.
Use the matching credentials in `DATABASE_URL`. Data is in the Compose volume.
Use only one local database service on that port.

## API settings

The main Settings form contains two manually entered fields:

- **Provider name**: `OpenAI`, `Gemini`, `offline`, or `custom`.
- **API key**: masked; existing keys are never prefilled into the widget.

OpenAI and Gemini fill in a compatible endpoint and a default model:
`gpt-4.1-mini` and `gemini-2.5-flash`, respectively. These are convenience
defaults, not promises of free access or sufficient research quality. An
already configured model is preserved when staying on the same endpoint.
See the [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
and [Gemini compatibility documentation](https://ai.google.dev/gemini-api/docs/openai).

**Advanced** contains manually entered model IDs, the API base URL and optional
per-activity overrides. There are no provider/model dropdowns or model-list
requests. For another OpenAI-compatible service (including DeepSeek), configure
its endpoint/model there, then enter its key with provider `custom`.
Changing an endpoint clears its old key and activity overrides; a key is never
silently reused at a different endpoint.

All roles use **one active endpoint and key**, not separate accounts. Blank
role overrides inherit the research-step model or the literature model, then
the default model. A configured specialist can develop a selected activity.
Changing a model name does not enable native web search or a deep-research API.

Settings are **session-only**: page navigation preserves saved settings, but a
new session or server restart does not. Nothing writes keys to the database or
`.env`, and opening/saving Settings makes no API request. Reset restores the
environment configuration. The database connection has its own collapsed section.

For persistent configuration and CLI use, edit `.env` locally:

```dotenv
DATABASE_URL=postgresql:///research_harness
LLM_PROVIDER=placeholder
LLM_MODEL=placeholder-model
LLM_API_KEY=
LLM_BASE_URL=https://api.openai.com/v1
OPENALEX_API_KEY=
```

Remote CLI configuration uses `LLM_PROVIDER=openai`, `openai-compatible`, or
`deepseek`, an explicit compatible `LLM_BASE_URL`, model ID and key.
Without a remote provider, deterministic legacy extraction and local tools
still work; autonomous research and LLM planning report that they are unavailable.
The CLI additionally requires `--llm` for those operations.

Running research sends bounded stored statements, proof notes, outcomes and
evidence to the configured endpoint. It does not send credentials or arbitrary
local files as research context. Never commit keys, `.env`, or private logs.

## Literature retrieval: what it actually does

Literature review is implemented by the harness, not by a model's built-in browser:

1. Try local PDFs already attached to the selected task.
2. Otherwise query OpenAlex for up to five open-access candidates.
3. Ask the model to select a relevant source. A unique keyword match may be
   used when the model returns no selection or fails; that failure is recorded.
4. Try public HTTPS PDF locations, then open HTML article/main text.
5. Check source identity, extracted-content hash and review scope against prior
   successful distillations. Reuse an unchanged distillation and look for a new
   source; otherwise send extracted text in bounded chunks to the LLM.
6. Validate structured claims and require an **exact quote** on the supplied
   page/HTML section before storing anything.

### Distilled memory and repeat reviews

The autonomous literature path stores **distilled findings, not whole papers**:
self-contained statements, explicit assumptions and limitations, proof/technique
notes, relevance, and exact supporting excerpts. Theorems, conjectures and open
problems use their existing tables; reusable ideas use draft derived results.
Paper rows retain bibliographic metadata and source links. PDFs are processed in
memory; an attached local PDF remains a file, not a database blob. Evidence records
retain the source URL/identifier, page or HTML-section location, and a hash of the
extracted text so a later source revision can be distinguished.

Repeat intake uses conservative identities:

- DOI and arXiv aliases match, but explicit arXiv revisions stay distinct.
  Otherwise, matching requires normalized title, year and authors.
- Findings match on kind, whitespace-normalized statement, assumptions and
  limitations. Mathematical case and punctuation are preserved. A stronger bound
  or changed condition remains a separate finding.
- The same finding can be reused across tasks and papers through unit links.
  New citations or supporting passages add evidence; improved proof notes annotate
  existing evidence instead of inserting duplicate rows. Existing confidence and
  ownership are preserved.
- Extraction checkpoints live in timeline events. They apply only to the same
  source, extracted text, goal/purpose and distillation version. Changed text or
  purpose is eligible for another extraction; a review with no findings is not
  marked successfully distilled. Retrieval may still occur to detect revisions.

The extraction model sees bounded task findings. A separate comparison request
then checks each nonexact new claim against up to six relevant stored findings
of the same kind, including cross-task findings. Its structured verdict considers
mathematical scope, assumptions, objectives and bounds. Suggested equivalence is
saved as `possible_duplicate`; extensions and contradictions retain their target
reference and explanation. These are **unverified relationships**. A malformed or
unavailable comparison leaves the supported new claim intact. This can flag
paraphrases that string matching misses, but it cannot prove mathematical
equivalence: such claims stay separate until the distinction is resolved. Exact
matching remains the only automatic merge. No existing rows are deleted or
retroactively merged. Literature writes and checkpoint creation are transactional,
with a PostgreSQL advisory lock protecting concurrent intake. This behavior belongs
to the autonomous literature path; legacy/manual claim insertion is not globally
deduplicated. No schema migration is required.

OpenAlex access is separate from LLM access. `OPENALEX_API_KEY` is optional
configuration for discovery, not your OpenAI/Gemini key. Access and rate limits
depend on that service; see [OpenAlex authentication](https://help.openalex.org/api/authentication/).

Downloads can fail because of HTTP access restrictions, DNS/TLS errors, a stale
URL, no open full text, encrypted/scanned PDFs, or HTML requiring JavaScript.
A stronger model cannot fix these access problems. Search may also return no
relevant candidates. The current path is deliberately small: it does not perform
broad web crawling, bypass paywalls, run OCR, or verify mathematical proofs.

PDF limits: 20 MB, 100 pages, 300,000 extracted characters. HTML limits: 2 MB,
300,000 characters, with sufficient article/main text. Extraction chunks are
bounded to 18,000 characters; a larger single PDF page is rejected.
PDF blobs are not stored in PostgreSQL. Local paths, paper metadata, claims,
proof notes and quoted evidence are stored. Quotes establish attribution, not truth.

Attach your own accessible PDF when remote retrieval is blocked:

```bash
python -m src.cli add-paper --task-id 1 --title "Paper title" --authors "A. Author" --year 2026 --pdf-path papers/paper.pdf
python -m src.cli research-loop --task-id 1 --unit-id 1 --literature-from-unit --steps 1 --llm
```

Use IDs that exist in your database. The forced-survey option requires a unit;
without one, run a first autonomous step or create a literature unit explicitly.

## Debugging a stopped run

Open **Last run → Error details**. It reports:

- Failing research stage, provider/model/role and validation fields.
- HTTP status, recognized provider error code, request ID and retry interval
  when supplied. An unexplained 429 remains ambiguous between rate and quota.
- A literature trace: local-file attempts, search candidate counts, selection
  and fallback, download/parse stages, byte/page counts, extraction chunks,
  quote validation and elapsed times.

Failed runs write an owner-only JSONL record to `data/research_errors.jsonl`,
linked by diagnostic ID. The CLI returns the same diagnostic context.
Successful literature runs retain their trace in the unit's timeline event,
visible under **Models and retrieval details**. These are completed-attempt
diagnostics, not a live streaming progress display.

```bash
tail -n 1 data/research_errors.jsonl
```

The log omits prompts, source text, full URLs, raw provider response bodies and
API keys. File locations in stack traces contain filenames, not stack locals.
If log writing fails, the GUI reports that no entry was written. Old entries
cannot be enriched retroactively.

A quota/credit error is not evidence of model weakness; it requires account
action. A schema-validation error means output failed the contract and was not
stored. The latest exception is not necessarily a retrieval error—check the stage.
See [OpenAI error meanings](https://developers.openai.com/api/docs/guides/error-codes).
Retry counts remain bounded; the diagnostics expose the server's retry interval
but do not currently change the client's short retry schedule.

## CLI essentials

With the virtual environment active:

```bash
python -m src.cli --help
python -m src.cli add-research-task --name "Safety strategies" --description "Investigate finite-memory strategies for global safety."
python -m src.cli list-research-tasks
python -m src.cli research-loop --task-id 1 --steps 1 --llm
python -m src.cli list-research-units --task-id 1
python -m src.cli show-research-unit --unit-id 1
python -m src.cli research-loop --task-id 1 --unit-id 1 --steps 2 --llm
```

`plan-research --goal "..." --task-id 1 --llm` displays a proposed plan only;
it does not persist or execute it. Manual unit/link/result, concept, paper,
conjecture and research-map commands remain available; use command-specific
`--help` for their arguments.

The deterministic experiment layer supports tiny ATS/CDM/2DM safety games:

```bash
python -m src.cli generate-game --kind ATS --processes 2 --states 2 --seed 7 --output generated/games/tiny.json
python -m src.cli brute-check --input generated/games/tiny.json --depth 5
```

Autonomous experiment steps can use only the existing trusted, tested checker
artifact under explicit bounds. Manual `register-code-artifact` and
`run-experiment` commands record paths, commands, Git revisions, outputs and
summaries; the latter executes the command you explicitly supply.
A bounded observation is not an unbounded theorem.

## Code and data map

| Area | Responsibility |
| --- | --- |
| `src/gui/app.py`, `settings.py`, `unit_graph.py` | GUI, session settings, selectable unit graph |
| `src/cli.py`, `dashboard.py` | CLI entry point and GUI-facing application functions |
| `src/autonomous_research.py` | Current bounded research runner |
| `src/research_context.py`, `research_actions.py`, `research_units.py` | Scoped context, trusted actions and unit links/results |
| `src/llm.py` | Shared provider, model routing, bounded retries and Pydantic response validation |
| `src/literature/discovery.py` | OpenAlex discovery, PDF/HTML reading, grounded extraction and trace |
| `src/literature/identity.py`, `memory.py` | Conservative identities, distilled findings, evidence and repeat-review checkpoints |
| `src/db.py`, `schemas.py` | PostgreSQL tables, migrations, queries and Pydantic data objects |
| `src/experiments/`, `brute_solver.py`, `models.py` | Implemented small-game models, generation and checkers |
| `src/code_artifacts.py`, `experiment_manager.py`, `git_utils.py` | Artifact and experiment provenance |
| `src/research_planning.py` | Display-only research-plan generation |
| `scripts/`, `docker-compose.yml` | Explicit PostgreSQL provisioning |
| `tests/` | Database, workflow, adversarial, GUI and checker tests |
| `papers/`, `generated/`, `results/`, `data/` | Local sources, generated files, results and diagnostics |

PostgreSQL stores tasks, units, unit/object links, papers, concepts, theorems,
conjectures, open problems, reductions, proof attempts, evidence, derived results,
code/experiment metadata and timeline events. Code and generated files stay on
disk; Git revisions provide code provenance. Empty library/experiment directories
are scaffolding, not implemented solver families.

### Legacy tools

`src/extract.py`, `curate.py`, `ingest.py`, `orchestrator.py`,
`research_controller.py`, `research_policy.py`, and
`src/literature/demo.py` retain older workflows. They are not the GUI's
autonomous runner. Legacy extraction writes pending entries; the manual
`run-pipeline` is not equivalent to `research-loop`. The seed literature demo
uses separate topic/note/summary records, not autonomous discovery.
See [the legacy literature workflow](docs/literature_workflow.md).
No legacy approval controls are exposed by the current GUI or CLI.

## Tests

Tests use a real local PostgreSQL server with temporary, automatically removed
schemas and fake LLM/discovery responses. They do not require paid API calls.
Set `TEST_DATABASE_URL` to a **non-production** database before running;
otherwise the tests use `DATABASE_URL`/the local default.

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q
```

Disabling dotenv keeps live API settings out of offline test assumptions.
On PowerShell, set `$env:PYTHON_DOTENV_DISABLED="1"` before running pytest.
