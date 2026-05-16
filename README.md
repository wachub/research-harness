# Research Harness

Local command-line MVP for formal-methods and theory research on ATS games, CDM games, 2DM games, control games, and distributed synthesis.

The harness stores literature metadata, theorem-like research artifacts, pending LLM-extracted entries, generated tiny games, and brute-force safety checks. It is intentionally local-first: SQLite for storage, environment variables for LLM configuration, and no web app.

## Install

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

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
```

If `LLM_API_KEY` is empty, extraction runs in dry-run mode. The API key is never printed.

## Initialize The Database

```powershell
python -m src.cli init-db
```

By default this creates `data/research.db`.

## Add And List Papers

```powershell
python -m src.cli add-paper --title "Example Paper" --authors "A. Author;B. Writer" --year 2026 --venue "Draft" --pdf-path "papers/example.pdf" --notes "Initial note"
python -m src.cli list-papers
```

## Run Extraction

Paste text through stdin:

```powershell
"Theorem. Every tiny safe instance has a memoryless winning strategy under the given assumptions." | python -m src.cli extract-from-text
```

Or read from a file:

```powershell
python -m src.cli extract-from-text --file notes/example.txt
```

LLM-assisted extraction writes only to `pending_entries`. It never writes directly to the main theorem, model, reduction, or open-problem tables.

## Curate Pending Entries

```powershell
python -m src.cli list-pending
python -m src.cli approve-pending 1
python -m src.cli approve-pending 2 --reject --reason "Duplicate"
```

Approval validates the pending payload and then inserts it into the matching main table.

## Generate Tiny Games

```powershell
python -m src.cli generate-game --kind ATS --processes 3 --states 2 --depth 5 --seed 7 --output data/tiny_game.json
```

Supported kinds are `ATS`, `CDM`, and `2DM`. The generated file is a small JSON safety game.

## Brute-Force Safety

```powershell
python -m src.cli brute-check --input data/tiny_game.json --depth 5
```

The checker enumerates memoryless distributed strategies for very small games and checks all schedules up to the requested depth.

## Tests

```powershell
python -m pytest
```

