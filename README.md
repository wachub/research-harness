# Research Harness

Research harness for decidability, complexity, and strategy synthesis in distributed games and automata-theoretic synthesis.

This is a local CLI-first system for LICS-style research frontiers across distributed synthesis, asynchronous games, control games, Petri games, games on graphs, imperfect-information games, trace theory, Zielonka/asynchronous automata, MSO/logical characterizations, and parity/safety/reachability/liveness objectives. ATS/CDM/2DM games are supported as one seed model family and experiment layer, not as the boundary of the project.

## Philosophy

LLMs propose; code checks where possible; human approves important research claims.

Extraction writes only to `pending_entries`. Claims become durable research memory only after manual curation and approval. Evidence spans are available so important theorems, reductions, open problems, and concepts can point back to exact paper locations instead of relying on vague memory.

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

This creates or migrates `data/research.db`, then seeds broad research clusters and ontology concepts.

## Recommended Workflow

1. Create or choose a research cluster.
2. Add papers.
3. Extract candidate concepts, theorems, reductions, open problems, and conjecture seeds.
4. Curate pending entries and approve only the claims you trust.
5. Add conjectures and proof attempts.
6. Generate small ATS/CDM/2DM examples where useful.
7. Run bounded brute checks.
8. Store experiment runs, derived results, and evidence spans.

## Papers

```powershell
python -m src.cli add-paper --title "Example Paper" --authors "A. Author;B. Writer" --year 2026 --venue "Draft" --cluster-id 1
python -m src.cli list-papers
```

## Research Clusters And Concepts

```powershell
python -m src.cli list-clusters
python -m src.cli add-cluster --name "Partial-information parity games" --priority 9

python -m src.cli list-concepts
python -m src.cli add-concept --name "Observation equivalence" --type logic --aliases "obs-eq"
python -m src.cli link-concepts --source 1 --target 2 --relation related_to
```

Seed clusters include restricted multi-decision-maker synthesis, two-process distributed reachability, logical/automata characterizations of distributed strategies, Petri/control games, asynchronous automata and trace theory, and imperfect-information games.

## Extraction And Curation

Paste text through stdin:

```powershell
"Theorem. ATS games with safety objective are decidable for two-process architectures." | python -m src.cli extract-from-text
```

Or read from a file:

```powershell
python -m src.cli extract-from-text --file notes/example.txt
```

Review candidates:

```powershell
python -m src.cli list-pending
python -m src.cli show-pending-detail 1
python -m src.cli curate-pending
python -m src.cli approve-pending 1
python -m src.cli reject-pending 2 --reason "Duplicate"
python -m src.cli flag-pending 3 --reason "Needs exact theorem number"
```

The curator warns about missing source locations, missing assumptions, duplicate statements, vague complexity language, decidability claims without architecture/information assumptions, and alias collisions in concepts.

## Research Map Queries

```powershell
python -m src.cli theorems-by-cluster 1
python -m src.cli theorems-by-model "ATS games"
python -m src.cli theorems-by-objective safety
python -m src.cli open-problems-by-cluster 1
python -m src.cli show-research-map
```

`show-research-map` prints active clusters, key papers, key theorems, known upper/lower bounds, open gaps, and candidate conjectures.

## Conjectures

```powershell
python -m src.cli add-conjecture --statement "Every bounded-memory controller for this fragment has a finite-state normal form." --cluster-id 1 --attack-plan "Search tiny counterexamples first."
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

Reusable implementation code should live under `src/libraries/`, `src/experiments/`, or ordinary Git-tracked experiment scripts. SQLite stores artifact metadata, command lines, result summaries, input/output paths, and git commit hashes. It does not store source-code blobs.

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

Run one bounded pipeline step:

```powershell
python -m src.cli run-pipeline --cluster-id 1 --mode literature
python -m src.cli run-pipeline --cluster-id 1 --mode experiments
```

## Tests

```powershell
python -m pytest
```
