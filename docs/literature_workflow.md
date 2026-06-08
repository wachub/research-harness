# Literature Workflow

The current literature layer is intentionally small. It demonstrates one local workflow:

```powershell
python -m src.cli research-demo --dry-run
```

The demo uses the fixed smoke-test topic:

```text
causally ordered two-decision-maker ATS/CDM games and decidability of distributed safety synthesis
```

It loads approved local seed JSON files from `results/approved/`, creates paper-like records, stores literature notes and summaries, links evidence spans, and writes:

```text
results/literature/demo_literature_map.md
```

Research claims are not newly approved by this demo. It only reuses existing approved seed artifacts and labels unsettled points as needing verification.

Query stored evidence:

```powershell
python -m src.cli query-literature --topic-id 1 --question "What is known about global safety in CDM or ATS games?"
```

The query command searches only stored literature summaries and evidence spans. For unrelated questions, it returns `insufficient stored evidence`.

Write a small reasoning memo from stored evidence:

```powershell
python -m src.cli research-memo --topic-id 1 --question "Does causal ordering in two-decision-maker ATS/CDM games plausibly recover decidability for distributed safety synthesis?"
```

The memo is written to:

```text
results/literature/topic_<id>_research_memo.md
```

Check the quality of the stored-evidence workflow:

```powershell
python -m src.cli quality-check-literature --topic-id 1
```

Generate next literature/theory verification tasks:

```powershell
python -m src.cli generate-verification-tasks --topic-id 1
```

These commands do not fetch papers, run web search, or insert approved theorem claims.
