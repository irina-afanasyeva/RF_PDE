# Burgers RF Research Documentation

This directory documents the ongoing random-feature (RF) + local-correction
study of the viscous Burgers equation, developed on the `burgers-local-features`
branch. It is maintained alongside (not inside) the research code in
`Burgers/`.

## Layout

- **[RESEARCH_LOG.md](RESEARCH_LOG.md)** — the scientific/technical
  chronology: hypothesis → experiment → result → interpretation. This is
  where numerical results and their meaning live, including negative
  results. Read this to understand *what has been tried and what it showed*.
- **[meetings/](meetings/)** — one file per advisor meeting (`YYYY-MM-DD.md`):
  what was known/presented at that meeting, what was discussed, and what was
  decided as next steps. Read this to understand *why* a direction was
  pursued, not just what the code did.
- **[TODO.md](TODO.md)** — the current actionable research plan, prioritized
  from the immediate next experiment onward, plus a running list of
  completed experiments so they aren't accidentally repeated.
- **[IDEAS.md](IDEAS.md)** — hypotheses and candidate directions that are
  not yet committed experiments, including a list of ideas already tested
  and found not to work.

## Maintenance convention

- Detailed numerical results belong in `RESEARCH_LOG.md` only — avoid
  repeating them in meeting notes, `TODO.md`, or `IDEAS.md`.
- After each new experiment: add an entry to `RESEARCH_LOG.md`
  (hypothesis → experiment → result → interpretation), then update
  `TODO.md` (mark done, add the next step) and `IDEAS.md` (promote or
  demote hypotheses) as needed.
- After each advisor meeting: add one new file under `meetings/`.
- Git commit hashes and exact script paths are included in
  `RESEARCH_LOG.md` where they aid traceability, but this is a scientific
  log, not a changelog — the hypothesis/result/interpretation is the point,
  not the commit history.
- Preserve negative results and changes of interpretation as they actually
  happened. Do not rewrite the log to look more linear in hindsight than the
  research was.
