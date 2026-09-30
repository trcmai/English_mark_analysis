# Payroll Passport (prototype)

Answers payroll questions from ranked, traceable evidence, or routes them to an expert when the
evidence isn't good enough, then stores the expert's validated answer as reusable knowledge.

> The sample knowledge base (`data/sample_kb.json`) is **fictional**. Its values are not real legislation.
> It covers six countries (NL, DE, BE, FR, UK, VN) and 13 topics, with one expert set per country.

## How an answer is produced

1. **Retrieve** sources tagged with the question's topic.
2. **Applicability rules** (deterministic): country, client (other clients' documents are never used),
   employee scope, and in force on the question date.
3. **Outdated detection**: superseded by an in-force source, or citing only outdated sources
   (stale dependency, same topic only).
4. **Rank** with trust-weighted Personalized PageRank: `cites` edges carry endorsement, and the
   teleport vector is a trust prior (authority tier × expert validation × client-specific match).
5. **Gate** (`decision.py`): `ANSWER`, `ANSWER_WITH_GAPS`, or `ESCALATE` (conflicting claims,
   low-authority evidence only, no source, or a high-risk topic).
6. **Draft** (`llm.py`): Claude phrases the answer **only from the ranked evidence** (structured
   output, citations checked against the evidence set). Without API credentials, a deterministic
   template answer is used.
7. **Escalate** (`experts.py`): experts ranked by country, topic and load, with a prepared case brief.
8. **Capture** (`capture.py`): the validated answer becomes a knowledge item with evidence links and
   a 12-month review date; it can supersede the source the expert rejected.

## Requirements

Python 3.12 or newer (tested on 3.12 and 3.13; `.python-version` selects 3.13 for uv/pyenv).

## Web interface

```bash
pip install -r requirements.txt       # or: uv sync
python -m payroll_passport.web            # http://127.0.0.1:8000
```

- **Ask** a question with its context (country, pay date, topic, client, employee).
- See the decision, the cited answer, trust-ranked evidence, and outdated / not-applicable sources.
- When escalated, the suggested expert and case brief are shown with a **validation form**. Saving
  stores a knowledge item and re-runs the question.
- **Knowledge base** tab lists every source.

The server works on `payroll_kb.json` in the current directory (created from the sample on first
run; ignored by git). Use `--kb`, `--host`, `--port` to change this. It binds to localhost only and
has no login, so don't expose it on a network as is.

## Command line

```bash

# Ask (add --no-llm to skip the Claude call)
python -m payroll_passport ask "Adult minimum hourly wage?" \
  --country NL --topic min_wage_hourly --date 2026-03-15 --employee adult

# Conflict -> escalation
python -m payroll_passport ask "Travel allowance per km?" --country NL --topic travel_allowance --date 2026-03-15

# Expert validates -> stored as knowledge (work on a copy of the sample)
cp payroll_passport/data/sample_kb.json my_kb.json
python -m payroll_passport validate --kb my_kb.json --expert EXP_NL_1 \
  --question "Travel allowance per km?" --country NL --topic travel_allowance --date 2026-03-15 \
  --answer "Use EUR 0.23/km" --claim travel_allowance_per_km=0.23:EUR \
  --evidence NL_TAX_TRAVEL_2026 --supersedes NL_GUIDANCE_TRAVEL_2026

python -m pytest tests
```

The Claude call uses `claude-opus-5-5` with server-side refusal fallbacks (`fallbacks="default"`)
and reads credentials from `ANTHROPIC_API_KEY` or an `ant auth login` profile.

## Known limitations

- Retrieval is exact topic-tag matching; there is no semantic search or document ingestion yet.
- Sources must be tagged by hand (scope, dates, structured claims).
- Conflict detection compares structured claims only; free-text rules aren't compared.
- Weights and thresholds (`TIER_PRIOR`, `MIN_TOP_SCORE`, ...) are unvalidated guesses. Tune them
  against a set of real questions with known answers.
- If the employee type is left blank, sources limited to a specific employee group (e.g. adults) are
  still used, but the result is downgraded to "answer with gaps" and the missing context is flagged.
- Legal override rules (e.g. a contract may exceed but not undercut a statutory minimum) are not
  modelled; client-specific terms are surfaced alongside the general rule instead.

`demo/pagerank_demo.py` is the standalone experiment comparing naive and trust-adapted PageRank.
