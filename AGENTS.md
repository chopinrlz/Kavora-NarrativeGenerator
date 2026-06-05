# AGENTS.md

## Project overview

Kavora Narrative Generator — a Python CLI (planned in `agent.py`) that uses LangChain and Anthropic Claude to generate sample narratives aligned with tags in `tags.csv`. The repo is a sandbox template; `agent.py` is not yet implemented.

## Cursor Cloud specific instructions

### Runtime

- **Python:** 3.12+ works (README mentions 3.14; Cloud VMs ship with 3.12). Use a project virtualenv at `.venv/`.
- **One-time system package:** If `python3 -m venv` fails, install `python3.12-venv` via apt before creating the venv.
- **Activate venv:** `source .venv/bin/activate` (or prefix commands with `.venv/bin/`).

### Dependencies

See `requirements.txt`. Refresh with:

```bash
python3 -m venv .venv 2>/dev/null || true
.venv/bin/pip install -r requirements.txt
```

### Running / verifying

| Action | Command |
|--------|---------|
| Run CLI (placeholder) | `.venv/bin/python agent.py` |
| Syntax check | `.venv/bin/python -m py_compile agent.py` |

There is no lint config, test suite, Makefile, or Docker setup in this repo yet.

### Data files

- `tags.csv` has a UTF-8 BOM. Open with `encoding="utf-8-sig"` when using `csv.DictReader`.
- 147 canonical tags across Legal, Financial, Insurance, Healthcare, and Government domains.

### Modes (planned)

- **Test mode:** No API key; generate random word narratives locally.
- **Real mode:** Requires `ANTHROPIC_API_KEY` in environment (LangChain Anthropic convention).

### Services

Single-process CLI only — no local servers, databases, or docker-compose services.
