# TRIDENT — NEAT-based Skill Discovery for MED Framework

TRIDENT uses NEAT / ES-HyperNEAT to autonomously explore and collect skills within MED.
For project details, phase status, and Japanese reference: see `CLAUDE_USER.md` and `forUser/`.

## Tech Stack
- Python 3.13 + Poetry (venv: `.venv`)
- JAX 0.9.2 + TensorNEAT 0.1.0 (GPU: CUDA 12, RTX 4060 8GB)
- FAISS-cpu 1.13.2 + QDax 0.5.0 (MAP-Elites)
- pytest 9.x

## Commands
| Command | Purpose |
|---------|---------|
| `poetry run pytest tests/ -q` | Run all tests (136 tests) |
| `poetry run python scripts/neat_benchmark.py` | NEAT benchmark (30 min) |
| `poetry run python scripts/med_integration_verify.py` | MED integration check |
| `poetry run python scripts/phase0_verify.py` | Environment check |

## Key Paths
| Path | Role |
|------|------|
| `src/interfaces/` | A/B/C-type NEAT interfaces (NeatIndexer, NeatGate, NeatSlotFiller) |
| `src/med_integration/` | MED adapter + ContextSensitiveSearch (AssociationFn) |
| `src/es_hyperneat.py` | ES-HyperNEAT custom extension |
| `src/map_elites_archive.py` | TRIDENTArchive (MAP-Elites via QDax) |
| `src/novelty_search.py` | NoveltyArchive (custom JAX implementation) |
| `tests/` | pytest suite |
| `logs/` | Benchmark JSONL logs |
| `.claude/rules/` | Claude rules (English, path-conditional) |
| `forUser/` | Japanese translations of .claude/ files |

## Behavioral Principles
- Always run `poetry run pytest tests/ -q` before marking any task complete
- Read existing code before modifying
- Do not mark tasks complete until tests pass and behavior is verified
- When context is running low, say so explicitly
- All `.claude/` files and `CLAUDE.md` are in English; `forUser/` has Japanese translations
- When creating or updating any rule or skill, always update the forUser counterpart
