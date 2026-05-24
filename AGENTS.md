# hyph-bench

Benchmark code and datasets for hyphenation pattern generation and optimization.

Read `CLAUDE.md` for current project status, key commands, optimizer architecture, dataset structure, and paper-specific notes.

Use `uv run ...` for Python commands. The project requires Python >=3.10.
Patgen must be available for optimization and evaluation workflows.

Key commands:

- Use `uv run python -m scripts.optimize ...` for optimizer runs.
- Use `uv run python -m scripts.cross_validate ...` for cross-validation.
- Use `make translate_all` to regenerate translate files.

Do not commit changes unless explicitly requested.
Do not launch long optimization sweeps, full benchmark runs, or dataset regeneration unless explicitly requested.
Avoid rewriting generated results, optimizer state, or large dataset files unless the task specifically requires it.
