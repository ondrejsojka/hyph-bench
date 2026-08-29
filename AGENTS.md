# hyph-bench

Benchmark code and datasets for hyphenation pattern generation and optimization.

Read `README.md` for setup, dataset layout, optimizer workflows, the paper reproduction protocol, and dataset licenses.

Use `uv run ...` for Python commands. The project requires Python >=3.10.
Patgen must be available. Pass a non-default binary with `--patgen`, or `PATGEN_BIN` where a batch script supports it.
Large datasets need the high-capacity build (`/home/dev/patgen-10x` here); the packaged `patgen` aborts on them.

Key commands:

- Use `uv run python -m scripts.optimize_validation ...` for held-out optimizer runs (train/validation/test).
- Use `uv run python -m scripts.optimize_shared_parameters ...` for the shared-parameter search.
- Use `uv run python -m scripts.cross_validate ...` for cross-validation.
- `scripts.optimize` is in-sample and legacy; do not use it for camera-ready results.
- Use `make translate_all` to regenerate translate files.

There is no test suite and no linter. Verify changes with a short smoke run on a small dataset such as `th/orchid`, not with invented test commands.

Do not commit changes unless explicitly requested.
Do not launch long optimization sweeps, full benchmark runs, or dataset regeneration unless explicitly requested.
Avoid rewriting generated results, optimizer state, or large dataset files unless the task specifically requires it.
