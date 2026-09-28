# Documentation

- [Setup](Setup.md) — environments, installation, directories, datasets, and data
  preparation. **Start here.**
- [Configuration](Configuration.md) — the hydra-based config system, config groups,
  the override grammar, and experiments.
- [Runs](Runs.md) — run-directory anatomy, captured metadata/state, naming and
  versioning, and resuming.
- [Submission](Submission.md) — running commands interactively and submitting them
  to a scheduler (SLURM or a scheduler-agnostic path for anything else).
- [Model Zoo](ModelZoo.md) — the provided checkpoints and their experiment configs.

For the offline tooling *around* training — evaluation, tracking baselines,
visualization, and dataset preparation — see [`scripts/`](../scripts/README.md),
which has its own README per tool.
