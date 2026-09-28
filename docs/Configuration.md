# Configuration

## Basics

Configuration uses the [`hydra`](https://hydra.cc/docs/intro/) framework (for
config composition only — none of its job management). Config files live under
`config/` and are structured to follow PyTorch Lightning (Trainer, Callbacks,
Loggers), so familiarity with Lightning helps.

The entrypoint loads `config/root.yaml`, a stub that wires together the defaults
below. It does **not** fix an experiment — that must be selected on the command
line via the `experiment` group:

```sh
./main.py train experiment=sge/nuscenes-dev
./main.py train experiment=sge/nuscenes-dev trainer.devices=2   # + an override
```

## Config groups

`root.yaml` composes these groups (each is a subdirectory of `config/` with one
file per option, plus a `default.yaml`):

| group | what it configures |
|---|---|
| `experiment` | **required** — the model, data, and training setup for a run (see below) |
| `paths` | project directories and per-dataset paths (`config/paths/default.yaml`) |
| `data` | dataset/label defaults shared across experiments |
| `trainer` | the Lightning `Trainer` (devices, precision, logger, callbacks, …) |
| `validation` | validation/metric protocol (`null` unless an experiment sets it) |
| `init` | model initialization (e.g. `init.checkpoint` for pre-trained weights) |
| `torch` | torch/CUDA settings (e.g. `torch.compile`, matmul precision) |
| `slurm` | SLURM submission defaults (see [Submission.md](Submission.md)) |
| `machine` | optional per-machine overrides, tracked in git |
| `local` | optional local overrides **not** tracked in git (secrets etc.) |
| `debug` | optional debug conveniences |

`machine`, `local` and `debug` are opt-in (selected on the command line, e.g.
`machine=my-node`), except `local` which is auto-loaded if present.

## Overrides

Any option can be set on the command line; overrides take precedence over files.
One grammar applies everywhere (interactive or `submit`, composed config or a run's
standalone `config.yaml`):

- **Change** an existing option: `trainer.devices=2`
- **Add** a new option: `+model.some_new_flag=true`
- **Select** a config group: `experiment=…`, `machine=…`
- **Delete** an option: `~some.key` *(composed configs only)*

Values use OmegaConf/hydra syntax; see the
[hydra override docs](https://hydra.cc/docs/advanced/override_grammar/basic/) for
lists, nulls, quoting, etc.

The `-c/--config` option controls what is loaded: a path **relative to `config/`**
is composed by hydra (the default is `config/root.yaml`); any **other** path is
loaded as an already-composed standalone file (this is how `-d/--directory` reuses
a run's frozen `config.yaml`). The override grammar above is identical for both.

## Logging and experiment tracking

Metrics are logged through one or more backends selected by the `trainer/logger`
config group (`config/trainer/logger/`); logs are written under the run directory
(see [Runs.md](Runs.md)). The default enables **CSV + TensorBoard**. Available
options: `tensorboard`, `aim`, `mlflow`, `csv`, `wandb`, `none` (disable).

Switch the backend from the command line by selecting a different option for the
group:

```sh
./main.py train experiment=… trainer/logger=mlflow
./main.py train experiment=… trainer/logger=none          # disable logging
```

TensorBoard and CSV work out of the box.  Aim, MLflow, and W&B need their
optional extras — `uv sync --extra aim` / `--extra mlflow` / `--extra wandb`
(see [Setup.md](Setup.md)).  To enable a backend only on your own machine, set
it in `config/local/` (git-ignored) rather than changing the committed default.

## Console output

Two independent things drive what the console shows.

**Log messages and tracebacks** render with
[`rich`](https://github.com/Textualize/rich) only when stdout is an interactive
terminal; when output is redirected (e.g. a SLURM job writing to a file) they fall
back to plain, width-pinned output so the log file stays clean. Two environment
variables override the detection:

- `RICH_LOGS=0|1` — force rich log formatting off/on (default: on iff stdout is a tty).
- `RICH_TRACEBACKS=0|1` — rich vs. plain tracebacks (default: on).

**The progress bar** is a trainer callback, independent of the above. The default
`rich_progress_bar` is always rich and merely goes static off-tty. For long
non-interactive runs, the `log_progress` callback
(`config/trainer/callbacks/log_progress.yaml`) is usually preferable — instead of a
live bar it emits a plain progress line to the logger every N steps. Enable it from a
`machine` or `local` config, disabling the rich bar (a callback set to `null` is
skipped, and Lightning allows only one):

```yaml
# @package _global_

trainer:
  callbacks:
    rich_progress_bar: null      # disable the rich progress bar
    log_progress:                # ...and log a plain progress line instead
      type: LogProgress
      interval: 50
      logger: progress
      log_level: info
```

A `tqdm_progress_bar` callback is also available.

## Experiments (`config/experiment`)

An experiment is the main description of a run — model, data pipeline, optimizer,
and metadata. Experiments override the defaults above and compose reusable
fragments from subdirectories (e.g. `data/`, `model/`, `optimizer/`, `env/`), so
related experiments share pipeline definitions and differ only where they need to.

## Local and machine configs

- **`config/machine`** — machine-local overrides that *are* tracked in git (e.g.
  device counts or paths specific to a known cluster). Select with `machine=<name>`.
- **`config/local`** — overrides that should *not* be in git, for secrets and
  per-checkout settings. For example the `Notify` callback needs credentials that
  must not be committed. Auto-loaded if `config/local/default.yaml` exists.
