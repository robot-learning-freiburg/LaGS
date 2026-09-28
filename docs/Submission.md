# Running and submitting jobs

`main.py` is the entrypoint for all four commands — `train`, `eval`, `predict`,
`preprocess`. Each can be run **interactively** (in the current shell) or
**submitted** to a scheduler. Every run writes a self-contained run directory at `runs/<name>/<version>/` see [Runs.md](Runs.md) for its contents.

> [!TIP]
> Give each training run a distinct, meaningful name by setting `meta.run.name` (e.g.
> `./main.py [submit] train … meta.run.name=<your-new-name>`) rather than relying on
> the random name or reusing the same name for every run — it keeps runs easy to
> find and manage.

## Interactive runs

Run a command directly. The experiment (and any overrides) are given on the
command line — see [Configuration.md](Configuration.md):

```sh
./main.py train experiment=sge/nuscenes-dev            # train
./main.py eval    -d runs/<name>/<version>             # evaluate a run's last.ckpt
./main.py predict -d runs/<name>/<version>             # write predictions
./main.py preprocess experiment=sge/nuscenes-dev       # warm caches only
```

`preprocess` builds and tears down the data modules for all stages without
training, so any first-run indexing/caching is done up front (subsequent runs
reuse the caches under `cache/`).

## Command options

Every command takes config **overrides** as trailing arguments (see
[Configuration.md](Configuration.md)); the other options, documented per command
below, select *what* to load. The same options apply to the matching
`submit <command>` (which additionally takes `--scheduler` and `--dry-run`).

### `train`

Trains a model. Select the experiment (and any changes) via overrides; the config
defaults to `config/root.yaml`.

| option | purpose |
|---|---|
| `-c, --config` | Config to load instead of the default. A path **relative to `config/`** is composed by hydra; any other path is loaded as a standalone file. |
| `--resume PATH[:ckpt]` | Continue a run from its directory: loads its frozen `config.yaml` and `last.ckpt` (append `:<name>` for `<name>.ckpt`). Overrides are **not** allowed. See [Runs.md](Runs.md). |
| `--resume-checkpoint PATH` | Start a **new** run whose weights are initialized from a specific checkpoint. |

To start from pre-trained weights without `--resume-checkpoint`, set
`init.checkpoint=…` as an override (this is how the second training stage is
initialized from the first — see the [README](../README.md)).

```sh
# fresh run
./main.py train experiment=sge/nuscenes-r50-f1p7

# continue a stopped run
./main.py train --resume runs/<name>/<version>
```

### `eval` / `predict`

`eval` scores a checkpoint on the validation data; `predict` writes its
predictions to disk. Both need at least one of `-d` or `-p`.

| option | purpose |
|---|---|
| `-d, --directory` | Run-directory shortcut: load both `<dir>/config.yaml` **and** `<dir>/last.ckpt` — i.e. equivalent to `-c <dir>/config.yaml -p <dir>/last.ckpt`. |
| `-c, --config` | Config to load (composed if relative to `config/`, else standalone). Defaults to `<dir>/config.yaml` under `-d`, otherwise `config/root.yaml`. |
| `-p, --checkpoint` | Model checkpoint (weights) to load. Defaults to `<dir>/last.ckpt` under `-d`. |
| `--strict / --non-strict` | Require checkpoint keys to match the model exactly (default: `--strict`). |

Because `-d` just expands to `-c <dir>/config.yaml -p <dir>/last.ckpt`, passing
`-c` and/or `-p` alongside it overrides either half — so you can run one checkpoint
against a different config, or one config against a specific checkpoint:

```sh
# a run's own config + its last checkpoint (the common case)
./main.py eval -d runs/<name>/<version>

# a specific checkpoint from that run, same config
./main.py eval -d runs/<name>/<version> -p runs/<name>/<version>/best.ckpt

# a standalone checkpoint against a freshly composed config
./main.py eval experiment=sge/nuscenes-r50-f3p5 -p ckpts/sge-r50-nusc-f3p5.ckpt

# write predictions, tolerating checkpoint/model key mismatches
./main.py predict -d runs/<name>/<version> --non-strict
```

### `preprocess`

Takes only `-c, --config` and overrides. It builds and tears down the data pipeline
to warm caches up front; the other commands do this on first run automatically.

```sh
./main.py preprocess experiment=sge/nuscenes-r50-f1p7
```

## Submitted runs

The `submit` group **mirrors the interactive commands exactly**: `submit train`,
`submit eval`, `submit predict` and `submit preprocess` take the same options as
their interactive counterparts ([Command options](#command-options) above), plus
`--scheduler` and `--dry-run`. The difference is only *when* and *where* the run
executes — launching is split into two steps:

1. **Prepare** — freeze the composed config and metadata into a fresh run
   directory (`config.yaml` + `submit.yaml`, the run manifest). Nothing runs yet.
2. **Dispatch** — hand the prepared directory to a **scheduler backend**, selected
   with `--scheduler` (default `slurm`).

The prepared run is executed by re-invoking the entrypoint with a hidden
`--run-dir` pointing at it, which loads everything from the frozen manifest:

```sh
./main.py <command> --run-dir <run_dir>
```

> [!TIP]
> Submitted runs are non-interactive — their stdout is usually redirected to a log
> file rather than a terminal — so it is worth configuring a few things up front via
> [Configuration.md](Configuration.md): choose a metrics/experiment-tracking backend
> ([Logging and experiment tracking](Configuration.md#logging-and-experiment-tracking))
> and switch to the plain, per-step progress line that suits file logs
> ([Console output](Configuration.md#console-output)).

Config overrides can additionally set scheduler parameters, e.g.
`slurm.partition=gpu slurm.nodes=2`.

### `--scheduler slurm` (default)

Generates a batch script (`slurm_submit.sh`) from the run's `submit.yaml` and the
template `config/slurm/submit.sh.template`, then submits it with `sbatch`. The
script runs `srun python main.py <command> --run-dir <run_dir>`, records the job
id, and (via `--signal`/`--requeue` in the template) supports preemption/requeue.

```sh
# submit training on 2 nodes of the 'gpu' partition
./main.py submit train experiment=sge/nuscenes-dev slurm.nodes=2 slurm.partition=gpu

# prepare + generate the script but do NOT sbatch
./main.py submit train experiment=sge/nuscenes-dev --dry-run
```

SLURM settings come from `config/slurm/default.yaml`, overridden by `slurm.*` on
the command line, and are stored in the run's `submit.yaml`. When submitting from
an existing run (`eval`/`predict`/`--resume` off a run directory), the SLURM
settings are inherited from that run's `submit.yaml` and can be overridden with
`slurm.*`.

| `slurm.*` key | maps to | notes |
|---|---|---|
| `job_name` | `--job-name` | defaults to the run name |
| `partition` | `--partition` | |
| `nodes` | `--nodes` | |
| `ntasks_per_node` | `--ntasks-per-node` | `null` → partition default |
| `gres` | `--gres` | e.g. `gpu:4`, `gpu:a100:2` |
| `cpus_per_gpu` / `cpus_per_task` | `--cpus-per-*` | |
| `time` | `--time` | `HH:MM:SS` or `D-HH:MM:SS` |
| `account` / `qos` / `constraint` / `mem` / `exclude` | resp. flag | |
| `sbatch_args` | extra `#SBATCH` lines | list of raw args |

**Resources derived from the run config.** The provided experiments compose an
`env/slurm.yaml` fragment (packaged into the `slurm` group) that derives the
resource request from the rest of the config, so it tracks the training setup
automatically rather than needing to be set by hand:

| `slurm.*` | derived from | effect |
|---|---|---|
| `ntasks_per_node` | `trainer.devices` | one task per GPU |
| `gres` | `trainer.devices` | request `gpu:<devices>` |
| `cpus_per_task` | `data.loader.train.num_workers` | `num_workers + 1` |

(the fragment also sets fixed `nodes` and `mem`). So `trainer.devices=4` already
requests 4 GPUs with 4 tasks per node — no need to touch `gres`/`ntasks_per_node`.
An explicit `slurm.*` override still wins over the derived value.

> [!TIP]
> **Per-machine job setup (`slurm-setup.sh`).** If you need to set anything up on
> the node before the run — load modules, activate the environment, export
> environment variables (proxies, `NCCL_DEBUG`, …) — put it in
> `config/local/slurm-setup.sh`. If that file exists it is sourced inside the job,
> on each node, right before the run command. It lives under `config/local/`
> (git-ignored), so it stays machine-local. Copy the provided template to start:
>
> ```sh
> cp config/local/slurm-setup.sh.example config/local/slurm-setup.sh
> # then edit: module loads, environment activation, env vars, …
> ```
>
> For this repo's environment, that means activating the micromamba env and the
> project `.venv` so the job runs against the same interpreter and extensions as an
> interactive run:
>
> ```sh
> # activate environment
> eval "$(micromamba shell hook --shell=bash)"
> micromamba activate uv-cu118
> source .venv/bin/activate
> ```

### `--scheduler none`

Prepares the run directory and stops — no `slurm_submit.sh`, and a
scheduler-agnostic `submit.yaml` (no `slurm:` block). Use this to run under a
scheduler this framework doesn't natively support, or interactively.

1. **Prepare** the run directory:

   ```sh
   ./main.py submit train experiment=sge/nuscenes-dev --scheduler none
   ```

   This prints the exact command to execute the prepared run, e.g.:

   ```
   ./main.py train --run-dir runs/<name>/<version>
   ```

2. **Execute** it — directly, or from inside your own job script (PBS, LSF, k8s, a
   bare `ssh`, …):

   ```sh
   ./main.py train --run-dir runs/<name>/<version>
   ```

   `--run-dir` loads everything (config, checkpoint/strict, overrides) from the
   prepared manifest, so no other config options are passed at this step.

To later run the same config on SLURM, re-submit with `--scheduler slurm` (which
regenerates a matching script).

## Adding a scheduler backend

Backends live in `src/tracker/main.py`:

1. Add the name to the `SCHEDULERS` tuple (so it appears in `--scheduler`).
2. Handle it in `_dispatch_run(run_dir, command, scheduler, dry_run)` — turn the
   prepared `run_dir` into a submitted job (typically: read `submit.yaml`, render a
   script, submit it). The `slurm` branch is the reference implementation.

Scheduler-specific config (like the `slurm:` block) is written into `submit.yaml`
only for its backend by the `prepare_submit_*` functions in
`src/tracker/config/__init__.py`, gated on the selected `scheduler`.
