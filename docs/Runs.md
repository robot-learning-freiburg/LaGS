# Runs and metadata

Every command that does real work writes a **run directory** under `runs/`. A
run directory is self-contained: it captures the exact configuration, code
state, and environment a run was executed with, so runs are reproducible and
can be resumed, evaluated, or inspected later.

## Naming and versioning

A training run lives at:

```
runs/<name>/<version>/
```

- **`<name>`** — `meta.run.name` from the config, or a random human-readable
  name (e.g. `witty_koala`) if unset.
- **`<version>`** — a timestamp, `YYYY.MM.DDTHH.MM.SS`, generated at
  launch/submission time.

> [!TIP]
> Give each training run a distinct, meaningful `meta.run.name` — in the experiment
> config, or as an override (e.g. `meta.run.name=my-experiment`) — rather than
> relying on the random name or reusing the same name for every run. It keeps runs
> easy to find, filter, and manage.

`eval`, `predict` and `preprocess` do not create a new top-level run; they nest an
auto-incrementing **sub-run** under the run they operate on:

```
runs/<name>/<version>/eval/0/
runs/<name>/<version>/predict/0/     # predict/1, predict/2, … on repeat
```

Each run is tagged automatically with `mode=<command>` (in `meta.run.tags`),
alongside any tags set in the config — useful for filtering in the logger UI.

## Run directory contents

```
runs/<name>/<version>/
├── config.yaml     # frozen, fully-composed config this run used
├── state.yaml      # environment + git state captured at execution
├── git/            # HEAD sha + working-tree diffs (see below)
├── run.log         # file copy of the run's console log
├── submit.yaml     # run manifest — only present for submitted runs
├── last.ckpt       # checkpoints (names depend on the trainer/callbacks)
├── eval/N, predict/N   # inference sub-runs (as above)
└── resume-<...>/   # state captured on each resume / requeue
```

For SLURM-submitted runs you'll also see `slurm_submit.sh`, `slurm-<jobid>.out`
and `slurm_job_id.txt`.

### `config.yaml`

The complete composed configuration, frozen at launch. A few fields are stripped
because they are environment- or invocation-specific and re-derived on load:
command-line `args`, `paths.root`, and the `slurm` block. Paths are stored as
interpolations (e.g. `${paths.runs}/…`), so a run directory stays valid if the
project root moves. `eval`/`predict` load this file to reconstruct the model and
data pipeline.

### `state.yaml` and `git/`

Captured when the run actually executes (not at submit time), recording:

- timestamp, host, container host, working directory and project root;
- environment variables;
- git `HEAD` and whether there were staged / unstaged / untracked changes.

The `git/` directory stores the `HEAD` sha and the actual diffs
(`staged.diff`, `unstaged.diff`, `untracked.diff`), so the exact code state —
including uncommitted changes — is recoverable. On resume, the new state is
compared against the original and any differences (host, environment, git) are
logged as warnings.

### `submit.yaml` (the run manifest)

Written by `submit` (see [Submission.md](Submission.md)). It records *how* the run
was prepared — the command, the config source, overrides, and checkpoint/strict
settings — plus any scheduler-specific settings for the backend it was submitted
with (for example a `slurm:` block for the SLURM backend). Executing a prepared run
(`--run-dir`) reads this file to restore those settings. It is written for every
backend, including the scheduler-agnostic `--scheduler none` path.

## Resuming

Resume a training run from its directory:

```sh
./main.py train --resume runs/<name>/<version>
./main.py train --resume runs/<name>/<version>:on_exception   # a specific ckpt
```

`--resume` loads `config.yaml` and, by default, `last.ckpt` from the run directory
(append `:<name>` to pick `<name>.ckpt`). Overrides are **not** allowed with
`--resume` — the run continues with its frozen config. Each resume creates a
`resume-<timestamp>/` subdirectory holding freshly captured state (for the
before/after comparison above). To resume with a different config or checkpoint
from outside the run directory, start a new run with `--config` +
`--resume-checkpoint` instead.

To resume a **submitted** run, use `submit` with `--resume` (scheduler settings may
be adjusted via overrides, e.g. `slurm.*` for the SLURM backend):

```sh
./main.py submit train --resume runs/<name>/<version> slurm.nodes=4
```

On a SLURM **requeue** (preemption), the job automatically resumes from
`last.ckpt` and records its state under a `resume-…-rq<N>` subdirectory.
