# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import datetime
import filecmp
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import IO, Any, List, Optional, Union

import hydra
import lightning as L
import randomname
import torch
from omegaconf import OmegaConf

from ..utils import log as logutils
from ..utils import mp, repo, slurm, sys
from . import paths, registry, types, utils

DEFAULT_CONFIG = paths.config / "root.yaml"

_ENV_PATH_RUN = "PLTRACKER_RUN_PATH"
_ENV_PATH_STATE = "PLTRACKER_STATE_PATH"


slurm_submit_template_path = paths.config / "slurm" / "submit.sh.template"

log = logutils.get_logger(__name__)
_config: OmegaConf | None = None  # pylint: disable=invalid-name

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("len", len)


def load(path: Union[str, Path], overrides: List[str] = None) -> OmegaConf:
    """
    Load config from file and apply basic defaults.

    Note: This function is for interactive/non-SLURM use. For SLURM submitted jobs,
    use load_preconfigured() instead.
    """
    conf = _load_from_file(path, overrides)

    # configure pytorch/cuda settings
    _initialize(conf)

    return conf


def _load_config_defaults() -> OmegaConf:
    # minimal defaults: base paths and environment setup
    defaults = {
        "paths": {
            "root": str(paths.root),
        },
        "seed": {
            "value": random.randrange(0xFFFFFFFF),
            "workers": True,
        },
        "slurm": OmegaConf.load(paths.config / "slurm" / "default.yaml"),
    }

    return OmegaConf.create(defaults)


def _load_from_file(path: Union[str, Path], overrides: List[str] = None) -> OmegaConf:
    # sanitize inputs
    path = Path(path).absolute()
    overrides = overrides or []

    # minimal defaults: base paths and environment setup
    defaults = _load_config_defaults()

    # if this is a config relative to the hydra config dir: assume hydra
    # config, else assuma an already composed config and load it directly via
    # omegaconf
    if paths.config in path.parents:
        log.info("Loading hydra config from '%s'", path)

        hydra.initialize(
            config_path=os.path.relpath(paths.config, Path(__file__).parent),
            version_base="1.3",
            caller_stack_depth=1,
            job_name="main",
        )

        conf = hydra.compose(str(path.relative_to(paths.config)), overrides=overrides)
        conf = OmegaConf.merge(defaults, conf)

    else:
        log.info("Loading full config from file '%s'", path)

        # Accept hydra-style '+'/'++' key prefixes here too, so a single override
        # grammar works whether the config is composed by hydra (above, where new
        # keys require a leading '+') or loaded as a standalone file (this branch,
        # where OmegaConf.from_dotlist adds new keys unconditionally). Without
        # stripping, '+foo=1' would create a literal key named '+foo'.
        overrides = [o.lstrip("+") for o in overrides]
        overrides = OmegaConf.from_dotlist(overrides)

        conf = OmegaConf.load(path)
        conf = OmegaConf.merge(defaults, conf, overrides)

    return conf


def _initialize(conf: OmegaConf) -> None:
    # apply random seed
    L.seed_everything(conf.seed.value, workers=conf.seed.workers)

    # configure pytorch settings
    torch.set_float32_matmul_precision(
        OmegaConf.select(conf, "torch.float32_matmul_precision", default="highest")
    )


def save(
    conf: OmegaConf,
    f: Union[str, Path, IO[Any]],
    resolve: bool = False,
):
    conf = unlock(copy.deepcopy(conf))

    # Some fields are only included in the config so that we can push them to
    # the loggers and use them, e.g., for filtering in Aim or WandB. They are
    # automatically set/overridden when the config is loaded. So remove them
    # here.
    del conf.args  # command line arguments
    del conf.paths.root  # root path for relative paths
    del conf.slurm

    OmegaConf.save(config=conf, f=f, resolve=resolve)


def _build_run_path(name: str, version: str, base: str | Path = "${paths.runs}") -> str:
    return f"{base}/{name}/{version}"


def _update_tags(conf: OmegaConf, args: OmegaConf):
    conf.meta.run = OmegaConf.merge({"tags": []}, conf.meta.run)
    meta = conf.meta.run

    # remove old mode tags
    meta.tags = [tag for tag in meta.tags if not tag.startswith("mode=")]

    # add new mode tag
    meta.tags += [f"mode={args.command}"]


def _update_config_for_tracking(
    conf: OmegaConf, args: OmegaConf, timestamp: datetime.datetime
) -> OmegaConf:
    if args.command not in ["train", "eval", "predict", "preprocess"]:
        raise ValueError(f"unknown mode '{args.command}'")

    # store the command-line arguments
    conf.args = args

    # get run metadata
    conf = OmegaConf.merge(conf, {"meta": {"run": {}}})
    meta = conf.meta.run

    # get the run name or create a random one
    if "name" not in meta or meta.name is None:
        meta.name = randomname.get_name(sep="_")

    # get the run version or generate a new one
    if "version" not in meta or meta.version is None:
        meta.version = timestamp.strftime("%G.%m.%dT%H.%M.%S")

    # construct base run output path
    if "current_run" not in conf.paths or conf.paths.current_run is None:
        conf.paths.current_run = _build_run_path(meta.name, meta.version)

    # if the path exists in training mode: abort
    if args.command == "train":
        if Path(conf.paths.current_run).exists():
            raise RuntimeError("paths.current_run already exists, aborting")

    # for other modes: find a new sub-version
    else:
        mode_path = Path(conf.paths.current_run) / args.command

        if mode_path.exists():
            mode_versions = [x.name for x in mode_path.iterdir() if x.is_dir()]
            mode_versions = [int(x, base=10) for x in mode_versions if x.isdigit()]
            mode_versions = sorted(mode_versions)

            mode_version = mode_versions[-1] + 1
        else:
            mode_version = 0

        base_version = meta.version.split("/")[0]
        mode_version = f"{args.command}/{mode_version}"

        meta.version = f"{base_version}/{mode_version}"
        conf.paths.current_run = f"{conf.paths.current_run}/{mode_version}"

    # manage automatic tags
    _update_tags(conf, args)

    return conf


def _add_state_metadata(conf: OmegaConf, state: OmegaConf) -> OmegaConf:
    conf.meta.timestamp = state.timestamp
    conf.meta.git = state.git
    conf.meta.env = state.env
    conf.meta.host = state.host
    conf.meta.container_host = state.container_host

    return conf


def _setup_logging(
    run_dir: Path,
    action: str = "storing",
    submit_info: OmegaConf | None = None,
) -> None:
    """
    Set up file logging and log basic run information.

    Args:
        run_dir: Run directory path
        action: Action description ("storing", "continuing", "resuming", "starting", etc.)
        submit_info: Optional submit metadata (for preconfigured runs)
    """
    logutils.add_file_handler(run_dir / "run.log")

    # Load config to get run metadata
    conf = OmegaConf.load(run_dir / "config.yaml")

    log.info("%s run at '%s'", action.capitalize(), run_dir)
    log.info("Run name: '%s'", conf.meta.run.name)
    log.info("Run description: '%s'", conf.meta.run.description)
    log.info("Run tags: %s", conf.meta.run.tags)

    if submit_info is not None:
        log.info("Original submit time: %s", submit_info.timestamp)

    if slurm.is_slurm_job():
        log.info("SLURM job ID: %s", slurm.get_job_id())
        if slurm.is_requeue():
            log.info("Restart count: %d", slurm.get_restart_count())


def _save_state(
    conf: OmegaConf,
    args: OmegaConf,
    timestamp: datetime.datetime,
    state_dir: Path,
) -> None:
    """
    Collect system state and save to disk.

    Args:
        conf: Configuration object
        args: Command-line arguments
        timestamp: Timestamp for state collection
        state_dir: Directory to save state files
    """
    # Collect state
    state = _collect_state(conf, args, timestamp, git_path=state_dir / "git")

    # Save state
    OmegaConf.save(state, state_dir / "state.yaml")


def _merge_state_metadata(conf: OmegaConf, state_dir: Path) -> OmegaConf:
    """
    Load state from disk and merge metadata into config.

    Args:
        conf: Configuration object
        state_dir: Directory containing state.yaml

    Returns:
        Updated configuration with state metadata
    """
    state = OmegaConf.load(state_dir / "state.yaml")

    conf = unlock(conf)
    conf = _add_state_metadata(conf, state)
    conf = lock(conf)

    return conf


def _interactive_setup_resume(conf: OmegaConf, args: OmegaConf) -> OmegaConf:
    """
    Set up interactive resume run: load config, create resume directory, save state.
    Warns on state changes compared to original run.
    """
    # get timestamp
    timestamp = datetime.datetime.now()

    # store the command-line arguments
    conf = unlock(conf)
    conf.args = args
    conf = lock(conf)

    # set up output directory
    path = Path(conf.paths.current_run)
    if not path.exists():
        raise RuntimeError(f"run path '{path}' does not exist, cannot resume run")

    # set up logging and log run info
    _setup_logging(path, action="resuming")

    # create resume subdirectory for state
    timestr = timestamp.strftime("%G.%m.%dT%H.%M.%S")
    state_dir = path / f"resume-{timestr}"
    state_dir.mkdir()

    # collect and save state
    _save_state(conf, args, timestamp, state_dir=state_dir)

    # check for differences in state and config and warn if there are
    _check_state_changes(path, state_dir)

    # merge state metadata into config
    conf = _merge_state_metadata(conf, state_dir=state_dir)

    set_global_config(conf)

    # set environment variables for other ranks
    os.environ[_ENV_PATH_RUN] = str(path)
    os.environ[_ENV_PATH_STATE] = str(path)

    return conf


def _interactive_setup_new(
    conf: OmegaConf, args: OmegaConf, resume: bool = False
) -> OmegaConf:
    """
    Set up interactive run: create run directory, save config and state.
    Delegates to _interactive_setup_resume if resume=True.
    """

    # for interactive runs, rank 0 prepares everything before spawning the
    # other ranks
    if mp.rank != 0:
        run_dir = Path(os.environ[_ENV_PATH_RUN])
        state_dir = Path(os.environ[_ENV_PATH_STATE])

        conf = _load_from_file(run_dir / "config.yaml")

        conf = unlock(conf)
        conf.args = args
        conf = lock(conf)

        conf = _merge_state_metadata(conf, state_dir=state_dir)

        set_global_config(conf)

        return conf

    # if we resuming the run. we have already set up the config for tracking
    # and can continue from where we left off
    if resume:
        return _interactive_setup_resume(conf, args)

    # get timestamp
    timestamp = datetime.datetime.now()

    # set up metadata fields and current_run path
    conf = unlock(conf)
    conf = _update_config_for_tracking(conf, args, timestamp)
    conf = lock(conf)

    # set up output directory
    path = Path(conf.paths.current_run)
    path.mkdir(parents=True, exist_ok=False)

    # store full config
    save(conf, path / "config.yaml")

    # set up logging and log run info
    _setup_logging(path, action="starting")

    # collect and save state
    _save_state(conf, args, timestamp, state_dir=path)

    # merge state metadata into config
    conf = _merge_state_metadata(conf, state_dir=path)

    set_global_config(conf)

    # set environment variables for other ranks
    os.environ[_ENV_PATH_RUN] = str(path)
    os.environ[_ENV_PATH_STATE] = str(path)

    return conf


def init(
    conf: Path | str,
    overrides: tuple[str],
    command: str,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> OmegaConf:
    # collect command line arguments
    args = OmegaConf.create()
    args.command = command
    args.config = str(conf)
    args.overrides = overrides
    args.checkpoint = str(checkpoint) if checkpoint else None
    args.resume = resume

    # load config and set up experiment tracking
    conf = load(conf, overrides)
    conf = _interactive_setup_new(conf, args, resume)

    return conf


def lock(conf: OmegaConf) -> OmegaConf:
    OmegaConf.set_struct(conf, True)
    OmegaConf.set_readonly(conf, True)

    return conf


def unlock(conf: OmegaConf) -> OmegaConf:
    OmegaConf.set_struct(conf, False)
    OmegaConf.set_readonly(conf, False)

    return conf


def _collect_state(
    conf: OmegaConf,
    args: OmegaConf,
    timestamp: datetime.time,
    git_path: Path | None = None,
) -> OmegaConf:
    # get and store git state
    repo_state = repo.git_state()

    if git_path is None:
        git_path = Path(conf.paths.current_run) / "git"

    git_path.mkdir()

    with open(git_path / "HEAD", "w", encoding="utf-8") as fd:
        fd.write(repo_state.sha)

    with open(git_path / "staged.diff", "w", encoding="utf-8") as fd:
        fd.write(repo_state.diff_staged)

    with open(git_path / "unstaged.diff", "w", encoding="utf-8") as fd:
        fd.write(repo_state.diff_unstaged)

    with open(git_path / "untracked.diff", "w", encoding="utf-8") as fd:
        fd.write(repo_state.diff_untracked)

    # try to get the hostname of the container's host
    container_host = sys.get_container_hostname()

    # collect environment state
    state = {
        "timestamp": timestamp.astimezone().isoformat(),
        "host": platform.node(),
        "container_host": container_host,
        "paths": {
            "root": conf.paths.root,
            "cwd": str(Path.cwd()),
        },
        "git": {
            "head": repo_state.sha,
            "changes": {
                "staged": bool(repo_state.diff_staged),
                "unstaged": bool(repo_state.diff_unstaged),
                "untracked": bool(repo_state.diff_untracked),
            },
        },
        "args": args,
        "env": dict(os.environ.items()),
    }

    return OmegaConf.create(state)


def _check_state_changes(orig_path: Path, new_path: Path) -> None:
    # check for differences in env state
    orig_state = OmegaConf.load(orig_path / "state.yaml")
    new_state = OmegaConf.load(new_path / "state.yaml")

    if orig_state.host != new_state.host:
        log.warning("State changes detected: host")

    if orig_state.container_host != new_state.container_host:
        log.warning("State changes detected: container host")

    if orig_state.paths != new_state.paths:
        log.warning("State changes detected: paths")

    if orig_state.env != new_state.env:
        log.warning("State changes detected: environment")

    # check for differences in git state
    if not filecmp.cmp(orig_path / "git" / "HEAD", new_path / "git" / "HEAD"):
        log.warning("State changes detected: git HEAD")

    if not filecmp.cmp(
        orig_path / "git" / "staged.diff", new_path / "git" / "staged.diff"
    ):
        log.warning("State changes detected: git staged changes")

    if not filecmp.cmp(
        orig_path / "git" / "unstaged.diff", new_path / "git" / "unstaged.diff"
    ):
        log.warning("State changes detected: git unstaged changes")

    if not filecmp.cmp(
        orig_path / "git" / "untracked.diff", new_path / "git" / "untracked.diff"
    ):
        log.warning("State changes detected: git untracked changes")


def set_global_config(conf: OmegaConf) -> None:
    # pylint: disable-next=global-statement
    global _config

    assert _config is None

    _config = utils.copy(conf, readonly=True)


def get() -> OmegaConf:
    assert _config is not None
    return _config


def _make_path_relative(path: Union[str, Path, None]) -> str | None:
    """
    Convert an absolute path to be relative to paths.root.

    Args:
        path: Path to convert (can be None)

    Returns:
        String path relative to paths.root, or None if input is None
    """
    if path is None:
        return None

    path = Path(path).resolve()
    try:
        return str(path.relative_to(paths.root))
    except ValueError:
        # Path is outside paths.root, return absolute path
        return str(path)


def resolve_relative_path(path: Union[str, Path, None]) -> Path | None:
    """
    Resolve a path that may be relative to paths.root.

    Args:
        path: Path to resolve (can be None or already absolute)

    Returns:
        Absolute Path object, or None if input is None
    """
    if path is None:
        return None

    path = Path(path)
    if path.is_absolute():
        return path

    return (paths.root / path).resolve()


def _resolve_slurm_config(
    overrides: List[str],
    parent_run_dir: Path | None = None,
    fallback: OmegaConf | None = None,
) -> dict:
    """
    Resolve the SLURM submission config for a new submission.

    The SLURM settings a run was submitted with are stored in its submit.yaml
    (they are stripped from config.yaml on save, see save()). When submitting
    from an existing run directory we therefore inherit them from that
    submit.yaml as the base -- so resume/eval/predict reuse the partition,
    time, account, etc. the run was originally submitted with -- and then apply
    any slurm.* overrides on top.

    Args:
        overrides: Config overrides (only slurm.* entries are applied here).
        parent_run_dir: Existing run directory to inherit SLURM settings from,
            or None for a standalone submission.
        fallback: Base SLURM config to use when no parent submit.yaml is
            available. Defaults to the packaged slurm/default.yaml.

    Returns:
        Resolved SLURM config as a plain (resolved) container.
    """
    slurm_base = None

    if parent_run_dir is not None:
        parent_submit_yaml = parent_run_dir / "submit.yaml"
        if parent_submit_yaml.exists():
            parent_submit_info = OmegaConf.load(parent_submit_yaml)
            if "slurm" in parent_submit_info:
                slurm_base = OmegaConf.create(parent_submit_info.slurm)

    if slurm_base is None:
        slurm_base = (
            fallback
            if fallback is not None
            else OmegaConf.load(paths.config / "slurm" / "default.yaml")
        )

    # Apply slurm.* overrides on top of the base.
    if overrides:
        overrides_conf = OmegaConf.from_dotlist(overrides)
        slurm_base = OmegaConf.merge(slurm_base, overrides_conf.get("slurm", {}))

    return OmegaConf.to_container(slurm_base, resolve=True)


def prepare_submit_fit(
    conf: Union[str, Path, None],
    overrides: List[str],
    resume: Union[str, Path, None] = None,
    resume_checkpoint: Union[str, Path, None] = None,
    scheduler: str = "slurm",
) -> Path:
    """
    Prepare run directory for training submission.

    Handles:
    - New training
    - Pre-trained weights (via init.checkpoint in config/overrides)
    - Resume training (resume=path[:checkpoint])

    Args:
        conf: Config file path (ignored if resume is set)
        overrides: Config overrides (including init.checkpoint, slurm.* settings)
        resume: Resume from run directory (format: "path" or "path:checkpoint")
        resume_checkpoint: Checkpoint to resume from (for new runs)
        scheduler: Scheduler backend the run is prepared for.

    Returns:
        run_dir: Path to created run directory
    """

    # Handle resume mode
    if resume is not None:
        return prepare_submit_fit_resume(resume, overrides, scheduler=scheduler)

    conf_path = Path(conf)

    # 1. Load config with overrides
    conf = _load_from_file(conf_path, overrides)

    # 2. Extract SLURM config before updating for tracking (SLURM backend only)
    slurm_config = (
        _resolve_slurm_config(overrides, fallback=conf.slurm)
        if scheduler == "slurm"
        else None
    )

    # 3. Generate run metadata
    timestamp = datetime.datetime.now()
    args = OmegaConf.create()
    args.command = "train"
    args.config = str(conf_path)
    args.overrides = overrides
    args.resume = False

    conf = unlock(conf)
    conf = _update_config_for_tracking(conf, args, timestamp)
    conf = lock(conf)

    # 4. Create run directory
    run_dir = Path(conf.paths.current_run)
    run_dir.mkdir(parents=True, exist_ok=False)

    # 5. Save static config
    save(conf, run_dir / "config.yaml")

    # 6. Save submit metadata
    submit_info = {
        "command": "train",
        "timestamp": timestamp.isoformat(),
        "config_path": _make_path_relative(conf_path),
        "overrides": overrides,
        "resume_checkpoint": _make_path_relative(resume_checkpoint),
    }

    if slurm_config is not None:
        submit_info["slurm"] = slurm_config

    OmegaConf.save(submit_info, run_dir / "submit.yaml")

    log.info("Prepared training run: %s", _make_path_relative(run_dir))

    return run_dir


def prepare_submit_fit_resume(
    resume: str,
    overrides: List[str],
    scheduler: str = "slurm",
) -> Path:
    """
    Prepare resume directory for resubmission.

    Args:
        resume: Run directory path, optionally with :checkpoint suffix
        overrides: Config overrides (e.g., slurm.nodes=4)
        scheduler: Scheduler backend the resume is prepared for (see prepare_submit_fit).

    Returns:
        resume_dir: Resume subdirectory to be passed as --run-dir
    """
    # 1. Parse resume argument
    run_dir, _, resume_checkpoint = resolve_resume_config(resume)

    # 2. Load SLURM settings from the parent run's submit.yaml (SLURM backend only)
    slurm_config = (
        _resolve_slurm_config(overrides, parent_run_dir=run_dir)
        if scheduler == "slurm"
        else None
    )

    # 3. Create resume subdirectory
    timestamp = datetime.datetime.now()
    timestr = timestamp.strftime("%G.%m.%dT%H.%M.%S")
    resume_dir = run_dir / f"resume-{timestr}"
    resume_dir.mkdir()

    # 4. Save resume submit info
    submit_info = {
        "command": "train",
        "timestamp": timestamp.isoformat(),
        "parent_run_dir": _make_path_relative(run_dir),
        "resume_checkpoint": _make_path_relative(resume_checkpoint),
    }

    if slurm_config is not None:
        submit_info["slurm"] = slurm_config

    OmegaConf.save(submit_info, resume_dir / "submit.yaml")

    log.info("Prepared resume for: %s", _make_path_relative(run_dir))
    log.info("Resume checkpoint: %s", _make_path_relative(resume_checkpoint))
    log.info("Resume state directory: %s", _make_path_relative(resume_dir))

    # Return resume_dir as the run directory for SLURM submission
    # This will be passed as --run-dir to the training command
    return resume_dir


def _prepare_submit_inference(
    command: str,
    conf: Union[str, Path, None],
    checkpoint: Union[str, Path, None],
    directory: Optional[Path],
    strict: bool,
    overrides: List[str],
    scheduler: str = "slurm",
) -> Path:
    """
    Prepare eval/predict sub-run directory.

    Handles:
    - Inference from directory (uses directory config)
    - Inference from directory with config override
    - Inference from standalone config + checkpoint

    Args:
        command: "eval" or "predict"
        conf: Config file path
        checkpoint: Checkpoint path
        directory: Parent run directory
        strict: Strict checkpoint loading
        overrides: Config overrides (including slurm.* settings)

    Returns:
        run_dir: Path to inference run directory
    """
    timestamp = datetime.datetime.now()

    # Determine parent run directory
    parent_run_dir = Path(directory) if directory is not None else None

    # Load base config
    if directory is not None:
        if not parent_run_dir.exists():
            raise FileNotFoundError(f"Directory not found: {parent_run_dir}")

        if conf is None:
            conf = parent_run_dir / "config.yaml"

        # Set checkpoint path
        if checkpoint is None:
            checkpoint = parent_run_dir / "last.ckpt"

    if conf is None:
        conf = DEFAULT_CONFIG

    conf_path = conf
    conf = _load_from_file(conf, overrides)

    # Inherit the SLURM config from the parent run's submit.yaml (when running
    # from a run directory), falling back to the loaded config otherwise.
    # Resolved and written only for the SLURM backend.
    slurm_config = (
        _resolve_slurm_config(
            overrides, parent_run_dir=parent_run_dir, fallback=conf.slurm
        )
        if scheduler == "slurm"
        else None
    )

    # Create args and update config for tracking
    args = OmegaConf.create()
    args.command = command
    args.config = str(conf_path)
    args.overrides = overrides
    args.checkpoint = str(checkpoint)
    args.resume = False

    conf = unlock(conf)
    conf = _update_config_for_tracking(conf, args, timestamp)
    conf = lock(conf)

    run_dir = Path(conf.paths.current_run)
    run_dir.mkdir(parents=True, exist_ok=False)

    # Update paths to point to inference directory
    conf = unlock(conf)
    conf.paths.current_run = str(run_dir)
    conf = lock(conf)

    # Save config
    save(conf, run_dir / "config.yaml")

    # Save submit metadata
    submit_info = {
        "command": command,
        "timestamp": timestamp.isoformat(),
        "config_path": _make_path_relative(conf_path),
        "parent_run": _make_path_relative(parent_run_dir),
        "directory": _make_path_relative(directory),
        "checkpoint": _make_path_relative(checkpoint),
        "strict": strict,
        "overrides": overrides,
    }

    if slurm_config is not None:
        submit_info["slurm"] = slurm_config

    OmegaConf.save(submit_info, run_dir / "submit.yaml")

    log.info("Prepared %s run: %s", command, _make_path_relative(run_dir))
    log.info("Checkpoint: %s", _make_path_relative(checkpoint))

    return run_dir


def prepare_submit_eval(
    conf: Union[str, Path, None],
    checkpoint: Union[str, Path, None],
    directory: Optional[Path],
    strict: bool,
    overrides: List[str],
    scheduler: str = "slurm",
) -> Path:
    """Prepare eval sub-run directory (wrapper for _prepare_submit_inference)."""
    return _prepare_submit_inference(
        "eval", conf, checkpoint, directory, strict, overrides, scheduler=scheduler
    )


def prepare_submit_predict(
    conf: Union[str, Path, None],
    checkpoint: Union[str, Path, None],
    directory: Optional[Path],
    strict: bool,
    overrides: List[str],
    scheduler: str = "slurm",
) -> Path:
    """Prepare predict sub-run directory (wrapper for _prepare_submit_inference)."""
    return _prepare_submit_inference(
        "predict", conf, checkpoint, directory, strict, overrides, scheduler=scheduler
    )


def prepare_submit_preprocess(
    conf: Union[str, Path],
    overrides: List[str],
    scheduler: str = "slurm",
) -> Path:
    """
    Prepare run directory for preprocess submission.

    Args:
        conf: Config file path
        overrides: Config overrides (including slurm.* settings)
        scheduler: Scheduler backend the run is prepared for.

    Returns:
        run_dir: Path to created run directory
    """
    conf_path = Path(conf)

    # 1. Load config with overrides
    conf = _load_from_file(conf_path, overrides)

    # 2. Extract SLURM config (SLURM backend only)
    slurm_config = (
        _resolve_slurm_config(overrides, fallback=conf.slurm)
        if scheduler == "slurm"
        else None
    )

    # 3. Generate run metadata
    timestamp = datetime.datetime.now()
    args = OmegaConf.create()
    args.command = "preprocess"
    args.config = str(conf_path)
    args.overrides = overrides
    args.checkpoint = None
    args.resume = False

    conf = unlock(conf)
    conf = _update_config_for_tracking(conf, args, timestamp)
    conf = lock(conf)

    # 4. Create run directory
    run_dir = Path(conf.paths.current_run)
    run_dir.mkdir(parents=True, exist_ok=False)

    # 5. Save static config
    save(conf, run_dir / "config.yaml")

    # 6. Save submit metadata
    submit_info = {
        "command": "preprocess",
        "timestamp": timestamp.isoformat(),
        "config_path": _make_path_relative(conf_path),
        "overrides": overrides,
    }

    if slurm_config is not None:
        submit_info["slurm"] = slurm_config

    OmegaConf.save(submit_info, run_dir / "submit.yaml")

    log.info("Prepared preprocess run: %s", _make_path_relative(run_dir))

    return run_dir


def resolve_resume_config(resume: str) -> tuple[Path, Path, Path | None]:
    """
    Resolve resume string into config path and checkpoint.

    Args:
        resume: Resume string in format "path" or "path:checkpoint_name"

    Returns:
        (run_dir, config_path, checkpoint_path): Paths to run directory, config and checkpoint
    """
    parts = resume.split(":", maxsplit=1)
    run_dir = Path(parts[0])
    ckpt_name = parts[1] if len(parts) > 1 else "last"

    ckpt = run_dir / f"{ckpt_name}.ckpt"
    conf = run_dir / "config.yaml"

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if not conf.exists():
        raise FileNotFoundError(f"No config.yaml in: {run_dir}")

    if not ckpt.exists():
        if ckpt_name == "last":
            log.warning("No last checkpoint, Starting from scratch")
            ckpt = None
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_name}")

    return run_dir, conf, ckpt


def find_checkpoint(run_dir: Path, prefer_hpc: bool = False) -> Path | None:
    """
    Find checkpoint to resume from.

    Args:
        run_dir: Run directory
        prefer_hpc: If True, prefer hpc_ckpt_*.ckpt over last.ckpt

    Returns:
        checkpoint path or None if no checkpoint found
    """
    run_dir = Path(run_dir)

    if prefer_hpc:
        # Look for HPC checkpoints first
        hpc_ckpts = list(run_dir.glob("hpc_ckpt_*.ckpt"))
        if hpc_ckpts:
            # Return None so Lightning auto-finds it
            return None

        # Fall back to last.ckpt
        last_ckpt = run_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt

        log.warning("No checkpoint found in %s", run_dir)
        return None

    # Just look for last.ckpt
    last_ckpt = run_dir / "last.ckpt"
    if last_ckpt.exists():
        return last_ckpt

    log.warning("No last.ckpt found in %s", run_dir)
    return None


def _preconfigured_determine_dirs(
    run_dir: Path,
) -> tuple[Path, Path, OmegaConf]:
    """
    Determine config directory, state directory, and submit info for preconfigured runs.

    Args:
        run_dir: Run directory passed via --run-dir

    Returns:
        (config_dir, state_dir, submit_info): Config source, state location, and submit metadata
    """
    submit_info = OmegaConf.load(run_dir / "submit.yaml")

    # Check if this is a resume directory
    if "parent_run_dir" in submit_info:
        # Resume: config from parent, state in resume dir
        parent_dir = resolve_relative_path(submit_info.parent_run_dir)
        return parent_dir, run_dir, submit_info

    # Normal run: config and state both in run_dir
    return run_dir, run_dir, submit_info


def _preconfigured_setup_rank0(
    conf: OmegaConf,
    config_dir: Path,
    state_dir: Path,
    command: str,
    submit_info: OmegaConf,
) -> None:
    """
    Handle state collection and logging for rank 0 in preconfigured runs.

    Args:
        conf: Configuration object
        config_dir: Directory containing config.yaml (parent for resumes)
        state_dir: Directory for state files (resume-* for resumes)
        command: Command being executed
        submit_info: Submit metadata
    """
    is_resume = "parent_run_dir" in submit_info or "resume_checkpoint" in submit_info

    timestamp = datetime.datetime.now()

    # Reconstruct args for state collection
    args = OmegaConf.create()
    args.command = command
    args.config = str(config_dir / "config.yaml")
    args.overrides = submit_info.get("overrides", [])
    args.checkpoint = submit_info.get("checkpoint")
    args.resume = is_resume

    # Collect and save state
    _save_state(conf, args, timestamp, state_dir=state_dir)

    # Set up logging
    action = "resuming" if is_resume else "starting"
    _setup_logging(config_dir, action=action, submit_info=submit_info)


def _preconfigured_wait_rank_sync(state_dir: Path) -> None:
    """
    Wait for rank 0 to create state.yaml in preconfigured runs (non-rank-0 processes).

    Args:
        state_dir: Directory containing state.yaml
    """
    state_path = state_dir / "state.yaml"
    timeout = datetime.timedelta(minutes=10)
    start = datetime.datetime.now()

    while not state_path.exists():
        if datetime.datetime.now() - start > timeout:
            raise TimeoutError(f"Rank 0 did not create state.yaml in {state_dir}")
        time.sleep(1)


def init_preconfigured(
    run_dir: Path,
    command: str,
) -> tuple[OmegaConf, OmegaConf]:
    """
    Load and set up pre-configured run from directory.

    Handles all modes:
    - New training
    - Pre-trained weights
    - Resume training
    - Eval/predict

    Returns:
        (conf, submit_info): Configuration and submit metadata
    """
    run_dir = Path(run_dir)

    # 1. Determine config source and state location
    config_dir, state_dir, submit_info = _preconfigured_determine_dirs(run_dir)

    # If this is a slurm requeue, we need to set up a new state_dir
    if slurm.is_requeue():
        # If this is a requeue of the original run dir, create a new resume-*
        # directory for it.
        if config_dir == state_dir:
            state_dir = config_dir / f"resume-{config_dir.name}"

        state_dir = state_dir.with_name(
            f"{state_dir.name}-rq{slurm.get_restart_count()}"
        )

        if mp.rank == 0:
            state_dir.mkdir(parents=True, exist_ok=False)

    # 2. Load static config (from parent for resumes)
    conf = _load_from_file(config_dir / "config.yaml")

    # 3. Restore args and slurm config from submit.yaml
    is_resume = "parent_run_dir" in submit_info
    is_resume = is_resume or slurm.is_requeue()
    conf = unlock(conf)

    # Reconstruct args
    args = OmegaConf.create()
    args.command = command
    args.config = str(config_dir / "config.yaml")
    args.overrides = submit_info.get("overrides", [])
    args.checkpoint = submit_info.get("checkpoint")
    args.resume = is_resume
    conf.args = args

    # Restore slurm config if present (only written for the SLURM backend;
    # otherwise the default from the loaded config is kept, unused at runtime)
    if "slurm" in submit_info:
        conf.slurm = OmegaConf.create(submit_info.slurm)

    conf = lock(conf)

    # 4. Verify command matches
    if submit_info.command != command:
        raise ValueError(
            f"Run prepared for '{submit_info.command}' but executing '{command}'"
        )

    # 5. Handle state collection and logging
    if mp.rank == 0:
        _preconfigured_setup_rank0(conf, config_dir, state_dir, command, submit_info)
    else:
        _preconfigured_wait_rank_sync(state_dir)

    # 6. Check for state changes (only for resume cases)
    if is_resume and mp.rank == 0:
        _check_state_changes(config_dir, state_dir)

    # 7. Merge state metadata into config
    conf = _merge_state_metadata(conf, state_dir=state_dir)

    # 8. Initialize PyTorch settings
    _initialize(conf)

    set_global_config(conf)

    if "parent_run_dir" in submit_info:
        run_dir = resolve_relative_path(submit_info.parent_run_dir)

    return conf, submit_info, run_dir


def generate_slurm_script(
    run_dir: Path,
    command: str,
    slurm_config: dict,
) -> Path:
    """
    Generate SLURM job script from template.

    Args:
        run_dir: Run directory containing prepared config
        command: Command to run (train/eval/predict/preprocess)
        slurm_config: SLURM configuration dict

    Returns:
        script_path: Path to generated script
    """
    # pylint: disable=too-many-branches
    run_dir = Path(run_dir)

    # Get job name from config or slurm_config
    job_name = slurm_config.job_name

    # Load template from config directory
    with open(slurm_submit_template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Build optional SBATCH arguments
    optional_args = []
    if slurm_config.get("ntasks_per_node"):
        optional_args.append(
            f"#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}"
        )
    if slurm_config.get("gres"):
        optional_args.append(f"#SBATCH --gres={slurm_config['gres']}")
    if slurm_config.get("cpus_per_gpu"):
        optional_args.append(f"#SBATCH --cpus-per-gpu={slurm_config['cpus_per_gpu']}")
    if slurm_config.get("cpus_per_task"):
        optional_args.append(f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}")
    if slurm_config.get("time"):
        optional_args.append(f"#SBATCH --time={slurm_config['time']}")
    if slurm_config.get("account"):
        optional_args.append(f"#SBATCH --account={slurm_config['account']}")
    if slurm_config.get("qos"):
        optional_args.append(f"#SBATCH --qos={slurm_config['qos']}")
    if slurm_config.get("constraint"):
        optional_args.append(f"#SBATCH --constraint={slurm_config['constraint']}")
    if slurm_config.get("mem"):
        optional_args.append(f"#SBATCH --mem={slurm_config['mem']}")
    if slurm_config.get("exclude"):
        optional_args.append(f"#SBATCH --exclude={slurm_config['exclude']}")
    if slurm_config.get("dependency"):
        optional_args.append(f"#SBATCH --dependency={slurm_config['dependency']}")
    if slurm_config.get("sbatch_args"):
        for arg in slurm_config["sbatch_args"]:
            optional_args.append(f"#SBATCH {arg}")

    optional_args = "\n".join(optional_args)

    # Fill template
    script_content = template.format(
        job_name=job_name,
        partition=slurm_config["partition"],
        nodes=slurm_config["nodes"],
        run_dir=_make_path_relative(run_dir),
        command=command,
        main_script="./main.py",
        optional_sbatch_args=optional_args,
        config_root=_make_path_relative(paths.config),
    )

    # Save script
    script_path = run_dir / "slurm_submit.sh"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    log.info("Generated SLURM script: %s", script_path)

    return script_path


def submit_slurm_job(run_dir: Path | str) -> str:
    """
    Submit SLURM job via sbatch using the pre-generated script.

    Args:
        run_dir: Run directory containing slurm_submit.sh

    Returns:
        job_id: SLURM job ID
    """
    script_path = Path(run_dir) / "slurm_submit.sh"

    if not script_path.exists():
        raise FileNotFoundError(
            f"SLURM script not found at {script_path}. "
            "Did you forget to call generate_slurm_script()?"
        )

    # Submit job
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    # Check if submission was successful
    # Note: sbatch may return exit code 0 even on errors, so check stderr for errors
    has_error = result.returncode != 0 or "ERROR" in result.stderr

    if has_error:
        log.error("SLURM job submission failed with return code %d", result.returncode)
        log.error("stdout: %s", result.stdout)
        log.error("stderr: %s", result.stderr)
        raise RuntimeError(
            f"Failed to submit SLURM job. Return code: {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    # Parse job ID from output: "Submitted batch job 123456"
    output = result.stdout.strip()
    job_id = output.split()[-1]

    return job_id
