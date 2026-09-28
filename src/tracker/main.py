# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

import click
import lightning as L
from omegaconf import OmegaConf

from . import config, data, mod, utils
from .config import DEFAULT_CONFIG, paths

log = utils.log.get_logger(__name__)


def _build_model(
    conf: OmegaConf,
    checkpoint: str | None = None,
    strict: bool = True,
) -> L.LightningModule:
    if checkpoint:
        log.info(
            "loading model from checkpoint, type %s, path: %s",
            conf.model.type,
            checkpoint,
        )
        model = mod.module.load(
            checkpoint,
            model=conf.model,
            optimizer=conf.optimizer,
            lr_scheduler=conf.get("lr_scheduler"),
            # use map_location="cpu" to a) avoid issues when loading on a
            # machine with different GPUs and b) to avoid unnecessary GPU
            # memory usage
            map_location="cpu",
            strict=strict,
        )

    else:
        log.info("instantiating new model, type: %s", conf.model.type)
        model = mod.module.build(
            model=conf.model,
            optimizer=conf.optimizer,
            lr_scheduler=conf.get("lr_scheduler"),
        )

    model = utils.torch.compile(
        model, OmegaConf.select(conf, "torch.compile", default=False)
    )

    return model


def _build_components(
    conf: OmegaConf,
    command: str,
    checkpoint: str | None = None,
    strict: bool | None = None,
    resume: bool = False,
) -> tuple[L.Trainer, data.module.DataModule, L.LightningModule]:
    """
    Build trainer, datamodule, and model from a loaded config.

    For train: checkpoint/strict from config (init.checkpoint, init.strict)
    For eval/predict: checkpoint/strict from function parameters (CLI args)
    """
    callbacks = []

    # build data module
    datamod = data.module.build(conf.data)

    # For training, use init.checkpoint from config
    # For eval/predict, use passed checkpoint parameter
    if command == "train" and not resume:
        if checkpoint is None:
            checkpoint = OmegaConf.select(conf, "init.checkpoint")

            # Resolve relative paths for train
            if checkpoint is not None:
                checkpoint = Path(checkpoint).expanduser()
                if not Path(checkpoint).is_absolute():
                    checkpoint = paths.root / checkpoint

        if strict is None:
            strict = OmegaConf.select(conf, "init.strict", default=True)

    # build the actual model
    model = _build_model(conf, checkpoint, strict)

    # if we are evaluating / training with validation epochs: build validation strategy
    if command in ["train", "eval"] and "validation" in conf:
        callbacks += data.validation.build_callbacks(conf.validation)

    # if we are predicting: build collector/writer for predictions
    if command == "predict" and "prediction" in conf:
        callbacks += data.prediction.build_callbacks(conf.prediction)

    # build trainer
    trainer = config.types.trainer.build(conf, callbacks=callbacks)

    return trainer, datamod, model


# Scheduler backends available to `submit`. Add new entries (e.g. "pbs",
# "lsf") here and handle them in _dispatch_run().
SCHEDULERS = ("slurm", "none")

_scheduler_option = click.option(
    "--scheduler",
    type=click.Choice(SCHEDULERS),
    default="slurm",
    show_default=True,
    help="""Scheduler backend used to launch the prepared run. "slurm"
    generates a batch script from submit.yaml and submits it via sbatch.
    "none" only prepares the run directory and prints the command to execute
    it directly (via --run-dir) -- use this to run interactively or under a
    scheduler this framework does not natively support.""",
)


def _dispatch_run(run_dir: Path, command: str, scheduler: str, dry_run: bool) -> None:
    """Launch a prepared run directory via the selected scheduler backend.

    By this point the run directory is fully prepared (config.yaml +
    submit.yaml); this step only decides *how* it is launched. The
    scheduler-agnostic path is "none", which prepares nothing further and just
    reports the command to run.
    """
    if scheduler == "none":
        log.info("Prepared run directory: %s", run_dir)
        log.info(
            "Scheduler 'none': run it directly with:\n    ./main.py %s --run-dir %s",
            command,
            run_dir,
        )
        return

    if scheduler == "slurm":
        # Load SLURM config from submit.yaml
        submit_info = OmegaConf.load(run_dir / "submit.yaml")
        slurm_config = submit_info.slurm

        # Generate SLURM script
        config.generate_slurm_script(run_dir, command, slurm_config)

        # Submit job (unless dry-run)
        if not dry_run:
            job_id = config.submit_slurm_job(run_dir)
            log.info("Submitted SLURM job: %s", job_id)
        else:
            log.info("Dry run - job not submitted")
        return

    raise ValueError(f"unknown scheduler backend: '{scheduler}'")


@click.group()
def cli() -> None:
    """
    Main command line entry-point of this framework.

    Train, evaluate, produce predictions, or preprocess.
    """


@cli.command(name="train")
@click.option(
    "--run-dir",
    type=Path,
    hidden=True,
    help="""Pre-configured run directory (set by submit). When set,
    all configuration is loaded from the directory and other options are ignored.""",
)
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="""The configuration specifying the model, optimizer, data, and other
    training parameters. If relative to the config root directory, the config
    will be loaded as a composed hydra config. If not, the config will be
    expected to be a full yaml config and loaded via OmegaConf.""",
)
@click.option(
    "--resume",
    required=False,
    help="""Resume training in the specified run directory. The configuration
    will be loaded from $RUN_DIR/config.yaml. By default, the checkpoint will
    be loaded from $RUN_DIR/last.ckpt. The specific checkpoint name to be
    continued can be overridden by specifying it after a colon, e.g., --resume
    /path/to/run:on_exception will load /path/to/run/on_exception.ckpt. Checkpoints
    outside the run directory are not supported. Overrides are not supported.
    If more fine-grained control is required, please use --resume-checkpoint in
    combination with --config to initiate a new run with the specified checkpoint.
    """,
)
@click.option(
    "--resume-checkpoint",
    required=False,
    help="""Resume training from the specified checkpoint.
    """,
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def fit(
    run_dir: Path | None = None,
    conf: Path | str | None = None,
    overrides: list[str] = (),
    resume: Path | str | None = None,
    resume_checkpoint: Path | str | None = None,
) -> None:
    """
    Train a model using the specified parameters.

    Refer to the hydra-config and OmegaConf documentations for the syntax of
    config overrides (OVERRIDES).
    """
    # pylint: disable=too-many-locals

    # Initialize logging
    utils.log.initialize()

    # Step 1: Determine configuration source and setup parameters
    if run_dir is not None:
        # Load pre-configured run directory
        assert conf is None, "Cannot specify --config with --run-dir"
        assert overrides == (), "Cannot specify overrides with --run-dir"
        assert resume is None, "Cannot specify --resume with --run-dir"
        assert (
            resume_checkpoint is None
        ), "Cannot specify --resume-checkpoint with --run-dir"

        # Load pre-configured run (config already loaded and initialized)
        conf, submit_info, run_dir = config.init_preconfigured(run_dir, command="train")

        # Determine checkpoint to resume from
        if utils.slurm.is_requeue():
            # NOTE: Normally, we'd want to resume from the HPC checkpoint
            # dumped by lightning on preemption. However, in practice this is a
            # bit iffy. Lightning's checkpointing on preemption is not super
            # robust and often fails to produce a checkpoint. Making matters
            # worse, checkpoints are not cleaned up properly and often remain
            # in the run directory. This causes stale HPC checkpoints to be
            # picked up even though the run has long since been resumed and
            # produced a new checkpoint, and we should actually be resuming
            # from last.ckpt instead.
            resume_checkpoint = config.find_checkpoint(run_dir, prefer_hpc=False)
        elif submit_info.get("resume_checkpoint"):
            # Resolve path relative to project root
            resume_checkpoint = config.resolve_relative_path(
                submit_info.resume_checkpoint
            )

    else:
        if resume is not None:
            assert conf is None, "Cannot specify --config with --resume."
            assert len(overrides) == 0, "Overrides are not supported with --resume."
            assert (
                resume_checkpoint is None
            ), "Cannot specify --resume-checkpoint with --resume."

            # Parse and find checkpoint
            _, conf, resume_checkpoint = config.resolve_resume_config(resume)

        if conf is None:
            conf = DEFAULT_CONFIG

        # Load config and setup tracking
        conf = config.init(
            conf=conf,
            overrides=overrides,
            command="train",
            resume=resume is not None,
        )

    # Step 2: Build components
    trainer, datamod, model = _build_components(
        conf, command="train", resume=resume_checkpoint is not None
    )

    # Step 4: Train
    trainer.fit(model, datamodule=datamod, ckpt_path=resume_checkpoint)


@cli.command(name="eval")
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="""The configuration specifying the model, validation data, and
    validation protocol (e.g., metrics). If relative to the config root
    directory, the config will be loaded as a composed hydra config. If not,
    the config will be expected to be a full yaml config and loaded via
    OmegaConf.""",
)
@click.option(
    "-p",
    "--checkpoint",
    required=False,
    help="The model checkpoint to evaluate.",
)
@click.option(
    "-d",
    "--directory",
    required=False,
    help="""The run directory containing config and checkpoint to use for
    evaluation. Convenience option to specify both config and checkpoint in one
    go. The config will be loaded from $RUN_DIR/config.yaml and the checkpoint
    from $RUN_DIR/last.ckpt. Both may be overridden by specifying --config and
    --checkpoint explicitly.""",
)
@click.option(
    "--strict/--non-strict",
    default=True,
    help="""Whether the specified checkpoint is loaded in strict mode (default)
    or not (--non-strict).""",
)
@click.option(
    "--run-dir",
    hidden=True,
    help="Pre-configured run directory (set by SLURM job submission)",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def validate(
    conf: Path | str | None = None,
    overrides: list[str] = (),
    checkpoint: Path | str | None = None,
    directory: Path | str | None = None,
    strict: bool = True,
    run_dir: Path | str | None = None,
) -> None:
    """
    Evaluate a trained model using the specified parameters.

    Refer to the hydra-config and OmegaConf documentations for the syntax of
    config overrides (OVERRIDES).
    """

    # Initialize logging
    utils.log.initialize()

    # Step 1: Determine configuration source and setup parameters
    if run_dir is not None:
        assert conf is None, "Cannot specify --config with --run-dir"
        assert overrides == (), "Cannot specify overrides with --run-dir"
        assert checkpoint is None, "Cannot specify --checkpoint with --run-dir"
        assert directory is None, "Cannot specify --directory with --run-dir"
        assert strict, "Cannot specify --strict with --run-dir"

        # Load pre-configured run
        conf, submit_info, _ = config.init_preconfigured(run_dir, command="eval")

        # Determine checkpoint based on submit_info
        checkpoint = config.resolve_relative_path(submit_info.checkpoint)
        strict = submit_info.strict

        if checkpoint is None and submit_info.directory is not None:
            directory = config.resolve_relative_path(submit_info.directory)
            checkpoint = directory / "last.ckpt"

        if checkpoint is None:
            raise RuntimeError("No checkpoint specified in submit info")

    else:
        # Sanity-check the options.
        assert (
            directory is not None or checkpoint is not None
        ), "Either --directory or --checkpoint must be specified."

        # If a directory is specified, we load the config from the run directory and
        # the checkpoint from the last.ckpt file, unless overridden.
        if directory is not None:
            if not Path(directory).exists():
                raise FileNotFoundError(f"Run directory '{directory}' not found.")

            # set checkpoint via directory if not specified
            if checkpoint is None:
                checkpoint = Path(directory) / "last.ckpt"

            # set config via directory if not specified
            if conf is None:
                conf = Path(directory) / "config.yaml"

        if conf is None:
            conf = DEFAULT_CONFIG

        # Load config and setup tracking
        conf = config.init(
            conf=conf,
            overrides=overrides,
            command="eval",
            checkpoint=checkpoint,
        )

    # Step 2: Build components
    trainer, datamod, model = _build_components(
        conf, command="eval", checkpoint=checkpoint, strict=strict
    )

    # Step 3: Run evaluation
    trainer.validate(model, datamodule=datamod)


@cli.command(name="predict")
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="""The configuration specifying the model, prediction data, and
    prediction protocol (e.g., output format). If relative to the config root
    directory, the config will be loaded as a composed hydra config. If not,
    the config will be expected to be a full yaml config and loaded via
    OmegaConf.""",
)
@click.option(
    "-p",
    "--checkpoint",
    required=False,
    help="The model checkpoint to use for generating predictions.",
)
@click.option(
    "-d",
    "--directory",
    required=False,
    help="""The run directory containing config and checkpoint to use for
    generating predictions. Convenience option to specify both config and
    checkpoint in one go. The config will be loaded from $RUN_DIR/last.ckpt.
    Both may be overridden by specifying --config and --checkpoint
    explicitly.""",
)
@click.option(
    "--strict/--non-strict",
    default=True,
    help="""Whether the specified checkpoint is loaded in strict mode (default)
    or not (--non-strict).""",
)
@click.option(
    "--run-dir",
    hidden=True,
    help="Pre-configured run directory (set by SLURM job submission)",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def predict(
    conf: Path | str | None = None,
    overrides: list[str] = (),
    checkpoint: Path | str | None = None,
    directory: Path | str | None = None,
    strict: bool = True,
    run_dir: Path | str | None = None,
) -> None:
    """
    Compute predictions from a trained model checkpoint.

    Predictions will be generated and stored according to the provided
    configuration.

    Refer to the hydra-config and OmegaConf documentations for the syntax of
    config overrides (OVERRIDES).
    """

    # Initialize logging
    utils.log.initialize()

    # Step 1: Determine configuration source and setup parameters
    if run_dir is not None:
        # Load pre-configured run directory
        assert conf is None, "Cannot specify --config with --run-dir"
        assert overrides == (), "Cannot specify overrides with --run-dir"
        assert checkpoint is None, "Cannot specify --checkpoint with --run-dir"
        assert directory is None, "Cannot specify --directory with --run-dir"
        assert strict, "Cannot specify --strict with --run-dir"

        conf, submit_info, _ = config.init_preconfigured(run_dir, command="predict")

        # Determine checkpoint based on submit_info
        checkpoint = config.resolve_relative_path(submit_info.checkpoint)
        strict = submit_info.strict

        if checkpoint is None and submit_info.directory is not None:
            directory = config.resolve_relative_path(submit_info.directory)
            checkpoint = directory / "last.ckpt"

        if checkpoint is None:
            raise RuntimeError("No checkpoint specified in submit info")

    else:
        # Sanity-check the options.
        assert (
            directory is not None or checkpoint is not None
        ), "Either --directory or --checkpoint must be specified."

        # If a directory is specified, we load the config from the run directory and
        # the checkpoint from the last.ckpt file, unless overridden.
        if directory is not None:
            if not Path(directory).exists():
                raise FileNotFoundError(f"Run directory '{directory}' not found.")

            # set checkpoint via directory if not specified
            if checkpoint is None:
                checkpoint = Path(directory) / "last.ckpt"

            # set config via directory if not specified
            if conf is None:
                conf = Path(directory) / "config.yaml"

        if conf is None:
            conf = DEFAULT_CONFIG

        # Load config and setup tracking
        conf = config.init(
            conf=conf,
            overrides=overrides,
            command="predict",
            checkpoint=checkpoint,
        )

    # Step 2: Build components (unified)
    trainer, datamod, model = _build_components(
        conf, command="predict", checkpoint=checkpoint, strict=strict
    )

    # Step 3: Run prediction
    trainer.predict(model, datamodule=datamod, return_predictions=False)


@cli.command(name="preprocess")
@click.option(
    "-c",
    "--config",
    "conf",
    help="""The configuration specifying the model, data, and protocols (e.g.,
    validation / prediction). If relative to the config root directory, the
    config will be loaded as a composed hydra config. If not, the config will
    be expected to be a full yaml config and loaded via OmegaConf.""",
)
@click.option(
    "--run-dir",
    hidden=True,
    help="Pre-configured run directory (set by SLURM job submission)",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def preprocess(
    conf: Path | str | None = None,
    overrides: list[str] = (),
    run_dir: Path | str | None = None,
) -> None:
    """
    Run any preprocessing required by the specified config.

    Note: In this framework, we do not use a "real" preprocessing stage.
    Instead, all "preprocessing" is done inside the actual modules that require
    it. This means, that you can just run the respective stages
    (train/eval/predict) without having to do any preprocessing.

    However, since some stages do, for example, require building large indexes
    over the dataset, this will be slow for the first execution. To speed
    things up during subsequent executions, modules may cache data to disk.

    This command is intended to provide a way to run any preprocessing/caching
    steps before other stages run. To this end, it will load all required
    dataset modules and exit.

    Refer to the hydra-config and OmegaConf documentations for the syntax of
    config overrides (OVERRIDES).
    """

    # Initialize logging
    utils.log.initialize()

    # Step 1: Determine configuration source and load config
    if run_dir is not None:
        # Load pre-configured run
        assert len(overrides) == 0, "Cannot specify overrides with --run-dir"
        assert conf is None, "Cannot specify --config with --run-dir"

        conf, _submit_info, _ = config.init_preconfigured(run_dir, command="preprocess")

    else:
        if conf is None:
            conf = DEFAULT_CONFIG

        # Load and initialize new config
        conf = config.init(conf=conf, overrides=overrides, command="preprocess")

    # Step 2: Run preprocessing
    log.info("preprocess: building data module")
    datamod = data.module.build(conf.data)

    log.info("preprocess: running stage: fit/validate")
    datamod.setup("fit")  # this also covers 'validate'

    log.info("preprocess: running stage: test")
    datamod.setup("test")

    log.info("preprocess: running stage: predict")
    datamod.setup("predict")

    log.info("preprocess: tearing down...")
    datamod.teardown("predict")
    datamod.teardown("test")
    datamod.teardown("fit")


@cli.group(name="submit")
def submit() -> None:
    """
    Prepare run directories and launch jobs via a scheduler backend.

    This command group provides subcommands to prepare a run directory
    (freezing config and metadata) and launch it. The scheduler backend is
    selected via --scheduler (default: slurm; use --scheduler none to only
    prepare the run directory and run it yourself via --run-dir).
    """


@submit.command(name="train")
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="Config file for new training (required unless using --resume)",
)
@click.option(
    "--resume",
    required=False,
    help="Resume training from run directory (path[:checkpoint])",
)
@click.option(
    "--resume-checkpoint",
    required=False,
    help="Resume from specific checkpoint",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=False,
    help="Prepare but don't submit",
)
@_scheduler_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def submit_fit(
    conf: Path | str | None,
    resume: Path | str | None,
    resume_checkpoint: Path | str | None,
    dry_run: bool,
    scheduler: str,
    overrides: list[str],
) -> None:
    """
    Prepare and submit a training job (default backend: SLURM).

    Supports all training modes: new training, pre-trained weights, resume.
    The configuration is frozen at submit time and stored in the run directory.

    Use config overrides to set checkpoint and SLURM parameters, e.g.:
    init.checkpoint=/path/to/model.ckpt slurm.partition=gpu slurm.nodes=2

    For resume, overrides can be used to change SLURM settings:
    --resume runs/my_run slurm.nodes=4
    """
    utils.log.initialize()

    # Validation
    if resume is not None:
        assert conf is None, "Cannot specify --config with --resume"
        assert (
            resume_checkpoint is None
        ), "Cannot specify --resume-checkpoint with --resume"
    else:
        if conf is None:
            conf = DEFAULT_CONFIG

    # Prepare run (handles both new training and resume)
    # This loads the config and extracts SLURM settings from it
    run_dir = config.prepare_submit_fit(
        conf=conf,
        overrides=overrides,
        resume=resume,
        resume_checkpoint=resume_checkpoint,
        scheduler=scheduler,
    )

    # Generate and submit job
    _dispatch_run(run_dir, "train", scheduler, dry_run)


@submit.command(name="eval")
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="Config file (overrides run directory config)",
)
@click.option(
    "-p",
    "--checkpoint",
    required=False,
    help="Checkpoint to evaluate",
)
@click.option(
    "-d",
    "--directory",
    required=False,
    type=Path,
    help="Run directory containing trained model",
)
@click.option(
    "--strict/--non-strict",
    default=True,
    help="Strict checkpoint loading (default: strict)",
)
@click.option("--dry-run/--no-dry-run", default=False, help="Prepare but don't submit")
@_scheduler_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def submit_validate(
    conf: Path | str | None,
    checkpoint: Path | str | None,
    directory: Path | None,
    strict: bool,
    dry_run: bool,
    scheduler: str,
    overrides: list[str],
) -> None:
    """
    Prepare and submit an evaluation job (default backend: SLURM).

    Can evaluate from a run directory or standalone config + checkpoint.
    Use config overrides to set SLURM parameters.
    """
    utils.log.initialize()

    # Validation
    assert (
        directory is not None or checkpoint is not None
    ), "Either --directory or --checkpoint must be specified"

    # Prepare eval run
    run_dir = config.prepare_submit_eval(
        conf=conf,
        checkpoint=checkpoint,
        directory=directory,
        strict=strict,
        overrides=overrides,
        scheduler=scheduler,
    )

    # Generate and submit job
    _dispatch_run(run_dir, "eval", scheduler, dry_run)


@submit.command(name="predict")
@click.option(
    "-c",
    "--config",
    "conf",
    required=False,
    help="Config file (overrides run directory config)",
)
@click.option(
    "-p",
    "--checkpoint",
    required=False,
    help="Checkpoint to use for predictions",
)
@click.option(
    "-d",
    "--directory",
    required=False,
    type=Path,
    help="Run directory containing trained model",
)
@click.option(
    "--strict/--non-strict",
    default=True,
    help="Strict checkpoint loading (default: strict)",
)
@click.option("--dry-run/--no-dry-run", default=False, help="Prepare but don't submit")
@_scheduler_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def submit_predict(
    conf: Path | str | None,
    checkpoint: Path | str | None,
    directory: Path | None,
    strict: bool,
    dry_run: bool,
    scheduler: str,
    overrides: list[str],
) -> None:
    """
    Prepare and submit a prediction job (default backend: SLURM).

    Can predict from a run directory or standalone config + checkpoint.
    Use config overrides to set SLURM parameters.
    """
    utils.log.initialize()

    # Validation
    assert (
        directory is not None or checkpoint is not None
    ), "Either --directory or --checkpoint must be specified"

    # Prepare predict run
    run_dir = config.prepare_submit_predict(
        conf=conf,
        checkpoint=checkpoint,
        directory=directory,
        strict=strict,
        overrides=overrides,
        scheduler=scheduler,
    )

    # Generate and submit job
    _dispatch_run(run_dir, "predict", scheduler, dry_run)


@submit.command(name="preprocess")
@click.option(
    "-c",
    "--config",
    "conf",
    default=DEFAULT_CONFIG,
    help="Config file for preprocessing",
)
@click.option("--dry-run/--no-dry-run", default=False, help="Prepare but don't submit")
@_scheduler_option
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def submit_preprocess(
    conf: Path | str,
    dry_run: bool,
    scheduler: str,
    overrides: list[str],
) -> None:
    """
    Prepare and submit a preprocessing job (default backend: SLURM).

    Use config overrides to set SLURM parameters.
    """
    utils.log.initialize()

    # Prepare preprocess run
    run_dir = config.prepare_submit_preprocess(
        conf=conf,
        overrides=overrides,
        scheduler=scheduler,
    )

    # Generate and submit job
    _dispatch_run(run_dir, "preprocess", scheduler, dry_run)
