# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from lightning.fabric.utilities.logger import _convert_params, _flatten_dict
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning_utilities.core.imports import RequirementCache
from omegaconf import OmegaConf

from ... import config, utils
from ..registry import Registry

if TYPE_CHECKING:
    from aim.pytorch_lightning import AimLogger

_AIM_AVAILABLE = RequirementCache("aim>=3.0.0")
_MLFLOW_AVAILABLE = RequirementCache("mlflow>=1.0.0")

log = utils.log.get_logger(__name__)


def _build_generic_logger(
    cls,
    paths: OmegaConf,
    meta: OmegaConf,
    save_dir: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    **kwargs,
) -> Logger:
    if save_dir is None:
        save_dir = paths.runs

    if name is None:
        name = meta.run.name

    if version is None:
        version = meta.run.version

    return cls(save_dir=save_dir, name=name, version=version, **kwargs)


def _build_csv_logger(**kwargs) -> CSVLogger:
    return _build_generic_logger(CSVLogger, **kwargs)


def _build_tb_logger(**kwargs) -> TensorBoardLogger:
    return _build_generic_logger(TensorBoardLogger, **kwargs)


def _build_wandb_logger(
    paths: OmegaConf,
    meta: OmegaConf,
    project: str | None = None,
    run_name: str | None = None,
    version: str | None = None,
    save_dir: str | Path | None = None,
    **kwargs,
) -> WandbLogger:
    tags = meta.run.get("tags", [])
    notes = meta.run.get("description")

    if project is None:
        project = os.environ.get("WANDB_PROJECT", "tracker")

    if save_dir is None:
        save_dir = paths.root

    if run_name is None:
        run_name = meta.run.name

    if version is None:
        version = meta.run.version

    run_id = f"{run_name}-{version}"

    return WandbLogger(
        save_dir=paths.runs,
        project=project,
        name=run_name,
        id=run_id,
        tags=tags,
        notes=notes,
        **kwargs,
    )


def _build_aim_logger(
    paths: OmegaConf,
    meta: OmegaConf,
    save_dir: str | Path | None = None,
    repo: str | None = None,
    experiment: str | None = None,
    run_name: str | None = None,
    **kwargs,
) -> "AimLogger":
    if not _AIM_AVAILABLE:
        raise ModuleNotFoundError(str(_AIM_AVAILABLE))

    # pylint: disable-next=import-outside-toplevel
    from aim.pytorch_lightning import AimLogger

    # set defaults
    if save_dir is None:
        save_dir = paths.current_run

    if repo is None:
        repo = paths.root

    if run_name is None:
        run_name = meta.run.name

    if experiment is None:
        experiment = OmegaConf.select(meta, "experiment.name", default=None)

    # if we are resuming an experiment, try to continue the aim run
    if config.get().args.resume or utils.slurm.is_requeue():
        aim_data = Path(save_dir) / "aim.yaml"

        if aim_data.exists():
            aim_data = OmegaConf.load(aim_data)

            log.info("resuming Aim run with hash '%s'", aim_data.hash)

            logger = AimLogger(
                repo=aim_data.repo,
                experiment=aim_data.experiment,
                run_name=aim_data.run,
                run_hash=aim_data.hash,
                **kwargs,
            )

            return logger

    # build base logger
    logger = AimLogger(repo=repo, experiment=experiment, run_name=run_name, **kwargs)

    # The tag setup needs to be done on rank 0 due to coinciding issues: First
    # off, we need to drop any duplicate tags because attempting to add those
    # will raise an error. However, we can only obtain a valid set of tags on
    # rank 0. While non-zero ranks return a dummy logger object, attempting to
    # call set(logger.experiment.tags) on that results in a type error.
    if utils.mp.rank == 0:
        # add tags
        tags = meta.run.get("tags", [])
        tags = set(tags) - set(logger.experiment.tags)
        for tag in tags:
            logger.experiment.add_tag(tag)

        # add description
        desc = meta.run.get("description")
        if desc is not None:
            logger.experiment.description = str(desc)

        # store connection info
        aim_data = {
            "repo": repo,
            "experiment": experiment,
            "run": run_name,
            "hash": logger.experiment.hash,
        }

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(aim_data, save_dir / "aim.yaml")

    return logger


if _MLFLOW_AVAILABLE:
    # pylint: disable-next=ungrouped-imports
    from lightning.pytorch.loggers import MLFlowLogger as _MLFlowLogger
    from mlflow.entities import Param

    class MLFlowLogger(_MLFlowLogger):
        """
        Custom MLFlowLogger that handles parameter logging for resumed runs.

        The default MLFlowLogger raises an error when attempting to log parameters
        that already exist in a resumed run. This override checks existing parameters
        and skips them if they have the same value, or warns if they differ.
        """

        def log_hyperparams(
            self, params: Union[dict[str, Any], argparse.Namespace]
        ) -> None:
            if not rank_zero_only.rank == 0:
                return

            params = _convert_params(params)
            params = _flatten_dict(params)

            # Get existing parameters for this run
            run = self.experiment.get_run(self.run_id)
            existing_params = run.data.params if run else {}

            # Filter out params that already exist with the same value
            params_to_log = []
            for key, value in params.items():
                value_str = str(value)[:250]  # Truncate to 250 characters

                if key in existing_params:
                    # If same value, skip logging
                    if existing_params[key] == value_str:
                        continue

                    # If different value, warn but don't log to mlflow to avoid error
                    log.warning(
                        "Parameter '%s' already exists with different value. "
                        "Existing: '%s', New: '%s'. Skipping update.",
                        key,
                        existing_params[key],
                        value_str,
                    )
                    continue

                params_to_log.append(Param(key=key, value=value_str))

            # Log in chunks of 100 parameters (the maximum allowed by MLflow)
            for idx in range(0, len(params_to_log), 100):
                self.experiment.log_batch(
                    run_id=self.run_id,
                    params=params_to_log[idx : idx + 100],
                    **self._log_batch_kwargs,
                )


def _build_mlflow_logger(
    paths: OmegaConf,
    meta: OmegaConf,
    save_dir: str | Path | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    **kwargs,
) -> "MLFlowLogger":
    # pylint: disable=too-many-branches,too-many-locals
    if not _MLFLOW_AVAILABLE:
        raise ModuleNotFoundError(str(_MLFLOW_AVAILABLE))

    # pylint: disable-next=import-outside-toplevel
    from mlflow.utils.mlflow_tags import MLFLOW_RUN_NOTE

    # set defaults
    if save_dir is None:
        save_dir = paths.current_run

    if tracking_uri is None:
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", str(Path(paths.root) / "mlruns")
        )

    if experiment_name is None:
        experiment_name = OmegaConf.select(meta, "experiment.name", default=None)

        if experiment_name is None:
            experiment_name = "default"

    if run_name is None:
        run_name = meta.run.name

    # if we are resuming an experiment, try to continue the mlflow run
    if config.get().args.resume or utils.slurm.is_requeue():
        mlflow_data = Path(save_dir) / "mlflow.yaml"

        if mlflow_data.exists():
            mlflow_data = OmegaConf.load(mlflow_data)

            log.info("resuming MLFlow run with id '%s'", mlflow_data.run_id)

            # pylint: disable=possibly-used-before-assignment
            logger = MLFlowLogger(
                experiment_name=mlflow_data.experiment_name,
                run_name=mlflow_data.run_name,
                tracking_uri=mlflow_data.tracking_uri,
                run_id=mlflow_data.run_id,
                **kwargs,
            )

            return logger

    # Process tags
    tags_dict = {}
    tags = meta.run.get("tags", [])
    for tag in tags:
        # Split on first '=' or ':' to create key-value pairs
        if "=" in tag:
            key, _, value = tag.partition("=")
        elif ":" in tag:
            key, _, value = tag.partition(":")
        else:
            # No separator, use tag as key with True as value
            key, value = tag, True

        tags_dict[key] = value

    # Add description to tags using MLflow's standard tag key
    desc = meta.run.get("description")
    if desc is not None:
        tags_dict[MLFLOW_RUN_NOTE] = str(desc)

    # build base logger
    logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=tracking_uri,
        tags=tags_dict,
        **kwargs,
    )

    # store connection info
    if utils.mp.rank == 0:
        mlflow_data = {
            "tracking_uri": tracking_uri,
            "experiment_name": experiment_name,
            "run_name": run_name,
            "run_id": logger.run_id,
        }

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(mlflow_data, save_dir / "mlflow.yaml")

    return logger


registry = Registry("lighting.logger")
registry.register(_build_csv_logger, key="CSVLogger")
registry.register(_build_tb_logger, key="TensorBoardLogger")
registry.register(_build_wandb_logger, key="WandbLogger")
registry.register(_build_aim_logger, key="AimLogger")
registry.register(_build_mlflow_logger, key="MLFlowLogger")


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf, paths: OmegaConf, meta: OmegaConf) -> Logger:
    return registry.from_config(conf, paths=paths, meta=meta)
