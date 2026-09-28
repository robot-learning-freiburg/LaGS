# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import math
import platform
import traceback
import warnings
from datetime import datetime
from typing import Any, List

import apprise
import lightning
import torch
from apprise import NotifyType
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks import ThroughputMonitor as _ThroughputMonitor
from lightning.pytorch.utilities.exceptions import SIGTERMException
from omegaconf import OmegaConf

from ... import config, utils
from ..registry import Registry
from . import ext

log = utils.log.get_logger(__name__)

registry = Registry("lighting.callback")
registry.register_from_module(lightning.pytorch.callbacks, Callback)
registry.register_from_module(ext.callbacks, Callback)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> Callback:
    return registry.from_config(conf)


@registry.register
class DetectUnusedParameters(Callback):
    """
    A PyTorch Lightning callback to detect unused model parameters during
    training.

    This callback identifies model parameters that do not participate in the
    backward pass during the first forward-backward pass. The detection occurs
    right before the optimizer step in the first training iteration. If unused
    parameters are found, the callback raises a `RuntimeError` and stops
    training to enforce fixing the issue.

    Key Features:
    - Automatically registers hooks on model parameters to track their usage
      during the backward pass.
    - Optionally raises an exception if unused parameters are detected.
    - Stops monitoring after the first iteration to avoid unnecessary overhead.

    Usage:
        Add this callback to your PyTorch Lightning Trainer:

        ```python
        trainer = Trainer(callbacks=[DetectUnusedParametersCallback()])
        ```

    Raises:
        RuntimeError: If unused parameters are detected during the first
        training iteration.
    """

    def __init__(self, raise_error=True):
        super().__init__()

        self.raise_error = raise_error
        self.params = {}

    def on_train_start(self, trainer, pl_module):
        # Initialize a dictionary to track parameter usage
        self.params = {
            name: False for name, p in pl_module.named_parameters() if p.requires_grad
        }

        # Register hooks to detect parameter usage during backward
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.mark_used(name))

    def mark_used(self, name):
        # Mark the parameter as used
        self.params[name] = True

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # This hook runs after the backward pass but before the optimizer step
        unused_params = [name for name, used in self.params.items() if not used]
        if unused_params:
            log.warning("Detected unused parameters: %s", unused_params)

            if self.raise_error:
                raise RuntimeError(f"Unused parameters detected: {unused_params}")

        else:
            log.info("No unused parameters detected")

        # Stop further checks to avoid redundant prints
        trainer.callbacks.remove(self)


@registry.register
class Notify(Callback):
    """
    PyTorch Lightning callback to send notifications during the training lifecycle.

    This callback uses Apprise to send notifications for the following events:
    - Training completion
    - Exceptions during training, including optional handling of keyboard
      interruptions (e.g., Ctrl+C).

    Attributes:
        services (List[str]): List of Apprise service URLs for sending notifications.
        notify_on_interrupt (bool): Whether to send notifications for keyboard
            interrupts (Ctrl+C).
    """

    def __init__(
        self,
        services: List[str],
        notify_on_interrupt: bool = False,
        notify_on_sigterm: bool = True,
    ):
        """
        Initialize the Notify callback.

        Args:
            services (List[str]): List of Apprise service URLs to configure
                notification services.
                Example: ['slack://TOKEN/CHANNEL', 'mailto://user:pass@smtp.server.com']
            notify_on_interrupt (bool): If True, sends notifications for
                keyboard interruptions (Ctrl+C). Default is False.
            notify_on_sigterm (bool): If True, sends notifications for
                interruptions via SIGTERM.
        """
        super().__init__()

        self.apprise_obj = None
        self.notify_on_interrupt = notify_on_interrupt
        self.notify_on_sigterm = notify_on_sigterm

        if utils.mp.rank == 0:
            self.apprise_obj = apprise.Apprise()

            for url in services:
                self.apprise_obj.add(url)

    def get_run_info(self) -> str:
        """
        Fetch the run name and description from the global config.
        """
        conf = config.get()

        name = "Unknown"
        name = OmegaConf.select(conf, "meta.run.name", default=name)

        description = "No description available"
        description = OmegaConf.select(
            conf, "meta.run.description", default=description
        )

        return f"*Run*: `{name}`\n*Description*: _{description}_\n"

    def get_host_info(self) -> str:
        """
        Fetch the host and SLURM job information (if available).
        """

        info = f"*Host*: `{platform.node()}`"

        container_host = utils.sys.get_container_hostname()
        if container_host is not None:
            info += f" on `{container_host}`"

        if utils.slurm.is_slurm_job():
            job_id = utils.slurm.get_job_id()
            restart_count = utils.slurm.get_restart_count()

            info += f"\n*SLURM Job ID*: `{job_id}` (restart count: {restart_count})"

        return info

    def notify(
        self, title: str, body: str, notify_type: NotifyType = NotifyType.INFO
    ) -> None:
        """
        Send a notification using Apprise.

        Args:
            title (str): The title of the notification.
            body (str): The body content of the notification (supports Markdown).
            notify_type (NotifyType): The type of the notification.
        """
        if self.apprise_obj is None:
            return

        self.apprise_obj.notify(
            title=title,
            body=body,
            notify_type=notify_type,
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        """
        Called when training ends successfully.

        Sends a notification indicating the completion of training.
        """
        host_info = self.get_host_info()
        run_info = self.get_run_info()

        body = "Training Completed Successfully!\n"
        body += f"{host_info}\n\n{run_info}"

        self.notify(
            title="Training Complete",
            body=body,
            notify_type=NotifyType.SUCCESS,
        )

    def on_exception(self, trainer, pl_module, exception: Exception) -> None:
        """
        Called when an exception occurs during training.

        Sends a notification with the exception details, formatted using `traceback`.
        If the exception is a KeyboardInterrupt and `notify_on_interrupt` is set to True,
        it sends a specific notification for user interruptions. If `notify_on_interrupt` is False,
        no notification is sent for KeyboardInterrupt.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The PyTorch Lightning module instance.
            exception (Exception): The exception that occurred.
        """
        host_info = self.get_host_info()
        run_info = self.get_run_info()

        if isinstance(exception, KeyboardInterrupt):
            if self.notify_on_interrupt:
                body = "Training Interrupted: _User interrupt (Ctrl+C)_\n\n"
                body += f"{host_info}\n{run_info}"
                self.notify(
                    title="Training Interrupted",
                    body=body,
                    notify_type=NotifyType.WARNING,
                )

        elif isinstance(exception, SIGTERMException):
            if self.notify_on_sigterm:
                body = "Training Interrupted: _SIGTERM signal received_\n\n"
                body += f"{host_info}\n{run_info}"
                self.notify(
                    title="Training Interrupted",
                    body=body,
                    notify_type=NotifyType.WARNING,
                )

        else:
            msg = "".join(traceback.format_exception_only(exception))

            trace = "".join(
                traceback.format_exception(
                    type(exception), exception, exception.__traceback__
                )
            )

            body = "Training Error: _An exception occurred during training._\n\n"
            body += f"{host_info}\n{run_info}\n\n"
            body += f"*Exception*:\n```\n{msg}\n```\n"
            body += f"*Traceback*:\n```\n{trace}\n```\n"

            self.notify(
                title="Training Error",
                body=body,
                notify_type=NotifyType.FAILURE,
            )

        # Re-raise the exception to ensure proper handling in Lightning
        raise exception


class EpochProgress:
    def __init__(self):
        self._start_time = datetime.now()
        self._time = datetime.now()
        self._step = 0
        self._initial_step = 0
        self._total_steps = math.inf

    def start(self, total_steps: int, initial_step: int = 0):
        self._start_time = datetime.now()
        self._time = datetime.now()
        self._step = initial_step
        self._initial_step = initial_step
        self._total_steps = total_steps

    def step(self, step: int):
        now = datetime.now()

        iter_time = now - self._time
        iter_time = iter_time / max(step - self._step, 1)

        total_time = now - self._start_time
        steps_completed = step - self._initial_step
        if math.isfinite(self._total_steps) and steps_completed > 0:
            time_per_step = total_time / steps_completed
            remaining_time = time_per_step * (self._total_steps - step)
        else:
            remaining_time = "N/A"

        self._step = step
        self._time = datetime.now()

        iter_time = self._format_iteration_time(iter_time)
        total_time = str(total_time).split(".", maxsplit=1)[0]
        remaining_time = str(remaining_time).split(".", maxsplit=1)[0]

        return iter_time, total_time, remaining_time

    def stop(self):
        delta = datetime.now() - self._start_time

        self._step = math.inf

        return delta

    def _format_iteration_time(self, sec):
        sec = sec.total_seconds()
        sec = max(sec, 1.0e-5)

        return f"{1.0 / sec:.2f}it/s"


@registry.register
class LogProgress(ProgressBar):
    # pylint: disable=too-many-public-methods
    # pylint: disable=too-many-instance-attributes

    def __init__(self, interval: int = 50, logger="progress", log_level="INFO"):
        super().__init__()

        if isinstance(logger, str):
            logger = utils.log.get_logger(logger)

        if isinstance(log_level, str):
            log_level = log_level.upper()
            log_level = logging.getLevelNamesMapping()[log_level]

        self._enabled = True
        self._interval = interval
        self._logger = logger
        self._log_level = log_level

        self._train_progress = EpochProgress()
        self._val_progress = EpochProgress()
        self._test_progress = EpochProgress()
        self._predict_progress = EpochProgress()

        self._train_start_time = None
        self._val_start_time = None
        self._test_start_time = None
        self._predict_start_time = None

        self._prefix = []

    @property
    def refresh_rate(self) -> int:
        return self._interval

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    def _prefix_push(self, prefix: str):
        self._prefix.append(prefix)

    def _prefix_pop(self):
        if self._prefix:
            self._prefix.pop()

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self._prefix_push(self.sanity_check_description)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level, "%s: Starting", f"[{' | '.join(self._prefix)}]"
        )

    def on_sanity_check_end(self, trainer, pl_module) -> None:
        if self.is_enabled:
            self._logger.log(
                self._log_level, "%s: Complete", f"[{' | '.join(self._prefix)}]"
            )

        self._prefix_pop()

    def on_train_start(self, trainer, pl_module) -> None:
        self._train_start_time = datetime.now()

        # Handle mid-epoch resume: if we're resuming in the middle of an epoch,
        # on_train_epoch_start won't be called, so we need to set up the prefix here
        if trainer.fit_loop.epoch_progress.current.started:
            initial_step = trainer.fit_loop.epoch_loop.batch_progress.current.completed
            self._train_progress.start(self.total_train_batches, initial_step)

            epoch = f"{trainer.current_epoch}"
            if trainer.max_epochs is not None:
                epoch += f"/{trainer.max_epochs - 1}"

            self._prefix_push(f"Epoch {epoch}")

        if self.is_disabled:
            return

        self._logger.log(self._log_level, "%s: Starting", self.train_description)

        if trainer.fit_loop.epoch_progress.current.started:
            self._logger.log(
                self._log_level,
                "%s: Resuming epoch from step %s/%s",
                f"[{' | '.join(self._prefix)}]",
                initial_step + 1,
                self.total_train_batches,
            )

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._train_progress.start(self.total_train_batches)

        epoch = f"{trainer.current_epoch}"
        if trainer.max_epochs is not None:
            epoch += f"/{trainer.max_epochs - 1}"

        self._prefix_push(f"Epoch {epoch}")

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Starting epoch",
            f"[{' | '.join(self._prefix)}]",
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self._should_update(batch_idx + 1, self.total_train_batches):
            return

        iter_time, total_time, remaining_time = self._train_progress.step(batch_idx + 1)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Step %s/%s (%s - %s | %s) %s",
            f"[{' | '.join(self._prefix)}]",
            batch_idx + 1,
            self.total_train_batches,
            total_time,
            remaining_time,
            iter_time,
            self.get_metrics(trainer, pl_module),
        )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        time = self._train_progress.stop()
        time = str(time).split(".", maxsplit=1)[0]

        if self.is_enabled:
            self._logger.log(
                self._log_level,
                "%s: Completed epoch in %s %s",
                f"[{' | '.join(self._prefix)}]",
                time,
                self.get_metrics(trainer, pl_module),
            )

        self._prefix_pop()

    def on_train_end(self, trainer, pl_module) -> None:
        if self.is_disabled:
            return

        time = datetime.now() - self._train_start_time
        time = str(time).split(".", maxsplit=1)[0]
        self._logger.log(
            self._log_level,
            "%s: Completed training for %d epochs in %s",
            self.train_description,
            trainer.current_epoch,
            time,
        )

    def on_validation_start(self, trainer, pl_module) -> None:
        self._val_start_time = datetime.now()
        self._prefix_push(self.validation_description)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Starting validation",
            f"[{' | '.join(self._prefix)}]",
        )

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self._val_progress.start(self.total_val_batches_current_dataloader)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_update(
            batch_idx + 1, self.total_val_batches_current_dataloader
        ):
            return

        iter_time, total_time, remaining_time = self._val_progress.step(batch_idx + 1)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Step %s/%s (%s - %s | %s)",
            f"[{' | '.join(self._prefix)}]",
            batch_idx + 1,
            self.total_val_batches_current_dataloader,
            total_time,
            remaining_time,
            iter_time,
        )

    def on_validation_end(self, trainer, pl_module) -> None:
        self.reset_dataloader_idx_tracker()

        if self.is_enabled:
            time = datetime.now() - self._val_start_time
            time = str(time).split(".", maxsplit=1)[0]
            self._logger.log(
                self._log_level,
                "%s: Completed in %s %s",
                f"[{' | '.join(self._prefix)}]",
                time,
                self.get_metrics(trainer, pl_module),
            )

        self._prefix_pop()

    def on_test_start(self, trainer, pl_module) -> None:
        self._test_start_time = datetime.now()
        self._prefix_push(self.test_description)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level, "%s: Starting", f"[{' | '.join(self._prefix)}]"
        )

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self._test_progress.start(self.total_test_batches_current_dataloader)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_update(
            batch_idx + 1, self.total_test_batches_current_dataloader
        ):
            return

        iter_time, total_time, remaining_time = self._test_progress.step(batch_idx + 1)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Step %s/%s (%s - %s | %s)",
            f"[{' | '.join(self._prefix)}]",
            batch_idx + 1,
            self.total_test_batches_current_dataloader,
            total_time,
            remaining_time,
            iter_time,
        )

    def on_test_end(self, trainer, pl_module) -> None:
        self.reset_dataloader_idx_tracker()

        if self.is_enabled:
            time = datetime.now() - self._test_start_time
            time = str(time).split(".", maxsplit=1)[0]
            self._logger.log(
                self._log_level,
                "%s: Completed in %s",
                f"[{' | '.join(self._prefix)}]",
                time,
            )

        self._prefix_pop()

    def on_predict_start(self, trainer, pl_module) -> None:
        self._predict_start_time = datetime.now()
        self._prefix_push(self.predict_description)

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level, "%s: Starting", f"[{' | '.join(self._prefix)}]"
        )

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self._predict_progress.start(self.total_predict_batches_current_dataloader)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if not self._should_update(
            batch_idx + 1, self.total_predict_batches_current_dataloader
        ):
            return

        iter_time, total_time, remaining_time = self._predict_progress.step(
            batch_idx + 1
        )

        if self.is_disabled:
            return

        self._logger.log(
            self._log_level,
            "%s: Step %s/%s (%s - %s | %s)",
            f"[{' | '.join(self._prefix)}]",
            batch_idx + 1,
            self.total_predict_batches_current_dataloader,
            total_time,
            remaining_time,
            iter_time,
        )

    def on_predict_end(self, trainer, pl_module) -> None:
        self.reset_dataloader_idx_tracker()

        if self.is_enabled:
            time = datetime.now() - self._predict_start_time
            time = str(time).split(".", maxsplit=1)[0]
            self._logger.log(
                self._log_level,
                "%s: Completed in %s",
                f"[{' | '.join(self._prefix)}]",
                time,
            )

        self._prefix_pop()

    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (
            current % self.refresh_rate == 0 or current == total
        )


@registry.register
class ContiguityCheck(Callback):
    """
    A PyTorch Lightning callback to check for parameter contiguity.

    This callback checks if the model's parameters are contiguous in memory
    during the first training iteration. If any non-contiguous parameters are
    found, it raises a warning with details about the non-contiguous parameters.
    """

    def __init__(self, max_steps: int = None):
        """
        Args:
            max_steps (int, optional): Only check for the first N steps. If
                None, check every step.
        """
        super().__init__()
        self.max_steps = max_steps
        self._step_count = 0

    def on_fit_start(self, trainer, pl_module):
        # If max_steps is set and exceeded, remove this callback from the trainer
        if self.max_steps is not None and self._step_count >= self.max_steps:
            if self in trainer.callbacks:
                trainer.callbacks.remove(self)
            return

        non_contiguous_params = []

        for name, param in pl_module.named_parameters():
            if not param.is_contiguous():
                non_contiguous_params.append((name, param.shape, param.stride()))

        if non_contiguous_params:
            message = "Non-contiguous parameters found:"
            for name, shape, stride in non_contiguous_params:
                message += f"\n - {name}: shape={shape}, stride={stride}"

            warnings.warn(message)

        self._step_count += 1


@registry.register
class GradStrideCheck(Callback):
    """
    A PyTorch Lightning callback to check for mismatched parameter and gradient strides.
    This callback checks if the strides of model parameters and their gradients
    match during the backward pass. If a mismatch is detected, it raises a warning
    with details about the mismatched parameters and gradients.

    Useful for debugging bucketing issues or ensuring that the model's
    parameters and gradients are correctly aligned, especially in distributed
    training scenarios.
    """

    def __init__(self, max_steps: int = None):
        """
        Args:
            max_steps (int, optional): Only check for the first N steps. If
                None, check every step.
        """
        super().__init__()

        self.max_steps = max_steps
        self._step_count = 0

    def on_after_backward(self, trainer, pl_module):
        # If max_steps is set and exceeded, remove this callback from the trainer
        if self.max_steps is not None and self._step_count >= self.max_steps:
            if self in trainer.callbacks:
                trainer.callbacks.remove(self)
            return

        mismatched = []
        for name, param in pl_module.named_parameters():
            if param.grad is not None and param.requires_grad:
                if param.stride() != param.grad.stride():
                    mismatched.append(
                        (
                            name,
                            param.shape,
                            param.stride(),
                            param.grad.shape,
                            param.grad.stride(),
                        )
                    )

        if mismatched:
            message = "Mismatched param/grad strides detected:"
            for name, p_shape, p_stride, g_shape, g_stride in mismatched:
                message += (
                    f"\n - {name}:"
                    f"\n   param shape={p_shape}, stride={p_stride}"
                    f"\n   grad  shape={g_shape}, stride={g_stride}"
                )

            warnings.warn(message)

        self._step_count += 1


@registry.register
class PeakMemoryMonitor(Callback):
    """
    A PyTorch Lightning callback to measure and log peak GPU memory usage.

    Resets peak memory statistics before each batch and records the maximum
    allocated and reserved CUDA memory after each batch completes. Works
    across all stages (train, validation, test, predict).

    Metrics are logged (in GiB) via the Lightning module's log method:
    - ``<stage>/peak_memory_allocated_gib``
    - ``<stage>/peak_memory_reserved_gib``

    If CUDA is not available the callback is a no-op.
    """

    def __init__(self, log_every_n_steps: int = 1, warmup_steps: int = 0):
        """
        Args:
            log_every_n_steps (int): Log peak memory every N batches.
                Defaults to 1 (every batch).
            warmup_steps (int): Number of batches to skip at the start of each
                stage before logging begins. Defaults to 0.
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps

    def _reset(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def _log_peak(self, pl_module, stage: str, batch_idx: int):
        if not torch.cuda.is_available():
            return
        if batch_idx < self.warmup_steps:
            return
        if (batch_idx - self.warmup_steps) % self.log_every_n_steps != 0:
            return

        gib = 1024**3
        allocated = torch.cuda.max_memory_allocated() / gib
        reserved = torch.cuda.max_memory_reserved() / gib

        pl_module.log(f"{stage}/peak_mem_alloc_GiB", allocated, reduce_fx="max")
        pl_module.log(f"{stage}/peak_mem_alloc_mean_GiB", allocated, reduce_fx="mean")
        pl_module.log(f"{stage}/peak_mem_alloc_min_GiB", allocated, reduce_fx="min")
        pl_module.log(f"{stage}/peak_mem_reserved_GiB", reserved, reduce_fx="max")
        pl_module.log(f"{stage}/peak_mem_reserved_mean_GiB", reserved, reduce_fx="mean")
        pl_module.log(f"{stage}/peak_mem_reserved_min_GiB", reserved, reduce_fx="min")

    # --- train ---

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_peak(pl_module, "train", batch_idx)

    # --- validation ---

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._reset()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log_peak(pl_module, "val", batch_idx)

    # --- test ---

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._reset()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log_peak(pl_module, "test", batch_idx)

    # --- predict ---

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._reset()

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._log_peak(pl_module, "predict", batch_idx)
