from dataclasses import dataclass
from typing import Any, Optional

from accelerate import Accelerator
from torch import Tensor


@dataclass
class Metric:
    step_value: float = 0
    epoch_value: float = 0
    epoch_steps: int = 0


class MetricLogger:
    def __init__(
        self,
        accelerator: Accelerator,
        log_interval: Optional[int] = None,
    ):
        self.accelerator = accelerator
        self.log_interval = log_interval

        # Keepe track of the global steps and current epoch
        self.global_steps = 0
        self.epoch = 0

        self.metric_dict = {}

    def add_metric(self, metric_name: str, on_step: bool = False, on_epoch: bool = False) -> None:
        """Registers a metric to be logged."""
        self.metric_dict[metric_name] = Metric(on_step=on_step, on_epoch=on_epoch)

    def log(self, metric_name: str, value: float, on_step=False, on_epoch=False) -> None:
        """Logs the metric value and resets it to 0."""

        # Add the metric if it hasn't been added yet
        if metric_name not in self.metric_dict:
            self.metric_dict[metric_name] = Metric()

        # Aggregate the metric
        metric = self.metric_dict[metric_name]

        # Increment the appropriate metric value
        if on_step:
            metric.step_value += value
        if on_epoch:
            metric.epoch_value += value
            metric.epoch_steps += 1

        # Every log_interval steps, log metric to WandB
        if on_step and ((self.global_steps % self.log_interval) == 0):
            # Log the metric and the current epoch
            self.accelerator.log(
                {metric_name: metric.step_value / self.log_interval}, step=self.global_steps
            )
            self.accelerator.log({"Epoch": self.epoch}, step=self.global_steps)

            # Reset the metric
            metric.step_value = 0

    def step(self) -> None:
        """Increments the global step counter."""
        self.global_steps += 1

    def compute(self, metric_name: str) -> float:
        """Computes the current average metric value over the epoch."""
        metric = self.metric_dict[metric_name]
        return metric.epoch_value / metric.epoch_steps

    def reset(self) -> None:
        """Resets the internal metric dictionary."""
        self.metric_dict = {}

    def epoch_end(self):
        """Logs the average metric value over the epoch and resets it to 0."""

        # Go through each metric, looking for values that need to be logged at the end of the epoch
        for metric_name in self.metric_dict:
            metric = self.metric_dict[metric_name]

            # If we recorded steps throughout the epoch, then log it
            if metric.epoch_steps > 0:
                self.accelerator.log(
                    {"Epoch " + metric_name: metric.epoch_value / metric.epoch_steps},
                    step=self.global_steps,
                )

            metric.epoch_value = 0
            metric.epoch_steps = 0
