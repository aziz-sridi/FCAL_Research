from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class _State:
    current_ratio: float
    best_accuracy: float = float("-inf")
    best_localization_loss: float = float("inf")
    bad_epochs: int = 0
    plateau_epochs: int = 0
    cooldown_remaining: int = 0
    last_event: str = "initialized"


class AutoFCALScheduler:
    """Reduce-on-harm / increase-on-safe-plateau controller for FCAL strength."""

    def __init__(
        self,
        *,
        initial_gradient_ratio: float = 0.05,
        minimum_gradient_ratio: float = 0.01,
        maximum_gradient_ratio: float = 0.20,
        increase_factor: float = 1.25,
        decrease_factor: float = 0.5,
        patience: int = 2,
        cooldown: int = 1,
        accuracy_drop_tolerance: float = 0.005,
        localization_plateau_tolerance: float = 0.001,
        minimum_localization_loss: float = 1e-5,
        warmup_epochs: int = 5,
        ramp_epochs: int = 5,
        release_start_epoch: int | None = None,
        release_end_epoch: int | None = None,
    ) -> None:
        if not 0 <= minimum_gradient_ratio <= initial_gradient_ratio <= maximum_gradient_ratio:
            raise ValueError("Require min_ratio <= initial_ratio <= max_ratio")
        if increase_factor < 1 or not 0 < decrease_factor <= 1:
            raise ValueError("increase_factor must be >=1 and decrease_factor must be in (0,1]")
        self.minimum_gradient_ratio = float(minimum_gradient_ratio)
        self.maximum_gradient_ratio = float(maximum_gradient_ratio)
        self.increase_factor = float(increase_factor)
        self.decrease_factor = float(decrease_factor)
        self.patience = int(patience)
        self.cooldown = int(cooldown)
        self.accuracy_drop_tolerance = float(accuracy_drop_tolerance)
        self.localization_plateau_tolerance = float(localization_plateau_tolerance)
        self.minimum_localization_loss = float(minimum_localization_loss)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.release_start_epoch = release_start_epoch
        self.release_end_epoch = release_end_epoch
        self.state = _State(current_ratio=float(initial_gradient_ratio))

    def phase(self, epoch: int) -> float:
        """Return the zero-based epoch ramp multiplier."""
        human_epoch = int(epoch) + 1
        if human_epoch <= self.warmup_epochs:
            return 0.0
        ramp = min(1.0, (human_epoch - self.warmup_epochs) / max(1, self.ramp_epochs))
        if self.release_start_epoch is None or self.release_end_epoch is None:
            return ramp
        if human_epoch <= int(self.release_start_epoch):
            return ramp
        if human_epoch >= int(self.release_end_epoch):
            return 0.0
        progress = (human_epoch - int(self.release_start_epoch)) / max(
            1, int(self.release_end_epoch) - int(self.release_start_epoch),
        )
        return ramp * 0.5 * (1.0 + math.cos(math.pi * progress))

    @property
    def current_ratio(self) -> float:
        return self.state.current_ratio

    def step(self, epoch: int, validation_accuracy: float, localization_loss: float) -> str:
        """Update after validation, like a metric-driven learning-rate scheduler."""
        accuracy = float(validation_accuracy)
        localization = float(localization_loss)
        if accuracy > self.state.best_accuracy:
            self.state.best_accuracy = accuracy
            self.state.bad_epochs = 0
        elif accuracy < self.state.best_accuracy - self.accuracy_drop_tolerance:
            self.state.bad_epochs += 1
        else:
            self.state.bad_epochs = 0

        improvement = self.state.best_localization_loss - localization
        if improvement > self.localization_plateau_tolerance:
            self.state.best_localization_loss = localization
            self.state.plateau_epochs = 0
        else:
            self.state.plateau_epochs += 1

        if int(epoch) + 1 <= self.warmup_epochs + self.ramp_epochs:
            event = "warmup_or_ramp"
        elif self.state.cooldown_remaining > 0:
            self.state.cooldown_remaining -= 1
            event = "cooldown"
        elif self.state.bad_epochs >= self.patience:
            previous = self.state.current_ratio
            self.state.current_ratio = max(
                self.minimum_gradient_ratio,
                previous * self.decrease_factor,
            )
            self.state.bad_epochs = 0
            self.state.plateau_epochs = 0
            self.state.cooldown_remaining = self.cooldown
            event = "backoff" if self.state.current_ratio < previous else "at_minimum"
        elif (
            self.state.plateau_epochs >= self.patience
            and localization > self.minimum_localization_loss
            and accuracy >= self.state.best_accuracy - self.accuracy_drop_tolerance
        ):
            previous = self.state.current_ratio
            self.state.current_ratio = min(
                self.maximum_gradient_ratio,
                previous * self.increase_factor,
            )
            self.state.plateau_epochs = 0
            self.state.cooldown_remaining = self.cooldown
            event = "increase" if self.state.current_ratio > previous else "at_maximum"
        else:
            event = "hold"
        self.state.last_event = event
        return event

    def state_dict(self) -> dict:
        return asdict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = _State(**state)
