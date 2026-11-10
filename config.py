"""All tunable numbers of the project in one place."""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "snake_brain.json"


@dataclass(frozen=True)
class GameConfig:
    grid_size: int = 10

    @property
    def starvation_limit(self) -> int:
        """Steps allowed without eating before the episode is cut (prevents endless loops)."""
        return self.grid_size ** 2


@dataclass(frozen=True)
class RewardConfig:
    food: float = 10.0
    death: float = -10.0
    toward_food: float = 0.1
    away_from_food: float = -0.15


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon_start: float = 1.0  # 1.0 = fully random at the beginning
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.998  # applied once per episode


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 3000
    report_every: int = 250
    patience_episodes: int = 1500  # stop early if the average score stops improving
    max_seconds: float | None = None  # optional hard time budget
