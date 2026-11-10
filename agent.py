"""A tabular Q-learning agent: the whole "brain" is a small dictionary.

q_table[state] = [value of going straight, value of turning right, value of turning left]
After each move the value of the chosen action is nudged toward
    reward + discount * (best value available in the next state)
"""

import json
import random
from pathlib import Path

from config import AgentConfig
from environment import ACTION_COUNT, State


class QLearningAgent:
    def __init__(self, config: AgentConfig = AgentConfig(), seed: int | None = None):
        self.config = config
        self.epsilon = config.epsilon_start
        self.q_table: dict[State, list[float]] = {}
        self._random = random.Random(seed)

    def choose_action(self, state: State, explore: bool) -> int:
        """Explore = sometimes try a random move (needed while learning)."""
        if explore and self._random.random() < self.epsilon:
            return self._random.randrange(ACTION_COUNT)
        values = self._values_of(state)
        best_value = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best_value]
        return self._random.choice(best_actions)

    def learn(self, state: State, action: int, reward: float, next_state: State, done: bool) -> None:
        future_value = 0.0 if done else max(self._values_of(next_state))
        target = reward + self.config.discount_factor * future_value
        values = self._values_of(state)
        values[action] += self.config.learning_rate * (target - values[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        readable_table = {",".join(map(str, state)): values for state, values in self.q_table.items()}
        path.write_text(json.dumps(readable_table, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, config: AgentConfig = AgentConfig()) -> "QLearningAgent":
        agent = cls(config)
        stored_table = json.loads(path.read_text(encoding="utf-8"))
        agent.q_table = {tuple(map(int, key.split(","))): values for key, values in stored_table.items()}
        return agent

    def _values_of(self, state: State) -> list[float]:
        return self.q_table.setdefault(state, [0.0] * ACTION_COUNT)
