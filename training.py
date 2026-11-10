"""The learning loop. Shared by headless training and by the live viewer."""

import time
from typing import Callable

from agent import QLearningAgent
from config import TrainingConfig
from environment import SnakeEnvironment, State


def play_step(
    environment: SnakeEnvironment, agent: QLearningAgent, state: State, learning: bool
) -> tuple[State, bool]:
    """One move: the agent chooses, the game reacts, and (if learning) the agent updates."""
    action = agent.choose_action(state, explore=learning)
    next_state, reward, episode_over = environment.step(action)
    if learning:
        agent.learn(state, action, reward, next_state, episode_over)
    return next_state, episode_over


def run_episode(environment: SnakeEnvironment, agent: QLearningAgent, learning: bool) -> int:
    """Play one full game and return the score."""
    state, episode_over = environment.reset(), False
    while not episode_over:
        state, episode_over = play_step(environment, agent, state, learning)
    return environment.score


def average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def train(
    agent: QLearningAgent,
    environment: SnakeEnvironment,
    config: TrainingConfig = TrainingConfig(),
    log: Callable[[str], None] = print,
) -> list[int]:
    """Train on the CPU only, and stop as soon as more training stops paying off."""
    scores: list[int] = []
    best_average = float("-inf")
    episodes_without_improvement = 0
    started_at = time.monotonic()

    for episode in range(1, config.episodes + 1):
        scores.append(run_episode(environment, agent, learning=True))
        agent.decay_epsilon()
        if episode % config.report_every:
            continue

        recent_average = average(scores[-config.report_every:])
        log(f"episode {episode:5d} | average score {recent_average:5.2f} | epsilon {agent.epsilon:.3f}")
        if recent_average > best_average:
            best_average, episodes_without_improvement = recent_average, 0
        else:
            episodes_without_improvement += config.report_every

        if episodes_without_improvement >= config.patience_episodes:
            log("no more improvement - stopping early to save energy")
            break
        if config.max_seconds and time.monotonic() - started_at >= config.max_seconds:
            log("time budget reached - stopping")
            break
    return scores


def evaluate(agent: QLearningAgent, environment: SnakeEnvironment, episodes: int = 100) -> list[int]:
    """Play with the learned policy only (no exploring, no learning)."""
    return [run_episode(environment, agent, learning=False) for _ in range(episodes)]
