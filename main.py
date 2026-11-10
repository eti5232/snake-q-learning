"""Command line entry point.

    python main.py train    # fast, no window, saves the learned brain
    python main.py watch    # watch the saved brain play
    python main.py learn    # watch the brain learn from zero, live
"""

import argparse
from pathlib import Path

from agent import QLearningAgent
from config import DEFAULT_MODEL_PATH, TrainingConfig
from environment import SnakeEnvironment
from training import average, evaluate, train


def run_train(args: argparse.Namespace) -> None:
    environment = SnakeEnvironment(seed=args.seed)
    agent = QLearningAgent(seed=args.seed)
    train(agent, environment, TrainingConfig(episodes=args.episodes, max_seconds=args.max_seconds))

    scores = evaluate(agent, environment)
    print(f"greedy test over {len(scores)} games: average {average(scores):.2f}, best {max(scores)}")
    agent.save(args.model)
    print(f"brain saved to {args.model} ({len(agent.q_table)} states known)")


def run_watch(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise SystemExit(f"No saved brain at {args.model}. Run `python main.py train` first.")
    from viewer import launch  # imported lazily so `train` works without a display

    launch(SnakeEnvironment(seed=args.seed), QLearningAgent.load(args.model), learning=False, model_path=args.model)


def run_learn(args: argparse.Namespace) -> None:
    from viewer import launch

    launch(SnakeEnvironment(seed=args.seed), QLearningAgent(seed=args.seed), learning=True, model_path=args.model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A snake that teaches itself with Q-learning.")
    commands = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        command.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="where the brain is stored")
        command.add_argument("--seed", type=int, default=None, help="fix randomness for repeatable runs")
        return command

    train_command = add_command("train", run_train, "train without a window and save the brain")
    train_command.add_argument("--episodes", type=int, default=TrainingConfig.episodes)
    train_command.add_argument("--max-seconds", type=float, default=None, help="optional time budget")
    add_command("watch", run_watch, "watch the saved brain play")
    add_command("learn", run_learn, "watch the brain learn live (saved when you close the window)")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.handler(arguments)
