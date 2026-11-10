"""The Snake game, packaged as a reinforcement-learning environment.

The agent never sees the whole board. It sees a small "state" tuple of 8 numbers
(see `SnakeEnvironment.observe`), which keeps the learning table tiny.
"""

import random
from collections import deque

from config import GameConfig, RewardConfig

Cell = tuple[int, int]
State = tuple[int, ...]

# Headings in clockwise order, so "turn right" is +1 and "turn left" is -1.
HEADINGS: list[Cell] = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left

# Actions are relative to where the snake is currently looking.
ACTION_STRAIGHT, ACTION_TURN_RIGHT, ACTION_TURN_LEFT = 0, 1, 2
ACTION_COUNT = 3
TURN_OFFSETS = {ACTION_STRAIGHT: 0, ACTION_TURN_RIGHT: 1, ACTION_TURN_LEFT: -1}


class SnakeEnvironment:
    def __init__(
        self,
        game_config: GameConfig = GameConfig(),
        reward_config: RewardConfig = RewardConfig(),
        seed: int | None = None,
    ):
        self.game_config = game_config
        self.rewards = reward_config
        self._random = random.Random(seed)
        self.reset()

    @property
    def grid_size(self) -> int:
        return self.game_config.grid_size

    @property
    def head(self) -> Cell:
        return self.snake[0]

    def reset(self) -> State:
        middle = self.grid_size // 2
        self.snake: deque[Cell] = deque([(middle, middle)])  # head is the first item
        self.heading_index = self._random.randrange(len(HEADINGS))
        self.score = 0
        self.steps_since_food = 0
        self._place_food()
        return self.observe()

    def step(self, action: int) -> tuple[State, float, bool]:
        """Advance one tick. Returns (new_state, reward, episode_is_over)."""
        self.heading_index = (self.heading_index + TURN_OFFSETS[action]) % len(HEADINGS)
        distance_before = self._distance_to_food(self.head)
        new_head = self._cell_in_direction(self.head, self.heading_index)
        ate_food = new_head == self.food

        if self._is_deadly(new_head, tail_moves_away=not ate_food):
            return self.observe(), self.rewards.death, True
        self.snake.appendleft(new_head)
        if not ate_food:
            self.snake.pop()

        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
            board_is_full = self.food is None
            return self.observe(), self.rewards.food, board_is_full

        self.steps_since_food += 1
        if self.steps_since_food > self.game_config.starvation_limit:
            return self.observe(), self.rewards.death, True
        got_closer = self._distance_to_food(new_head) < distance_before
        reward = self.rewards.toward_food if got_closer else self.rewards.away_from_food
        return self.observe(), reward, False

    def observe(self) -> State:
        """(danger ahead, danger right, danger left, heading, food up, food right, food down, food left)"""
        ahead, right, left = (
            (self.heading_index + offset) % len(HEADINGS) for offset in (0, 1, -1)
        )
        food_offset_x, food_offset_y = self._offset_to_food(self.head)
        return (
            int(self._is_deadly(self._cell_in_direction(self.head, ahead))),
            int(self._is_deadly(self._cell_in_direction(self.head, right))),
            int(self._is_deadly(self._cell_in_direction(self.head, left))),
            self.heading_index,
            int(food_offset_y < 0),
            int(food_offset_x > 0),
            int(food_offset_y > 0),
            int(food_offset_x < 0),
        )

    def _place_food(self) -> None:
        occupied = set(self.snake)
        free_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        self.food: Cell | None = self._random.choice(free_cells) if free_cells else None

    def _is_deadly(self, cell: Cell, tail_moves_away: bool = False) -> bool:
        """Only the snake's own body kills: the walls wrap around to the opposite side."""
        return cell in self.snake and not (tail_moves_away and cell == self.snake[-1])

    def _offset_to_food(self, cell: Cell) -> Cell:
        """Shortest signed (x, y) offset to the food, going through the wrap-around if shorter."""
        food_x, food_y = self.food or cell
        half = self.grid_size // 2
        return (
            (food_x - cell[0] + half) % self.grid_size - half,
            (food_y - cell[1] + half) % self.grid_size - half,
        )

    def _distance_to_food(self, cell: Cell) -> int:
        offset_x, offset_y = self._offset_to_food(cell)
        return abs(offset_x) + abs(offset_y)

    def _cell_in_direction(self, cell: Cell, heading_index: int) -> Cell:
        step_x, step_y = HEADINGS[heading_index]
        return (cell[0] + step_x) % self.grid_size, (cell[1] + step_y) % self.grid_size
