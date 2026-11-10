"""A small tkinter window (standard library, nothing to install) that shows the snake."""

import tkinter as tk
from pathlib import Path

from agent import QLearningAgent
from environment import HEADINGS, Cell, SnakeEnvironment
from training import average, play_step

CELL_PIXELS = 36
FRAME_DELAY_MS = 50
MAX_STEPS_PER_FRAME = 300  # the speed slider; drawing is skipped for all but the last step
CHART_WIDTH, CHART_HEIGHT = 300, 140
EPISODES_PER_CHART_POINT = 20
RECENT_EPISODES = 50

FONT = "Segoe UI"
WINDOW_BG, CARD_BG = "#11111b", "#1e1e2e"
TILE_LIGHT, TILE_DARK = "#24243a", "#1f1f33"
TEXT, MUTED = "#cdd6f4", "#7f849c"
SNAKE_HEAD, SNAKE_TAIL = "#a6e3a1", "#2f8f4e"
FOOD, FOOD_LEAF, EYE_WHITE, EYE_PUPIL = "#f38ba8", "#a6e3a1", "#ffffff", "#11111b"
ACCENT = "#89b4fa"


def blend(color_from: str, color_to: str, ratio: float) -> str:
    """Mix two '#rrggbb' colors; ratio 0 gives the first color, 1 gives the second."""
    start, end = (tuple(int(color[i:i + 2], 16) for i in (1, 3, 5)) for color in (color_from, color_to))
    return "#" + "".join(f"{round(a + (b - a) * ratio):02x}" for a, b in zip(start, end))


class SnakeViewer:
    def __init__(
        self, root: tk.Tk, environment: SnakeEnvironment, agent: QLearningAgent, learning: bool, model_path: Path
    ):
        self.root, self.environment, self.agent = root, environment, agent
        self.learning, self.model_path = learning, model_path
        self.state = environment.reset()
        self.scores: list[int] = []
        self.paused = False
        self.stat_texts: dict[str, tk.StringVar] = {}
        self._build_widgets()
        root.protocol("WM_DELETE_WINDOW", self._close)

    def start(self) -> None:
        self._tick()
        self.root.mainloop()

    # ---------- building the window ----------

    def _build_widgets(self) -> None:
        self.root.title("Snake AI")
        self.root.configure(bg=WINDOW_BG)
        self.root.resizable(False, False)

        board_pixels = self.environment.grid_size * CELL_PIXELS
        self.board = tk.Canvas(self.root, width=board_pixels, height=board_pixels, bg=CARD_BG, highlightthickness=0)
        self.board.grid(row=0, column=0, padx=(20, 12), pady=20, sticky="n")
        self._draw_board_tiles()

        side_panel = tk.Frame(self.root, bg=WINDOW_BG)
        side_panel.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="n")
        self._build_header(side_panel)
        self._build_stat_cards(side_panel)
        self._build_controls(side_panel)
        self._build_chart(side_panel)

    def _build_header(self, parent: tk.Frame) -> None:
        subtitle = "Learning live from zero" if self.learning else "Playing with the learned brain"
        tk.Label(parent, text="Snake AI", font=(FONT, 22, "bold"), fg=TEXT, bg=WINDOW_BG).pack(anchor="w")
        tk.Label(parent, text=subtitle, font=(FONT, 10), fg=MUTED, bg=WINDOW_BG).pack(anchor="w", pady=(0, 10))

    def _build_stat_cards(self, parent: tk.Frame) -> None:
        grid = tk.Frame(parent, bg=WINDOW_BG)
        grid.pack(fill="x")
        titles = ["Game", "Score", "Best", f"Average ({RECENT_EPISODES})", "States learned", "Random moves"]
        for index, title in enumerate(titles):
            card = self._make_card(grid)
            card.grid(row=index // 2, column=index % 2, padx=3, pady=3, sticky="ew")
            grid.columnconfigure(index % 2, weight=1)
            tk.Label(card, text=title, font=(FONT, 9), fg=MUTED, bg=CARD_BG).pack(anchor="w")
            self.stat_texts[title] = tk.StringVar()
            tk.Label(card, textvariable=self.stat_texts[title], font=(FONT, 18, "bold"), fg=TEXT, bg=CARD_BG).pack(anchor="w")

    def _build_controls(self, parent: tk.Frame) -> None:
        card = self._make_card(parent)
        card.pack(fill="x", pady=(9, 3), padx=3)
        tk.Label(card, text="Speed (steps per frame)", font=(FONT, 9), fg=MUTED, bg=CARD_BG).pack(anchor="w")
        self.steps_per_frame = tk.Scale(
            card, from_=1, to=MAX_STEPS_PER_FRAME, orient="horizontal", bg=CARD_BG, fg=TEXT, troughcolor=WINDOW_BG,
            activebackground=ACCENT, highlightthickness=0, bd=0, sliderrelief="flat", font=(FONT, 9),
        )
        self.steps_per_frame.pack(fill="x")
        self.pause_button = tk.Button(
            card, text="Pause", command=self._toggle_pause, font=(FONT, 10, "bold"), bg=ACCENT, fg=WINDOW_BG,
            activebackground=blend(ACCENT, "#ffffff", 0.25), relief="flat", bd=0, cursor="hand2", pady=5,
        )
        self.pause_button.pack(fill="x", pady=(6, 0))

    def _build_chart(self, parent: tk.Frame) -> None:
        card = self._make_card(parent)
        card.pack(fill="x", pady=3, padx=3)
        tk.Label(
            card, text=f"Average score per {EPISODES_PER_CHART_POINT} games", font=(FONT, 9), fg=MUTED, bg=CARD_BG
        ).pack(anchor="w")
        self.chart = tk.Canvas(card, width=CHART_WIDTH, height=CHART_HEIGHT, bg=CARD_BG, highlightthickness=0)
        self.chart.pack()

    @staticmethod
    def _make_card(parent: tk.Misc) -> tk.Frame:
        return tk.Frame(parent, bg=CARD_BG, padx=12, pady=8)

    # ---------- controls ----------

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def _close(self) -> None:
        if self.learning:
            self.agent.save(self.model_path)
            print(f"brain saved to {self.model_path} ({len(self.agent.q_table)} states known)")
        self.root.destroy()

    # ---------- the loop ----------

    def _tick(self) -> None:
        if not self.paused:
            for _ in range(self.steps_per_frame.get()):
                self._advance()
            self._redraw()
        self.root.after(FRAME_DELAY_MS, self._tick)

    def _advance(self) -> None:
        self.state, episode_over = play_step(self.environment, self.agent, self.state, self.learning)
        if episode_over:
            self.scores.append(self.environment.score)
            if self.learning:
                self.agent.decay_epsilon()
            self.state = self.environment.reset()

    # ---------- drawing ----------

    def _redraw(self) -> None:
        self._draw_snake_and_food()
        self._draw_chart()
        self._update_stats()

    def _update_stats(self) -> None:
        random_moves = f"{self.agent.epsilon:.0%}" if self.learning else "off"
        values = [
            str(len(self.scores) + 1),
            str(self.environment.score),
            str(max(self.scores, default=0)),
            f"{average(self.scores[-RECENT_EPISODES:]):.1f}",
            str(len(self.agent.q_table)),
            random_moves,
        ]
        for text_variable, value in zip(self.stat_texts.values(), values):
            text_variable.set(value)

    def _draw_board_tiles(self) -> None:
        size = self.environment.grid_size
        for x in range(size):
            for y in range(size):
                color = TILE_LIGHT if (x + y) % 2 else TILE_DARK
                self.board.create_rectangle(*self._cell_bounds((x, y), 0), fill=color, width=0)

    def _draw_snake_and_food(self) -> None:
        self.board.delete("moving")
        if self.environment.food:
            self._draw_food(self.environment.food)
        segments = list(self.environment.snake)
        for index in reversed(range(len(segments))):  # the head is drawn last, on top
            fade = index / max(len(segments) - 1, 1)
            self._draw_rounded(segments[index], blend(SNAKE_HEAD, SNAKE_TAIL, fade), padding=2)
        self._draw_eyes(self.environment.head)

    def _draw_food(self, cell: Cell) -> None:
        x0, y0, x1, y1 = self._cell_bounds(cell, 7)
        self.board.create_oval(x0, y0, x1, y1, fill=FOOD, width=0, tags="moving")
        self.board.create_oval(x1 - 12, y0 - 3, x1 - 3, y0 + 4, fill=FOOD_LEAF, width=0, tags="moving")

    def _draw_eyes(self, head: Cell) -> None:
        x0, y0, x1, y1 = self._cell_bounds(head, 0)
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        forward_x, forward_y = HEADINGS[self.environment.heading_index]
        side_x, side_y = -forward_y, forward_x
        for side in (-1, 1):
            eye_x = center_x + (forward_x * 0.12 + side_x * side * 0.22) * CELL_PIXELS
            eye_y = center_y + (forward_y * 0.12 + side_y * side * 0.22) * CELL_PIXELS
            self._draw_dot(eye_x, eye_y, 0.13 * CELL_PIXELS, EYE_WHITE)
            self._draw_dot(eye_x + forward_x * 2, eye_y + forward_y * 2, 0.07 * CELL_PIXELS, EYE_PUPIL)

    def _draw_dot(self, x: float, y: float, radius: float, color: str) -> None:
        self.board.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, width=0, tags="moving")

    def _draw_rounded(self, cell: Cell, color: str, padding: int) -> None:
        x0, y0, x1, y1 = self._cell_bounds(cell, padding)
        r = CELL_PIXELS * 0.25
        outline = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1,
                   x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
        self.board.create_polygon(outline, smooth=True, fill=color, tags="moving")

    @staticmethod
    def _cell_bounds(cell: Cell, padding: int) -> tuple[int, int, int, int]:
        x, y = cell[0] * CELL_PIXELS, cell[1] * CELL_PIXELS
        return x + padding, y + padding, x + CELL_PIXELS - padding, y + CELL_PIXELS - padding

    def _draw_chart(self) -> None:
        self.chart.delete("all")
        blocks = [
            average(self.scores[start:start + EPISODES_PER_CHART_POINT])
            for start in range(0, len(self.scores), EPISODES_PER_CHART_POINT)
        ]
        top, bottom, margin = 18, CHART_HEIGHT - 6, 4
        for fraction in (0, 0.5, 1):
            y = bottom - fraction * (bottom - top)
            self.chart.create_line(margin, y, CHART_WIDTH - margin, y, fill=TILE_LIGHT)
        if len(blocks) < 2:
            self.chart.create_text(CHART_WIDTH / 2, CHART_HEIGHT / 2, text="waiting for data...", fill=MUTED, font=(FONT, 9))
            return

        highest = max(max(blocks), 1)
        x_step = (CHART_WIDTH - 2 * margin) / (len(blocks) - 1)
        points = [
            (margin + index * x_step, bottom - value / highest * (bottom - top)) for index, value in enumerate(blocks)
        ]
        flat_points = [coordinate for point in points for coordinate in point]
        area = flat_points + [points[-1][0], bottom, points[0][0], bottom]
        self.chart.create_polygon(area, fill=blend(CARD_BG, ACCENT, 0.18), width=0)
        self.chart.create_line(*flat_points, fill=ACCENT, width=2, smooth=True)
        last_x, last_y = points[-1]
        self.chart.create_oval(last_x - 4, last_y - 4, last_x + 4, last_y + 4, fill=ACCENT, width=0)
        self.chart.create_text(margin, 2, anchor="nw", fill=MUTED, font=(FONT, 9), text=f"max {highest:.1f}")


def launch(environment: SnakeEnvironment, agent: QLearningAgent, learning: bool, model_path: Path) -> None:
    SnakeViewer(tk.Tk(), environment, agent, learning, model_path).start()
