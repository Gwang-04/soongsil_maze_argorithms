"""Maze canvas for local observations, remembered tiles and post-escape review."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
import tkinter as tk

from maze_env import CompletedMap, Observation, Position, Tile


COLORS = {
    "background": "#ffffff", "ink": "#172743", "muted": "#667085",
    "accent": "#009dcc", "grid": "#e6ecf2",
    "unknown": "#e0e6ed", "unknown_grid": "#d8e0e8",
    "wall": "#37495e", "wall_edge": "#28394b", "floor": "#fcfdff",
    "start": "#edf6fd",
    "key": "#b48017", "exit": "#263d5c", "open_exit": "#27835f",
    "trap": "#c34646", "trap_background": "#fff0f0", "agent": "#087eac",
    "agent_background": "#e7f6fd",
    "path": "#377fb8", "repeated_path": "#164e7c",
    "ghost": "#7651a5",
}
ASSET_PATH = Path(__file__).resolve().parent / "assets" / "syungsyung.png"


@dataclass(frozen=True)
class GridGeometry:
    rows: int
    columns: int
    cell_size: int
    x: int
    y: int

    @classmethod
    def fit(cls, width: int, height: int, shape: tuple[int, int], zoom: float = 1.0) -> GridGeometry:
        rows, columns = shape
        available_width, available_height = max(columns, width - 48), max(rows, height - 42)
        fitted = max(1, min(available_width // columns, available_height // rows))
        cell = max(1, round(fitted * zoom))
        x = 32 + max(0, (available_width - columns * cell) // 2)
        y = 26 + max(0, (available_height - rows * cell) // 2)
        return cls(rows, columns, cell, x, y)

    def bounds(self, position: Position, inset: float = 0) -> tuple[float, float, float, float]:
        row, col = position
        x, y = self.x + col * self.cell_size, self.y + row * self.cell_size
        return x + inset, y + inset, x + self.cell_size - inset, y + self.cell_size - inset

    def center(self, position: Position) -> tuple[float, float]:
        x, y, right, bottom = self.bounds(position)
        return (x + right) / 2, (y + bottom) / 2

    def sprite_bounds(self, position: Position) -> tuple[float, float, float, float]:
        return self.bounds(position, self.cell_size * 0.15)


class MascotImages:
    """Optional Pillow improves downsampling; ordinary Tk is a complete fallback."""

    def __init__(self, master: tk.Misc, path: Path = ASSET_PATH, *, smooth: bool = True):
        self.master, self.path = master, path
        self._cache: dict[int, tk.PhotoImage] = {}
        self._pil_image = None
        self._image_tk = None
        self._source = None
        if smooth:
            try:
                from PIL import Image, ImageTk

                with Image.open(path) as image:
                    self._pil_image = image.convert("RGBA")
                content = self._pil_image.getbbox()
                if content is not None:
                    self._pil_image = self._pil_image.crop(content)
                self._image_tk = ImageTk
            except (ImportError, OSError):
                self._pil_image = None
        if self._pil_image is None:
            try:
                self._source = tk.PhotoImage(master=master, file=str(path))
            except tk.TclError:
                pass

    def for_size(self, maximum: int) -> tk.PhotoImage | None:
        maximum = max(1, int(maximum))
        if maximum not in self._cache:
            if self._pil_image is not None:
                from PIL import Image

                image = self._pil_image.copy()
                image.thumbnail((maximum, maximum), Image.Resampling.LANCZOS)
                self._cache[maximum] = self._image_tk.PhotoImage(image, master=self.master)
            elif self._source is not None:
                factor = max(1, ceil(max(self._source.width(), self._source.height()) / maximum))
                self._cache[maximum] = self._source.subsample(factor, factor)
        return self._cache.get(maximum)


def draw_tile_surface(canvas: tk.Canvas, tile: Tile | None, box, *, current: bool = False, tags=()) -> None:
    if tile == Tile.WALL:
        fill, outline = COLORS["wall"], ""
    elif tile is None:
        fill, outline = COLORS["unknown"], COLORS["unknown_grid"]
    else:
        fill = {Tile.START: COLORS["start"], Tile.TRAP: COLORS["trap_background"]}.get(tile, COLORS["floor"])
        outline = COLORS["grid"]
    if current:
        fill = COLORS["agent_background"]
    canvas.create_rectangle(*box, fill=fill, outline=outline, width=1, tags=tags)


def draw_wall_outline(
    canvas: tk.Canvas, box, *, joined_sides: tuple[bool, ...] = (False, False, False, False),
) -> None:
    """Outline exposed sides in top, right, bottom, left order."""
    x, y, right, bottom = box
    sides = ((x, y, right, y), (right, y, right, bottom), (x, bottom, right, bottom), (x, y, x, bottom))
    for coordinates, joined in zip(sides, joined_sides):
        if not joined:
            canvas.create_line(*coordinates, fill=COLORS["wall_edge"], width=1, tags="wall-edge")


def draw_tile_icon(canvas: tk.Canvas, tile: Tile, box, *, has_key: bool = False, tags=()) -> None:
    x, y, right, bottom = box
    size = min(right - x, bottom - y)
    cx, cy = (x + right) / 2, (y + bottom) / 2
    stroke = max(1, round(size / 16))
    if tile == Tile.START:
        canvas.create_text(x + size * .20, y + size * .23, text="S", fill="#537da6",
                           font=("맑은 고딕", max(7, int(size * .23)), "bold"), tags=tags)
    elif tile == Tile.KEY:
        radius = size * .11
        ring_x, ring_y = cx - size * .14, cy - size * .08
        canvas.create_oval(ring_x - radius, ring_y - radius, ring_x + radius, ring_y + radius,
                           outline=COLORS["key"], width=stroke + 1, tags=tags)
        canvas.create_line(ring_x + radius * .7, ring_y + radius * .7,
                           cx + size * .20, cy + size * .22, fill=COLORS["key"], width=stroke + 1, tags=tags)
        canvas.create_line(cx + size * .08, cy + size * .13, cx + size * .16, cy + size * .05,
                           fill=COLORS["key"], width=stroke, tags=tags)
    elif tile == Tile.EXIT:
        color = COLORS["open_exit"] if has_key else COLORS["exit"]
        canvas.create_rectangle(cx - size * .20, cy - size * .29, cx + size * .20, cy + size * .29,
                                outline=color, width=stroke, tags=tags)
        if has_key:
            canvas.create_polygon(cx - size * .18, cy - size * .28, cx + size * .08, cy - size * .18,
                                  cx + size * .08, cy + size * .37, cx - size * .18, cy + size * .28,
                                  fill="#def3e8", outline=color, width=stroke, tags=tags)
        else:
            canvas.create_arc(cx - size * .08, cy - size * .09, cx + size * .08, cy + size * .10,
                               start=0, extent=180, style="arc", outline=color, width=stroke, tags=tags)
            canvas.create_rectangle(cx - size * .08, cy, cx + size * .08, cy + size * .13,
                                    fill=color, outline=color, tags=tags)
    elif tile == Tile.TRAP:
        canvas.create_polygon(cx, cy - size * .26, cx - size * .27, cy + size * .24,
                              cx + size * .27, cy + size * .24, fill=COLORS["trap_background"],
                              outline=COLORS["trap"], width=stroke, tags=tags)
        canvas.create_text(cx, cy + size * .03, text="!", fill=COLORS["trap"],
                           font=("맑은 고딕", max(7, int(size * .30)), "bold"), tags=tags)


def draw_ghost_icon(canvas: tk.Canvas, box, *, tags=()) -> None:
    x, y, right, bottom = box
    size = min(right - x, bottom - y)
    cx, cy = (x + right) / 2, (y + bottom) / 2
    left, right = cx - size * .28, cx + size * .28
    lower = cy + size * .26
    color = COLORS["ghost"]
    canvas.create_arc(left, cy - size * .28, right, cy + size * .28,
                      start=0, extent=180, style="pieslice", fill=color, outline="", tags=tags)
    canvas.create_polygon(
        left, cy - size * .02, right, cy - size * .02, right, lower,
        cx + size * .14, cy + size * .16, cx, lower, cx - size * .14, cy + size * .16, left, lower,
        fill=color, outline="", tags=tags,
    )
    radius = size * .05
    eye_y = cy - size * .06
    for eye_x in (cx - size * .10, cx + size * .10):
        canvas.create_oval(eye_x - radius, eye_y - radius, eye_x + radius, eye_y + radius,
                           fill="white", outline="", tags=tags)


class MazeCanvas(tk.Canvas):
    def __init__(self, master, images: MascotImages, **kwargs):
        super().__init__(master, background=COLORS["background"], highlightthickness=0,
                         takefocus=True, **kwargs)
        self.images = images
        self.observation: Observation | None = None
        self.completed_map: CompletedMap | None = None
        self.remembered_tiles: dict[Position, Tile] | None = None
        self._travel_edges: tuple[tuple[Position, Position], ...] = ()
        self.geometry_info: GridGeometry | None = None
        self.zoom = 1.0
        self._content_size = (1, 1)
        self._agent_image = None
        self.bind("<Configure>", self._resized)
        self.bind("<Button-1>", lambda event: self.focus_set())
        self.bind("<MouseWheel>", self._mouse_wheel)

    def _resized(self, event):
        if self.observation is not None:
            self.render(
                self.observation, self.completed_map, remembered_tiles=self.remembered_tiles,
                travel_edges=self._travel_edges,
            )

    def _mouse_wheel(self, event):
        if event.state & 0x4:
            self.event_generate("<<ZoomIn>>" if event.delta > 0 else "<<ZoomOut>>")
        else:
            amount = -1 if event.delta > 0 else 1
            if event.state & 0x1:
                self.xview_scroll(amount, "units")
            else:
                self.yview_scroll(amount, "units")
        return "break"

    def set_zoom(self, zoom: float) -> None:
        self.zoom = min(3.0, max(1.0, float(zoom)))
        if self.observation is not None:
            self.render(
                self.observation, self.completed_map, remembered_tiles=self.remembered_tiles,
                travel_edges=self._travel_edges, recenter=True,
            )

    def center_on_agent(self) -> None:
        if self.observation is None or self.geometry_info is None:
            return
        cx, cy = self.geometry_info.center(self.observation.position)
        width, height = self.winfo_width(), self.winfo_height()
        content_width, content_height = self._content_size
        left = max(0, min(cx - width / 2, content_width - width))
        top = max(0, min(cy - height / 2, content_height - height))
        self.xview_moveto(left / content_width)
        self.yview_moveto(top / content_height)

    def render(
        self, observation: Observation, completed_map: CompletedMap | None = None, *,
        remembered_tiles: dict[Position, Tile] | None = None,
        travel_edges: tuple[tuple[Position, Position], ...] = (), recenter: bool = False,
    ) -> None:
        if completed_map is not None:
            if not observation.done:
                raise ValueError("A completed map must not be displayed before escape.")
            if completed_map.shape != observation.shape:
                raise ValueError("The completed map shape must match the observation.")
            if remembered_tiles is not None:
                raise ValueError("Choose either the remembered map or the completed map.")
        elif travel_edges and remembered_tiles is None:
            raise ValueError("The travel path requires a remembered or completed map.")
        moved = self.observation is None or self.observation.position != observation.position or self.observation.shape != observation.shape
        self.observation = observation
        self.completed_map = completed_map
        self.remembered_tiles = remembered_tiles
        self._travel_edges = travel_edges
        width, height = max(100, self.winfo_width()), max(100, self.winfo_height())
        geometry = GridGeometry.fit(width, height, observation.shape, self.zoom)
        resized = self.geometry_info != geometry
        self.geometry_info = geometry
        self._content_size = (max(width, geometry.x + geometry.columns * geometry.cell_size + 16),
                              max(height, geometry.y + geometry.rows * geometry.cell_size + 16))
        self.configure(scrollregion=(0, 0, *self._content_size))
        self.delete("all")
        if completed_map is not None:
            visible = {cell.position: cell.tile for cell in completed_map.cells}
        else:
            visible = {} if remembered_tiles is None else remembered_tiles.copy()
            visible.update((cell.position, cell.tile) for cell in observation.visible)
        cell_size = geometry.cell_size
        for row in range(geometry.rows):
            for col in range(geometry.columns):
                position = row, col
                tile = visible.get(position)
                tags = (f"cell:{row}:{col}", "unknown" if tile is None else "visible", f"tile:{tile.value}" if tile is not None else "masked")
                box = geometry.bounds(position)
                draw_tile_surface(self, tile, box, current=position == observation.position, tags=tags)
                if tile not in (None, Tile.WALL, Tile.FLOOR):
                    draw_tile_icon(self, tile, box, has_key=observation.has_key,
                                   tags=("icon", f"icon:{tile.value}", f"cell:{row}:{col}"))
        for (row, col), tile in visible.items():
            if tile == Tile.WALL:
                neighbors = ((row - 1, col), (row, col + 1), (row + 1, col), (row, col - 1))
                joined_sides = tuple(visible.get(position) == Tile.WALL for position in neighbors)
                draw_wall_outline(self, geometry.bounds((row, col)), joined_sides=joined_sides)
        if travel_edges:
            self._draw_travel_path(geometry, visible, travel_edges)
            self.tag_raise("icon")
        if observation.ghost_position is not None and not observation.done:
            draw_ghost_icon(self, geometry.bounds(observation.ghost_position), tags="ghost")
        font = ("맑은 고딕", 8 if cell_size < 26 else (10 if cell_size >= 42 else 9))
        for row in range(geometry.rows):
            _, cy = geometry.center((row, 0))
            self.create_text(geometry.x - 17, cy, text=str(row), fill=COLORS["muted"], font=font, tags="coordinate")
        for col in range(geometry.columns):
            cx, _ = geometry.center((0, col))
            self.create_text(cx, geometry.y - 16, text=str(col), fill=COLORS["muted"], font=font, tags="coordinate")
        # A single, inset highlight identifies the exact logical cell, not its neighbours.
        self.create_rectangle(*geometry.bounds(observation.position, 1.5), fill="", outline=COLORS["agent"],
                              width=2, tags="agent-cell")
        x, y, right, bottom = geometry.sprite_bounds(observation.position)
        self._agent_image = self.images.for_size(max(1, int(min(right - x, bottom - y))))
        cx, cy = geometry.center(observation.position)
        if self._agent_image is not None:
            self.create_image(cx, cy, image=self._agent_image, tags="agent")
        else:
            self.create_oval(cx - cell_size * .19, cy - cell_size * .19,
                             cx + cell_size * .19, cy + cell_size * .19,
                             fill=COLORS["agent"], outline="", tags="agent")
        if self.zoom == 1.0:
            self.xview_moveto(0)
            self.yview_moveto(0)
        elif recenter or (completed_map is None and (moved or resized)):
            self.center_on_agent()

    def _draw_travel_path(
        self, geometry: GridGeometry, tiles: dict[Position, Tile], edges: tuple[tuple[Position, Position], ...],
    ) -> None:
        traversals: Counter[tuple[Position, Position]] = Counter()
        for origin, destination in edges:
            if origin not in tiles or destination not in tiles:
                continue
            first, second = sorted((origin, destination))
            traversals[first, second] += 1

        width = max(2, round(geometry.cell_size * .11))
        for (first, second), count in traversals.items():
            repeated = count > 1
            self.create_line(*geometry.center(first), *geometry.center(second),
                             fill=COLORS["repeated_path" if repeated else "path"],
                             width=width + int(repeated), capstyle=tk.ROUND, tags="travel-path")
