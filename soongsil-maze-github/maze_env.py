"""Partially observable key-and-exit maze, independent of terminal input.

Coordinates are (row, column), starting at zero. This module contains no
automatic agent. The environment owns the real map; an agent receives only
Observation and StepResult, and can maintain its own AgentMemory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


Position = tuple[int, int]


class Tile(str, Enum):
    WALL = "#"
    FLOOR = "."
    START = "S"
    EXIT = "E"
    KEY = "K"
    TRAP = "T"


class Action(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    WAIT = "wait"


DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


# One fixed, 15 x 15 level, including the outer wall. Several loops allow
# alternative routes. S and E are fixed; K is never disclosed by reset().
DEFAULT_MAP = (
    "###############",
    "#S............#",
    "#.###.###.###.#",
    "#...#.....#...#",
    "#.#.#.###.#.#.#",
    "#.#...#...#...#",
    "#.###.#.###.#.#",
    "#.....#T......#",
    "#.###.#####.#.#",
    "#...#..K..#...#",
    "#.#.###.#.#.#.#",
    "#.#.....#.....#",
    "#.###.###.###.#",
    "#............E#",
    "###############",
)


@dataclass(frozen=True)
class MazeLayout:
    """Fixed level with one key, one start, one exit and optional traps."""

    rows: tuple[str, ...] = DEFAULT_MAP

    def __post_init__(self) -> None:
        # Accept a list too, but retain an immutable copy.
        object.__setattr__(self, "rows", tuple(self.rows))
        if len(self.rows) < 3 or len(self.rows[0]) < 3:
            raise ValueError("A maze must be at least 3 x 3.")
        width = len(self.rows[0])
        if any(len(row) != width for row in self.rows):
            raise ValueError("All maze rows must have the same width.")
        allowed = {tile.value for tile in Tile}
        if any(char not in allowed for row in self.rows for char in row):
            raise ValueError("Map symbols must be # . S E K T.")
        if (
            any(char != "#" for char in self.rows[0] + self.rows[-1])
            or any(row[0] != "#" or row[-1] != "#" for row in self.rows)
        ):
            raise ValueError("The outside border must consist of walls.")
        for marker in ("S", "E", "K"):
            if sum(row.count(marker) for row in self.rows) != 1:
                raise ValueError(f"The maze must contain exactly one {marker}.")

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    def find(self, tile: Tile) -> Position:
        """Return the first matching position in row order."""
        positions = self.find_all(tile)
        if positions:
            return positions[0]
        raise ValueError(f"Tile {tile.value} was not found.")

    def find_all(self, tile: Tile) -> tuple[Position, ...]:
        marker = tile.value
        return tuple(
            (row, col) for row, line in enumerate(self.rows)
            for col, cell in enumerate(line) if cell == marker
        )


@dataclass(frozen=True)
class VisibleCell:
    position: Position
    tile: Tile


@dataclass(frozen=True)
class Observation:
    """Public information; never includes the full map or an unseen key."""

    shape: tuple[int, int]
    position: Position
    start_position: Position
    exit_position: Position
    has_key: bool
    visible: tuple[VisibleCell, ...]
    moves: int
    trap_hits: int
    done: bool
    turns: int = 0
    ghost_hits: int = 0
    ghost_position: Position | None = None


@dataclass(frozen=True)
class CompletedMap:
    """Initial terrain for post-escape review; never an agent observation."""

    shape: tuple[int, int]
    cells: tuple[VisibleCell, ...]


class EventKind(str, Enum):
    MOVED = "moved"
    WAITED = "waited"
    WALL_BLOCKED = "wall_blocked"
    EXIT_LOCKED = "exit_locked"
    KEY_PICKED_UP = "key_picked_up"
    TRAP_TRIGGERED = "trap_triggered"
    GHOST_COLLISION = "ghost_collision"
    KEY_RESPAWNED = "key_respawned"  # Recorded by earlier versions.
    ESCAPED = "escaped"
    ALREADY_FINISHED = "already_finished"


@dataclass(frozen=True)
class Event:
    kind: EventKind
    position: Position


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    events: tuple[Event, ...]

    @property
    def done(self) -> bool:
        return self.observation.done


class MazeEnv:
    """Deterministic movement rules with a radius-one, square observation.

    A step into a trap counts as one move. Teleporting back to the start is
    automatic and costs no additional move. Wall/locked-exit attempts cost
    zero moves. A trap return is not the end of an experiment.
    """

    def __init__(
        self, layout: MazeLayout | None = None, *, ghost_patrol: tuple[Position, ...] = (),
    ) -> None:
        self._layout = layout or MazeLayout()
        self._start = self._layout.find(Tile.START)
        self._exit = self._layout.find(Tile.EXIT)
        self._ghost_patrol = tuple(ghost_patrol)
        height, width = self._layout.shape
        for index, (row, col) in enumerate(self._ghost_patrol):
            if not (0 <= row < height and 0 <= col < width) or self._layout.rows[row][col] != Tile.FLOOR.value:
                raise ValueError("The ghost patrol must use floor tiles.")
            previous_row, previous_col = self._ghost_patrol[index - 1]
            if abs(row - previous_row) + abs(col - previous_col) != 1:
                raise ValueError("The ghost patrol must move one cell per turn, including its return.")
        self.reset()

    def reset(self) -> Observation:
        """Start a new experiment, including hiding all traps again."""
        self._position = self._start
        self._has_key = False
        self._revealed_traps: set[Position] = set()
        self._moves = 0
        self._trap_hits = 0
        self._turns = 0
        self._ghost_hits = 0
        self._done = False
        return self.get_observation()

    def _ghost_position(self) -> Position | None:
        if self._ghost_patrol:
            return self._ghost_patrol[self._turns % len(self._ghost_patrol)]
        return None

    def _perceived_tile(self, position: Position) -> Tile:
        row, col = position
        tile = Tile(self._layout.rows[row][col])
        if tile == Tile.KEY and self._has_key:
            return Tile.FLOOR
        if tile == Tile.TRAP and position not in self._revealed_traps:
            return Tile.FLOOR
        return tile

    def get_observation(self) -> Observation:
        """Return the current 3 x 3 view; walls do not occlude this square."""
        height, width = self._layout.shape
        row, col = self._position
        visible = tuple(
            VisibleCell((r, c), self._perceived_tile((r, c)))
            for r in range(max(0, row - 1), min(height, row + 2))
            for c in range(max(0, col - 1), min(width, col + 2))
        )
        ghost = self._ghost_position()
        if ghost is not None and not any(cell.position == ghost for cell in visible):
            ghost = None
        return Observation(
            shape=self._layout.shape,
            position=self._position,
            start_position=self._start,
            exit_position=self._exit,
            has_key=self._has_key,
            visible=visible,
            moves=self._moves,
            trap_hits=self._trap_hits,
            done=self._done,
            turns=self._turns,
            ghost_hits=self._ghost_hits,
            ghost_position=ghost,
        )

    def get_completed_map(self) -> CompletedMap:
        """Reveal original key/trap positions only after a successful escape.

        Reading the review does not change local sensing, memory or counters.
        The key marks its original placement, even though it has been picked up.
        """
        if not self._done:
            raise RuntimeError("The full map is available only after a successful escape.")
        return CompletedMap(
            self._layout.shape,
            tuple(VisibleCell((row, col), Tile(tile))
                  for row, line in enumerate(self._layout.rows)
                  for col, tile in enumerate(line)),
        )

    def step(self, action: Action | str) -> StepResult:
        """Move or wait one turn and return local information plus events.

        A trap reveals the cell just entered and returns the agent to the
        start while preserving key possession.
        """
        action = Action(action)
        if self._done:
            return StepResult(
                self.get_observation(),
                (Event(EventKind.ALREADY_FINISHED, self._position),),
            )

        origin = self._position
        waiting = action == Action.WAIT
        if waiting:
            if not self._ghost_patrol:
                raise ValueError("Waiting is available in ghost mode.")
            target = origin
        else:
            dr, dc = DELTAS[action]
            target = origin[0] + dr, origin[1] + dc
        height, width = self._layout.shape
        r, c = target
        if not (0 <= r < height and 0 <= c < width):
            return StepResult(
                self.get_observation(), (Event(EventKind.WALL_BLOCKED, target),)
            )
        tile = Tile(self._layout.rows[r][c])
        if tile == Tile.WALL:
            return StepResult(
                self.get_observation(), (Event(EventKind.WALL_BLOCKED, target),)
            )
        if tile == Tile.EXIT and not self._has_key:
            return StepResult(
                self.get_observation(), (Event(EventKind.EXIT_LOCKED, target),)
            )

        ghost_before = self._ghost_position()
        self._turns += 1
        ghost_after = self._ghost_position()
        if waiting:
            events = [Event(EventKind.WAITED, target)]
        else:
            self._position = target
            self._moves += 1
            events = [Event(EventKind.MOVED, target)]
        if tile == Tile.TRAP:
            self._revealed_traps.add(target)
            self._trap_hits += 1
            events.append(Event(EventKind.TRAP_TRIGGERED, target))
            self._position = self._start
        elif ghost_after is not None and (
            target == ghost_after or (target == ghost_before and origin == ghost_after)
        ):
            self._ghost_hits += 1
            events.append(Event(EventKind.GHOST_COLLISION, target))
            self._position = self._start
        elif tile == Tile.KEY and not self._has_key:
            self._has_key = True
            events.append(Event(EventKind.KEY_PICKED_UP, target))
        elif tile == Tile.EXIT:
            self._done = True
            events.append(Event(EventKind.ESCAPED, target))

        return StepResult(self.get_observation(), tuple(events))


@dataclass
class AgentMemory:
    """Knowledge built only from observations/events, separate from the world.

    A visible floor is not necessarily safe: it can hide an untriggered trap.
    visited_positions includes the entered trap cell, and travel_history also
    records the automatic return to the start (without counting it as a move).
    """

    known_tiles: dict[Position, Tile] = field(default_factory=dict)
    visited_positions: set[Position] = field(default_factory=set)
    known_traps: set[Position] = field(default_factory=set)
    travel_history: list[Position] = field(default_factory=list)
    observed_positions: set[Position] = field(default_factory=set)

    def clear(self) -> None:
        self.known_tiles.clear()
        self.visited_positions.clear()
        self.known_traps.clear()
        self.travel_history.clear()
        self.observed_positions.clear()

    def observed_tiles(self) -> dict[Position, Tile]:
        """Tiles seen in local observations or revealed by a trap encounter."""
        return {
            position: tile for position, tile in self.known_tiles.items()
            if position in self.observed_positions
        }

    def update(
        self, observation: Observation, events: Iterable[Event] = ()
    ) -> None:
        # Exit and start coordinates are supplied at the beginning by rule.
        self.known_tiles[observation.start_position] = Tile.START
        self.known_tiles[observation.exit_position] = Tile.EXIT
        for cell in observation.visible:
            self.observed_positions.add(cell.position)
            self.known_tiles[cell.position] = cell.tile
            if cell.tile == Tile.TRAP:
                self.known_traps.add(cell.position)

        for event in events:
            if event.kind == EventKind.MOVED:
                self.visited_positions.add(event.position)
                self.travel_history.append(event.position)
            elif event.kind == EventKind.TRAP_TRIGGERED:
                self.observed_positions.add(event.position)
                self.known_traps.add(event.position)
                self.known_tiles[event.position] = Tile.TRAP
            elif event.kind == EventKind.KEY_PICKED_UP:
                self.known_tiles[event.position] = Tile.FLOOR
            elif event.kind == EventKind.KEY_RESPAWNED:
                self.known_tiles[event.position] = Tile.KEY

        self.visited_positions.add(observation.position)
        if not self.travel_history or self.travel_history[-1] != observation.position:
            self.travel_history.append(observation.position)

