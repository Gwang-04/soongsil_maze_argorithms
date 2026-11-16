"""Search policies using public observations and remembered cells only.

DFS is online depth-first exploration with recovery after a trap. BFS and A*
share nearest-observation-boundary exploration; A* changes routing to a known
key/exit only. Every action is replanned, so no stale route survives a reset.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from time import perf_counter
from types import MappingProxyType
from typing import Callable, Mapping

from maze_env import Action, AgentMemory, DELTAS, EventKind, Observation, Position, StepResult, Tile


ACTION_ORDER = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)
ALGORITHM_LABELS = {"manual": "수동", "dfs": "온라인 DFS", "bfs": "BFS 탐색", "astar": "BFS 탐색 + A* 경로"}
PHASE_LABELS = {
    "key": "열쇠로 이동", "exit": "출구로 이동", "explore": "새 시야 탐색",
    "dfs_advance": "DFS 전진", "dfs_backtrack": "DFS 되돌아가기",
    "dfs_recover": "남은 갈림길로 복귀", "exhausted": "탐색할 경로 없음",
}
SearchState = tuple[int, int, bool]


@dataclass(frozen=True)
class Knowledge:
    """Immutable snapshot; contains no MazeEnv or MazeLayout reference."""

    shape: tuple[int, int]
    tiles: Mapping[Position, Tile]
    traps: frozenset[Position]

    @classmethod
    def from_public(cls, observation: Observation, memory: AgentMemory) -> Knowledge:
        return cls(observation.shape, MappingProxyType(dict(memory.known_tiles)), frozenset(memory.known_traps))

    def successors(self, state: SearchState):
        row, col, has_key = state
        for action in ACTION_ORDER:
            dr, dc = DELTAS[action]
            target = row + dr, col + dc
            tile = self.tiles.get(target)
            if tile is None or tile in (Tile.WALL, Tile.TRAP) or target in self.traps:
                continue
            if tile == Tile.EXIT and not has_key:
                continue
            yield action, (target[0], target[1], has_key or tile == Tile.KEY)

    def reveals_unknown(self, position: Position) -> bool:
        height, width = self.shape
        row, col = position
        return any(
            (r, c) not in self.tiles
            for r in range(max(0, row - 1), min(height, row + 2))
            for c in range(max(0, col - 1), min(width, col + 2))
        )


@dataclass(frozen=True)
class Route:
    actions: tuple[Action, ...] | None
    target: Position | None
    expanded_nodes: int
    peak_frontier: int
    seconds: float


def _reconstruct(parents, state: SearchState) -> tuple[Action, ...]:
    actions = []
    while parents[state] is not None:
        state, action = parents[state]
        actions.append(action)
    return tuple(reversed(actions))


def bfs_route(
    knowledge: Knowledge, start: SearchState, *, goal: Position | None = None,
    predicate: Callable[[SearchState], bool] | None = None,
) -> Route:
    """Shortest known route. A predicate also supports nearest exploration goals."""
    if (goal is None) == (predicate is None):
        raise ValueError("Supply exactly one goal or predicate.")
    began = perf_counter()
    is_goal = predicate if predicate is not None else lambda state: state[:2] == goal
    queue = deque([start])
    parents = {start: None}
    expanded, peak = 0, 1
    while queue:
        state = queue.popleft()
        if is_goal(state):
            return Route(_reconstruct(parents, state), state[:2], expanded, peak, perf_counter() - began)
        expanded += 1
        for action, child in knowledge.successors(state):
            if child not in parents:
                parents[child] = state, action
                queue.append(child)
        peak = max(peak, len(queue))
    return Route(None, None, expanded, peak, perf_counter() - began)


def astar_route(knowledge: Knowledge, start: SearchState, goal: Position) -> Route:
    """Unit-cost A* with Manhattan distance; ties prefer lower h, then insertion."""
    began = perf_counter()
    heuristic = lambda state: abs(state[0] - goal[0]) + abs(state[1] - goal[1])
    sequence = count()
    frontier = [(heuristic(start), heuristic(start), next(sequence), 0, start)]
    parents = {start: None}
    best = {start: 0}
    expanded, peak = 0, 1
    while frontier:
        _, _, _, cost, state = heappop(frontier)
        if cost != best.get(state):
            continue
        if state[:2] == goal:
            return Route(_reconstruct(parents, state), state[:2], expanded, peak, perf_counter() - began)
        expanded += 1
        for action, child in knowledge.successors(state):
            next_cost = cost + 1
            if next_cost < best.get(child, float("inf")):
                best[child] = next_cost
                parents[child] = state, action
                h = heuristic(child)
                heappush(frontier, (next_cost + h, h, next(sequence), next_cost, child))
        peak = max(peak, len(frontier))
    return Route(None, None, expanded, peak, perf_counter() - began)


@dataclass
class SearchStats:
    search_calls: int = 0
    expanded_nodes: int = 0
    route_expanded_nodes: int = 0
    exploration_expanded_nodes: int = 0
    peak_frontier: int = 0
    path_planning_seconds: float = 0.0

    def add(self, route: Route, *, routing: bool = False) -> None:
        self.search_calls += 1
        self.expanded_nodes += route.expanded_nodes
        if routing:
            self.route_expanded_nodes += route.expanded_nodes
        else:
            self.exploration_expanded_nodes += route.expanded_nodes
        self.peak_frontier = max(self.peak_frontier, route.peak_frontier)
        self.path_planning_seconds += route.seconds


@dataclass(frozen=True)
class Decision:
    action: Action | None
    phase: str
    target: Position | None
    stats: SearchStats = field(default_factory=SearchStats)


@dataclass(frozen=True)
class RoutingTask:
    knowledge: Knowledge
    start: SearchState
    goal: Position
    phase: str


class SearchAgent:
    def __init__(self, algorithm: str, *, capture_tasks: bool = False) -> None:
        if algorithm not in ("dfs", "bfs", "astar"):
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm = algorithm
        self.capture_tasks = capture_tasks
        self.route_tasks: list[RoutingTask] = []
        self._trail: list[Position] = []
        self._recovery_target: Position | None = None

    def observe(self, result: StepResult) -> None:
        """Keep only a connected DFS trail; spatial exploration memory is external."""
        position = result.observation.position
        kinds = {event.kind for event in result.events}
        if EventKind.TRAP_TRIGGERED in kinds:
            self._trail = [position]
            self._recovery_target = None
        elif EventKind.MOVED in kinds:
            if position in self._trail:
                self._trail = self._trail[:self._trail.index(position) + 1]
            else:
                self._trail.append(position)

    def choose(self, observation: Observation, memory: AgentMemory) -> Decision:
        knowledge = Knowledge.from_public(observation, memory)
        start = (*observation.position, observation.has_key)
        stats = SearchStats()
        if observation.has_key:
            goal, phase = observation.exit_position, "exit"
        else:
            keys = sorted(position for position, tile in knowledge.tiles.items() if tile == Tile.KEY)
            goal, phase = (keys[0], "key") if keys else (None, "explore")

        if goal is not None:
            if self.capture_tasks:
                self.route_tasks.append(RoutingTask(knowledge, start, goal, phase))
            route = astar_route(knowledge, start, goal) if self.algorithm == "astar" else bfs_route(knowledge, start, goal=goal)
            stats.add(route, routing=True)
            if route.actions:
                return Decision(route.actions[0], phase, goal, stats)

        if self.algorithm == "dfs":
            return self._depth_first(observation, memory, knowledge, start, stats)

        route = bfs_route(knowledge, start, predicate=lambda state: knowledge.reveals_unknown(state[:2]))
        stats.add(route)
        if route.actions:
            return Decision(route.actions[0], "explore", route.target, stats)
        return Decision(None, "exhausted", None, stats)

    def _depth_first(self, observation, memory, knowledge, start, stats) -> Decision:
        position = observation.position
        if not self._trail or self._trail[-1] != position:
            self._trail = [position]
        if self._recovery_target is not None:
            if position != self._recovery_target:
                route = bfs_route(knowledge, start, goal=self._recovery_target)
                stats.add(route)
                if route.actions:
                    return Decision(route.actions[0], "dfs_recover", self._recovery_target, stats)
            self._recovery_target = None
            self._trail = [position]
        # This is one local expansion, including on a backtracking decision.
        stats.expanded_nodes += 1
        stats.exploration_expanded_nodes += 1
        stats.peak_frontier = max(stats.peak_frontier, len(self._trail))
        successors = list(knowledge.successors(start))
        for action, child in successors:
            if child[:2] not in memory.visited_positions:
                return Decision(action, "dfs_advance", child[:2], stats)
        if len(self._trail) > 1:
            parent = self._trail[-2]
            for action, child in successors:
                if child[:2] == parent:
                    return Decision(action, "dfs_backtrack", parent, stats)

        # Trap returns discard the old physical trail, not unexplored branches.
        # Relocate through known cells to a visited junction with an unvisited edge.
        def pending(state):
            return state[:2] in memory.visited_positions and any(
                child[:2] not in memory.visited_positions
                for _, child in knowledge.successors(state)
            )

        route = bfs_route(knowledge, start, predicate=pending)
        stats.add(route)
        if route.actions:
            self._recovery_target = route.target
            return Decision(route.actions[0], "dfs_recover", route.target, stats)
        return Decision(None, "exhausted", None, stats)


def agent_settings(algorithm: str, max_actions: int) -> dict:
    return {
        "algorithm": algorithm,
        "exploration": "online_dfs_with_known_route_recovery" if algorithm == "dfs" else "nearest_observation_boundary_bfs",
        "known_goal_routing": "astar" if algorithm == "astar" else "bfs",
        "neighbor_order": [action.value for action in ACTION_ORDER],
        "astar_ties": "lower_h_then_insertion_order" if algorithm == "astar" else None,
        "heuristic": "Manhattan distance to the selected known goal" if algorithm == "astar" else None,
        "timing_method": "perf_counter elapsed time inside agent/planner; not process CPU time",
        "max_action_attempts": max_actions,
        "replan": "before_each_action",
        "unknown_cells": "not traversable in the planning model",
        "known_traps": "blocked",
        "expanded_nodes_definition": "states whose successors are examined; repeated planning counts again; goal-only pops excluded",
        "comparison_note": "DFS/BFS compare exploration; A* shares BFS exploration and changes known-goal routing only.",
    }
