"""Behavior tests. Full-map search below validates levels, not a game agent."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from main import read_command, render_frame
from maze_env import (
    Action,
    AgentMemory,
    DELTAS,
    EventKind,
    MazeEnv,
    MazeLayout,
    Tile,
)


TEST_MAP = (
    "#########",
    "#S..E...#",
    "#.......#",
    "#..K.T..#",
    "#.......#",
    "#.......#",
    "#########",
)


def route(layout, start, goal, blocked=(), blocked_edge=None):
    """Independent full-information path finder for tests only."""
    blocked = set(blocked)
    queue = deque([(start, [])])
    seen = {start}
    height, width = layout.shape
    while queue:
        position, actions = queue.popleft()
        if position == goal:
            return actions
        for action, (dr, dc) in DELTAS.items():
            target = position[0] + dr, position[1] + dc
            r, c = target
            if not (0 <= r < height and 0 <= c < width):
                continue
            if (
                target in seen
                or target in blocked
                or layout.rows[r][c] == "#"
                or frozenset((position, target)) == blocked_edge
            ):
                continue
            seen.add(target)
            queue.append((target, actions + [action]))
    return None


def drive(env, memory, layout, target):
    observation = env.get_observation()
    blocked = set(layout.find_all(Tile.TRAP))
    if not observation.has_key:
        blocked.add(layout.find(Tile.EXIT))
    actions = route(layout, observation.position, target, blocked)
    if actions is None:
        raise AssertionError(f"No test route to {target}.")
    for action in actions:
        result = env.step(action)
        memory.update(result.observation, result.events)
    return env.get_observation()


def screen_grid(observation, memory):
    frame = render_frame(observation, memory, "test")
    return [
        line.split("|", 1)[1].split()
        for line in frame.splitlines()
        if len(line) > 4 and line[:2].isdigit() and line[3] == "|"
    ]


class MazeRulesTests(unittest.TestCase):
    def setUp(self):
        self.layout = MazeLayout(TEST_MAP)
        self.env = MazeEnv(self.layout)
        self.memory = AgentMemory()
        self.memory.update(self.env.get_observation())

    def step(self, action):
        result = self.env.step(action)
        self.memory.update(result.observation, result.events)
        return result

    def test_initial_observation_is_local_and_does_not_disclose_items(self):
        observation = self.env.get_observation()
        self.assertEqual(observation.position, (1, 1))
        self.assertEqual(observation.exit_position, (1, 4))
        self.assertEqual(
            {cell.position for cell in observation.visible},
            {(row, col) for row in range(3) for col in range(3)},
        )
        self.assertFalse(any(cell.tile in (Tile.KEY, Tile.TRAP) for cell in observation.visible))
        self.assertNotIn(self.layout.find(Tile.KEY), self.memory.known_tiles)
        self.assertNotIn(self.layout.find(Tile.TRAP), self.memory.known_tiles)
        self.assertEqual(self.memory.visited_positions, {(1, 1)})

    def test_hidden_placements_do_not_change_initial_observation(self):
        other = MazeLayout(
            (
                "#########",
                "#S..E...#",
                "#.......#",
                "#..T.K..#",
                "#.......#",
                "#.......#",
                "#########",
            )
        )
        self.assertEqual(self.env.get_observation(), MazeEnv(other).get_observation())

    def test_view_includes_diagonals_without_wall_occlusion(self):
        layout = MazeLayout(("#######", "#S#..E#", "##....#", "#..K.T#", "#######"))
        visible = {cell.position: cell.tile for cell in MazeEnv(layout).get_observation().visible}
        self.assertEqual(visible[(1, 2)], Tile.WALL)
        self.assertEqual(visible[(2, 1)], Tile.WALL)
        self.assertEqual(visible[(2, 2)], Tile.FLOOR)

    def test_wall_attempt_is_free_and_does_not_extend_history(self):
        result = self.step(Action.UP)
        self.assertEqual(result.events[0].kind, EventKind.WALL_BLOCKED)
        self.assertEqual(result.observation.position, (1, 1))
        self.assertEqual(result.observation.moves, 0)
        self.assertEqual(self.memory.travel_history, [(1, 1)])
        self.assertEqual(self.step(Action.RIGHT).observation.moves, 1)
        self.assertEqual(self.step(Action.LEFT).observation.moves, 2)

    def test_locked_exit_blocks_entry_without_counting_a_move(self):
        self.step(Action.RIGHT)
        self.step(Action.RIGHT)
        before = self.env.get_observation()
        history = list(self.memory.travel_history)
        result = self.step(Action.RIGHT)
        self.assertEqual(result.observation, before)
        self.assertEqual(result.events[0].kind, EventKind.EXIT_LOCKED)
        self.assertEqual(self.memory.travel_history, history)
        self.assertFalse(result.done)

    def test_key_is_picked_up_automatically_and_not_duplicated(self):
        key = self.layout.find(Tile.KEY)
        drive(self.env, self.memory, self.layout, key)
        self.assertTrue(self.env.get_observation().has_key)
        self.assertEqual(self.memory.known_tiles[key], Tile.FLOOR)
        self.step(Action.LEFT)
        result = self.step(Action.RIGHT)
        self.assertTrue(result.observation.has_key)
        self.assertNotIn(EventKind.KEY_PICKED_UP, [event.kind for event in result.events])

    def test_trap_is_hidden_until_triggered_then_remembered_after_teleport(self):
        trap = self.layout.find(Tile.TRAP)
        drive(self.env, self.memory, self.layout, (3, 4))
        visible = {cell.position: cell.tile for cell in self.env.get_observation().visible}
        self.assertEqual(visible[trap], Tile.FLOOR)
        self.assertEqual(self.memory.known_tiles[trap], Tile.FLOOR)
        before = self.env.get_observation()
        result = self.step(Action.RIGHT)
        self.assertEqual(result.observation.position, self.layout.find(Tile.START))
        self.assertEqual(result.observation.moves, before.moves + 1)
        self.assertEqual(result.observation.trap_hits, 1)
        self.assertFalse(result.done)
        self.assertIn(trap, self.memory.known_traps)
        self.assertIn(trap, self.memory.visited_positions)
        self.assertEqual(self.memory.known_tiles[trap], Tile.TRAP)
        self.assertEqual(self.memory.travel_history[-2:], [trap, self.layout.find(Tile.START)])
        self.assertFalse(any(cell.tile == Tile.TRAP for cell in result.observation.visible))

    def test_trap_without_key_does_not_disclose_key_location(self):
        # Approach from below, staying outside the key's sensing range.
        actions = [Action.DOWN] * 4 + [Action.RIGHT] * 4 + [Action.UP] * 2
        for action in actions:
            result = self.step(action)
        self.assertEqual(result.observation.trap_hits, 1)
        self.assertFalse(result.observation.has_key)
        self.assertNotIn(EventKind.KEY_RESPAWNED, [event.kind for event in result.events])
        self.assertNotIn(self.layout.find(Tile.KEY), self.memory.known_tiles)

    def test_trap_returns_to_start_with_key_memory_and_counters(self):
        key = self.layout.find(Tile.KEY)
        drive(self.env, self.memory, self.layout, key)
        drive(self.env, self.memory, self.layout, (3, 4))
        before = self.env.get_observation()
        known_before = dict(self.memory.known_tiles)
        visited_before = set(self.memory.visited_positions)
        result = self.step(Action.RIGHT)
        self.assertEqual(result.observation.position, self.layout.find(Tile.START))
        self.assertTrue(result.observation.has_key)
        self.assertEqual(result.observation.moves, before.moves + 1)
        self.assertEqual(result.observation.trap_hits, before.trap_hits + 1)
        self.assertEqual(self.memory.known_tiles[key], Tile.FLOOR)
        self.assertTrue(visited_before <= self.memory.visited_positions)
        self.assertTrue(known_before.keys() <= self.memory.known_tiles.keys())
        self.assertEqual([event.kind for event in result.events], [EventKind.MOVED, EventKind.TRAP_TRIGGERED])
        drive(self.env, self.memory, self.layout, (2, 3))
        visible = {cell.position: cell.tile for cell in self.env.get_observation().visible}
        self.assertEqual(visible[key], Tile.FLOOR)
        self.assertTrue(drive(self.env, self.memory, self.layout, self.layout.find(Tile.EXIT)).done)

    def test_revealed_trap_is_visible_locally_and_can_trigger_again(self):
        drive(self.env, self.memory, self.layout, (3, 4))
        self.step(Action.RIGHT)
        drive(self.env, self.memory, self.layout, (3, 4))
        visible = {cell.position: cell.tile for cell in self.env.get_observation().visible}
        self.assertEqual(visible[self.layout.find(Tile.TRAP)], Tile.TRAP)
        self.assertEqual(self.step(Action.RIGHT).observation.trap_hits, 2)

    def test_two_traps_are_discovered_separately_and_both_preserve_the_key(self):
        rows = list(TEST_MAP)
        rows[5] = "#..T....#"
        self.layout = MazeLayout(tuple(rows))
        self.env = MazeEnv(self.layout)
        self.memory = AgentMemory()
        self.memory.update(self.env.get_observation())
        start, key = (self.layout.find(tile) for tile in (Tile.START, Tile.KEY))
        drive(self.env, self.memory, self.layout, key)

        approaches = (((3, 4), (3, 5)), ((5, 2), (5, 3)))
        for count, (approach, trap) in enumerate(approaches, 1):
            before = drive(self.env, self.memory, self.layout, approach)
            visible = {cell.position: cell.tile for cell in before.visible}
            self.assertEqual(visible[trap], Tile.FLOOR)
            self.assertNotIn(trap, self.memory.known_traps)
            result = self.step(Action.RIGHT)
            self.assertEqual(result.observation.position, start)
            self.assertEqual(result.observation.moves, before.moves + 1)
            self.assertEqual(result.observation.trap_hits, count)
            self.assertTrue(result.observation.has_key)
            self.assertIn(trap, self.memory.known_traps)
        self.assertEqual(self.memory.known_traps, set(self.layout.find_all(Tile.TRAP)))

    def test_success_requires_key_and_stops_further_movement(self):
        drive(self.env, self.memory, self.layout, self.layout.find(Tile.KEY))
        drive(self.env, self.memory, self.layout, self.layout.find(Tile.EXIT))
        finished = self.env.get_observation()
        self.assertTrue(finished.done)
        self.assertTrue(finished.has_key)
        result = self.step(Action.LEFT)
        self.assertEqual(result.observation, finished)
        self.assertEqual(result.events[0].kind, EventKind.ALREADY_FINISHED)

    def test_completed_map_is_escape_only_and_keeps_sensing_local(self):
        with self.assertRaises(RuntimeError):
            self.env.get_completed_map()
        drive(self.env, self.memory, self.layout, self.layout.find(Tile.KEY))
        with self.assertRaises(RuntimeError):
            self.env.get_completed_map()
        drive(self.env, self.memory, self.layout, self.layout.find(Tile.EXIT))
        observation = self.env.get_observation()
        memory_before = dict(self.memory.known_tiles)
        completed = self.env.get_completed_map()
        tiles = {cell.position: cell.tile for cell in completed.cells}
        self.assertEqual(completed.shape, self.layout.shape)
        self.assertEqual(len(tiles), self.layout.shape[0] * self.layout.shape[1])
        self.assertEqual(tiles[self.layout.find(Tile.KEY)], Tile.KEY)
        self.assertEqual(tiles[self.layout.find(Tile.TRAP)], Tile.TRAP)
        self.assertEqual(self.env.get_observation(), observation)
        self.assertEqual(len(observation.visible), 9)
        self.assertEqual(self.memory.known_tiles, memory_before)
        self.env.reset()
        with self.assertRaises(RuntimeError):
            self.env.get_completed_map()

    def test_new_experiment_resets_state_memory_and_trap_discovery(self):
        initial = self.env.reset()
        drive(self.env, self.memory, self.layout, self.layout.find(Tile.KEY))
        drive(self.env, self.memory, self.layout, (3, 4))
        self.step(Action.RIGHT)
        observation = self.env.reset()
        self.memory.clear()
        self.memory.update(observation)
        self.assertEqual(observation, initial)
        self.assertEqual(self.memory.known_traps, set())
        self.assertEqual(self.memory.visited_positions, {initial.position})
        self.assertEqual(self.memory.travel_history, [initial.position])
        self.assertNotIn(self.layout.find(Tile.KEY), self.memory.known_tiles)
        drive(self.env, self.memory, self.layout, (3, 4))
        visible = {cell.position: cell.tile for cell in self.env.get_observation().visible}
        self.assertEqual(visible[self.layout.find(Tile.TRAP)], Tile.FLOOR)

    def test_screen_shows_only_current_view_despite_retaining_memory(self):
        drive(self.env, self.memory, self.layout, (3, 4))
        observation = self.env.get_observation()
        visible = {cell.position for cell in observation.visible}
        grid = screen_grid(observation, self.memory)
        self.assertEqual(len(grid), self.layout.shape[0])
        self.assertIn((1, 1), self.memory.known_tiles)
        self.assertNotIn((1, 1), visible)
        self.assertEqual(grid[1][1], "?")
        for row in range(self.layout.shape[0]):
            for col in range(self.layout.shape[1]):
                if (row, col) not in visible:
                    self.assertEqual(grid[row][col], "?")
        self.assertEqual(grid[3][4], "@")
        self.assertEqual(grid[3][5], ".")  # Untriggered trap is hidden.

    def test_invalid_action_does_not_mutate_environment(self):
        before = self.env.get_observation()
        with self.assertRaises(ValueError):
            self.env.step("teleport")
        self.assertEqual(self.env.get_observation(), before)


class DefaultLevelTests(unittest.TestCase):
    def test_default_level_has_safe_routes_and_alternatives(self):
        layout = MazeLayout()
        self.assertEqual(layout.shape, (15, 15))
        start, key, exit_, trap = (
            layout.find(tile) for tile in (Tile.START, Tile.KEY, Tile.EXIT, Tile.TRAP)
        )
        for origin, target, blocked in (
            (start, key, {trap, exit_}),
            (key, exit_, {trap}),
        ):
            with self.subTest(origin=origin, target=target):
                actions = route(layout, origin, target, blocked)
                self.assertIsNotNone(actions, "The default level must be solvable without a trap.")
                # Removing each edge of one route must still leave another route.
                position = origin
                for action in actions:
                    dr, dc = DELTAS[action]
                    next_position = position[0] + dr, position[1] + dc
                    edge = frozenset((position, next_position))
                    self.assertIsNotNone(
                        route(layout, origin, target, blocked, blocked_edge=edge),
                        f"No alternative around edge {sorted(edge)}.",
                    )
                    position = next_position

    def test_default_level_can_be_played_to_success(self):
        layout = MazeLayout()
        env = MazeEnv(layout)
        memory = AgentMemory()
        memory.update(env.get_observation())
        drive(env, memory, layout, layout.find(Tile.KEY))
        final = drive(env, memory, layout, layout.find(Tile.EXIT))
        self.assertTrue(final.done)
        self.assertTrue(final.has_key)
        self.assertEqual(final.trap_hits, 0)
        self.assertGreater(final.moves, 0)

    def test_invalid_level_is_rejected(self):
        with self.assertRaises(ValueError):
            MazeLayout(("###", "#S#", "###"))
        with self.assertRaises(ValueError):
            MazeLayout(("#######", "#SEKT#", "#######"))


class ManualInputTests(unittest.TestCase):
    def test_windows_arrows_and_uppercase_wasd(self):
        cases = (
            (["\xe0", "H"], "w"),
            (["\xe0", "P"], "s"),
            (["\x00", "K"], "a"),
            (["\x00", "M"], "d"),
            (["W"], "w"),
            (["R"], "r"),
            (["Q"], "q"),
            (["\r"], ""),
            ([" "], " "),
        )
        for sequence, expected in cases:
            with self.subTest(sequence=sequence):
                keys = iter(sequence)
                fake_msvcrt = SimpleNamespace(getwch=lambda: next(keys))
                with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
                    self.assertEqual(read_command(immediate=True), expected)

    def test_ctrl_c_and_end_of_input_exit_cleanly(self):
        fake_msvcrt = SimpleNamespace(getwch=lambda: "\x03")
        with patch.dict(sys.modules, {"msvcrt": fake_msvcrt}):
            with self.assertRaises(KeyboardInterrupt):
                read_command(immediate=True)
        with patch("builtins.input", side_effect=EOFError):
            self.assertEqual(read_command(immediate=False), "q")

    def test_terminal_line_input_and_reset(self):
        root = Path(__file__).resolve().parents[1]
        temp_root = (root / "tmp" / "test-runs").resolve()
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="maze-input-test-", dir=temp_root) as log_dir:
            self.assertTrue(Path(log_dir).resolve().is_relative_to(temp_root))
            completed = subprocess.run(
                [sys.executable, "-X", "utf8", "main.py", "--line-input", "--no-clear", "--log-dir", log_dir],
                cwd=root,
                input="a\nd\nr\nq\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            )
            self.assertEqual(len(list(Path(log_dir).glob("*.json"))), 2)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("벽에 막혔습니다", completed.stdout)
        self.assertIn("누적 이동 1회", completed.stdout)
        self.assertIn("새 실험입니다", completed.stdout)
        self.assertIn("종료 기록: 이동 0회, 함정 0회.", completed.stdout)


if __name__ == "__main__":
    unittest.main()
