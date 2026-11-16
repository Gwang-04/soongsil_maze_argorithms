"""Solvability, scenario switching and recording on the fixed maps."""

from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiment_log import EndReason
from levels import BASIC_KEY_PLACEMENTS, MAPS
from main import ManualGame, render_frame
from maze_env import Action, DELTAS, MazeEnv, MazeLayout, Tile


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = (ROOT / "tmp" / "test-runs").resolve()


def path_to(layout, start, goal, blocked):
    queue = deque([(start, [])])
    seen = {start}
    height, width = layout.shape
    while queue:
        position, path = queue.popleft()
        if position == goal:
            return path
        for action, (dr, dc) in DELTAS.items():
            target = position[0] + dr, position[1] + dc
            row, col = target
            if (
                0 <= row < height and 0 <= col < width and target not in seen
                and target not in blocked and layout.rows[row][col] != "#"
            ):
                seen.add(target)
                queue.append((target, path + [action]))
    return None


class FixedMapTests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="maze-level-test-", dir=TEMP_ROOT)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(TEMP_ROOT))
        self.temporary.cleanup()

    def test_map_ids_sizes_and_initial_hidden_items(self):
        self.assertEqual(set(MAPS), {"basic15", "branch15", "hard21", "large41"})
        for map_id, definition in MAPS.items():
            with self.subTest(map_id=map_id):
                expected = {"hard21": 21, "large41": 41}.get(map_id, 15)
                self.assertEqual(definition.layout.shape, (expected, expected))
                self.assertEqual(definition.layout.find(Tile.START), (1, 1))
                self.assertEqual(definition.layout.find(Tile.EXIT), (expected - 2, expected - 2))
                self.assertEqual(len(definition.layout.find_all(Tile.TRAP)), 2 if map_id in {"hard21", "large41"} else 1)
                game = ManualGame(definition.layout, log_dir=self.directory)
                self.assertFalse(any(cell.tile in {Tile.KEY, Tile.TRAP} for cell in game.observation.visible))

    def test_every_map_has_trap_free_key_and_exit_routes_and_cycles(self):
        for map_id, definition in MAPS.items():
            layout = definition.layout
            with self.subTest(map_id=map_id):
                start, key, exit_ = (layout.find(tile) for tile in (Tile.START, Tile.KEY, Tile.EXIT))
                traps = set(layout.find_all(Tile.TRAP))
                self.assertIsNotNone(path_to(layout, start, key, traps | {exit_}))
                self.assertIsNotNone(path_to(layout, key, exit_, traps))
                vertices = {
                    (row, col) for row, line in enumerate(layout.rows)
                    for col, tile in enumerate(line) if tile != "#" and (row, col) not in traps
                }
                edges = sum(
                    (row + dr, col + dc) in vertices
                    for row, col in vertices for dr, dc in DELTAS.values()
                ) // 2
                self.assertGreaterEqual(edges - len(vertices) + 1, 1, "A safe loop must remain.")

    def test_key_placement_scenarios_only_move_the_hidden_key(self):
        base = MAPS["basic15"].layout
        terrain = tuple(row.replace("K", ".") for row in base.rows)
        initial = MazeEnv(base).get_observation()
        start, exit_, trap = (base.find(tile) for tile in (Tile.START, Tile.EXIT, Tile.TRAP))
        keys = set()
        for definition in BASIC_KEY_PLACEMENTS:
            with self.subTest(map_id=definition.map_id):
                layout = definition.layout
                key = layout.find(Tile.KEY)
                keys.add(key)
                self.assertEqual(tuple(row.replace("K", ".") for row in layout.rows), terrain)
                self.assertEqual(MazeEnv(layout).get_observation(), initial)
                first = path_to(layout, start, key, {trap, exit_})
                second = path_to(layout, key, exit_, {trap})
                self.assertIsNotNone(first)
                self.assertIsNotNone(second)
                self.assertEqual(len(first) + len(second), 24)
        self.assertEqual(len(keys), len(BASIC_KEY_PLACEMENTS))
        self.assertEqual(len(keys), 3)

    def test_all_maps_play_to_success_and_write_matching_map_record(self):
        lengths = {}
        for map_id, definition in MAPS.items():
            layout = definition.layout
            game = ManualGame(
                layout, log_dir=self.directory, map_id=map_id, map_name=definition.name,
            )
            key, exit_ = (layout.find(tile) for tile in (Tile.KEY, Tile.EXIT))
            traps = set(layout.find_all(Tile.TRAP))
            first = path_to(layout, game.observation.position, key, traps | {exit_})
            second = path_to(layout, key, exit_, traps)
            for action in first + second:
                game.move(action)
            self.assertTrue(game.observation.done)
            self.assertEqual(game.observation.trap_hits, 0)
            saved = game.last_saved
            data = json.loads(saved.json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["map"]["id"], map_id)
            self.assertEqual(data["map"]["shape"], list(layout.shape))
            self.assertEqual(data["summary"]["moves"], len(first) + len(second))
            lengths[map_id] = data["summary"]["moves"]
        self.assertGreater(lengths["hard21"], lengths["basic15"])
        self.assertGreater(lengths["hard21"], lengths["branch15"])
        self.assertEqual(len(list(self.directory.glob("*.json"))), len(MAPS))

    def test_map_switch_saves_previous_state_before_initializing_next(self):
        fixture = MazeLayout(("#######", "#SK..E#", "#..T..#", "#######"))
        game = ManualGame(fixture, log_dir=self.directory, map_id="fixture", map_name="테스트")
        game.move(Action.RIGHT)
        self.assertTrue(game.observation.has_key)
        selected = MAPS["hard21"]
        saved = game.change_map(selected.layout, selected.map_id, selected.name)
        previous = json.loads(saved.json_path.read_text(encoding="utf-8"))
        self.assertEqual(previous["map"]["id"], "fixture")
        self.assertEqual(previous["termination"]["reason"], "map_changed")
        self.assertTrue(previous["final_observation"]["has_key"])
        self.assertEqual(game.map_id, "hard21")
        self.assertEqual(game.observation.shape, (21, 21))
        self.assertEqual(game.observation.moves, 0)
        self.assertFalse(game.observation.has_key)
        self.assertEqual(game.memory.travel_history, [(1, 1)])
        self.assertEqual(len(game.memory.known_traps), 0)
        current = game.finish(EndReason.QUIT)
        self.assertEqual(json.loads(current.json_path.read_text(encoding="utf-8"))["map"]["id"], "hard21")

    def test_hard_map_renderer_contains_twenty_one_cells_per_row(self):
        definition = MAPS["hard21"]
        game = ManualGame(definition.layout, log_dir=self.directory)
        frame = render_frame(game.observation, game.memory, "test", definition.name)
        rows = [line.split("|", 1)[1].split() for line in frame.splitlines() if len(line) > 4 and line[:2].isdigit() and line[3] == "|"]
        self.assertEqual(len(rows), 21)
        self.assertTrue(all(len(row) == 21 for row in rows))

    def test_cli_map_selection_and_switches_record_all_runs(self):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--line-input", "--no-clear", "--log-dir", str(self.directory)],
            cwd=ROOT, input="d\n2\nd\n3\n7\nq\n", capture_output=True,
            text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        logs = [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")]
        self.assertEqual({log["map"]["id"] for log in logs}, set(MAPS))
        self.assertEqual(sorted(log["termination"]["reason"] for log in logs), ["map_changed", "map_changed", "map_changed", "quit"])
        counts = {log["map"]["id"]: log["summary"]["moves"] for log in logs}
        self.assertEqual(counts, {"basic15": 1, "branch15": 1, "hard21": 0, "large41": 0})

    def test_cli_hard_map_flag_and_map_list(self):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--map", "hard21", "--line-input", "--log-dir", str(self.directory)],
            cwd=ROOT, input="q\n", capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(data["map"]["id"], "hard21")
        listed = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--list-maps"],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=15,
        )
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertTrue(all(map_id in listed.stdout for map_id in MAPS))


if __name__ == "__main__":
    unittest.main()
