"""Persistence and experiment-boundary checks for manual runs."""

import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from experiment_log import EndReason
from main import ManualGame
from maze_env import Action, EventKind, MazeLayout


LAYOUT = MazeLayout((
    "#########",
    "#S..E...#",
    "#.......#",
    "#..K.T..#",
    "#.......#",
    "#.......#",
    "#########",
))
TEST_TEMP_ROOT = (Path(__file__).resolve().parents[1] / "tmp" / "test-runs").resolve()


class ExperimentLogTests(unittest.TestCase):
    def setUp(self):
        TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="maze-log-test-", dir=TEST_TEMP_ROOT)
        self.directory = Path(self.temporary.name)
        self.game = ManualGame(
            LAYOUT, log_dir=self.directory, map_id="fixture", map_name="테스트 맵",
        )

    def tearDown(self):
        # All paths are the direct children of this newly created temp directory.
        self.assertTrue(self.directory.resolve().is_relative_to(TEST_TEMP_ROOT))
        self.temporary.cleanup()

    def play(self, actions):
        for action in actions:
            self.game.move(action)

    def load(self, saved):
        return json.loads(saved.json_path.read_text(encoding="utf-8"))

    def test_quit_saves_utf8_pair_without_unobserved_map_information(self):
        saved = self.game.finish(EndReason.QUIT)
        data = self.load(saved)
        self.assertTrue(saved.text_path.is_file())
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["termination"], {"reason": "quit", "status": "aborted"})
        self.assertEqual(data["summary"]["action_attempts"], 0)
        self.assertEqual(data["steps"], [])
        self.assertEqual(data["map"]["id"], "fixture")
        self.assertTrue(data["started_at"].endswith("+09:00"))
        self.assertEqual(len(data["map"]["layout_sha256"]), 64)
        self.assertNotIn("key_position", data["map"])
        self.assertNotIn("trap_position", data["map"])
        self.assertEqual(data["final_memory"]["known_grid"][3][3], "?")
        self.assertEqual(data["final_memory"]["known_grid"][3][5], "?")
        self.assertIn("실험 기록", saved.text_path.read_text(encoding="utf-8"))
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_attempts_and_collisions_are_separate_from_moves(self):
        self.play([Action.UP, Action.RIGHT, Action.RIGHT, Action.RIGHT])
        saved = self.game.finish(EndReason.QUIT)
        data = self.load(saved)
        self.assertEqual(data["summary"]["action_attempts"], 4)
        self.assertEqual(data["summary"]["moves"], 2)
        self.assertEqual(data["summary"]["wall_collisions"], 1)
        self.assertEqual(data["summary"]["locked_exit_attempts"], 1)
        self.assertEqual(data["steps"][0]["before"]["position"], data["steps"][0]["after"]["position"])

    def test_repeated_trap_returns_are_recorded_with_key_retained(self):
        trip = [Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT]
        self.play(trip + trip)
        saved = self.game.finish(EndReason.QUIT)
        data = self.load(saved)
        summary = data["summary"]
        self.assertEqual(summary["moves"], 12)
        self.assertEqual(summary["trap_hits"], 2)
        self.assertEqual(summary["first_trap_discoveries"], 1)
        self.assertEqual(summary["repeated_trap_entries"], 1)
        self.assertEqual(summary["key_pickups"], 1)
        self.assertEqual(summary["key_losses"], 0)
        self.assertTrue(data["settings"]["trap_return_keeps_key"])
        self.assertEqual(data["final_observation"]["position"], [1, 1])
        self.assertTrue(data["final_observation"]["has_key"])
        self.assertEqual(data["steps"][5]["trap_encounter"], "first_discovery")
        self.assertEqual(data["steps"][11]["trap_encounter"], "repeat")
        for index in (5, 11):
            step = data["steps"][index]
            self.assertTrue(step["before"]["has_key"])
            self.assertTrue(step["after"]["has_key"])
            self.assertEqual([event["kind"] for event in step["events"]], ["moved", "trap_triggered"])
        self.assertEqual(data["final_memory"]["known_grid"][3][3], ".")
        self.assertEqual(data["final_memory"]["known_grid"][3][5], "T")
        self.assertEqual(data["final_memory"]["travel_history"][-2:], [[3, 5], [1, 1]])
        self.assertEqual(data["summary"]["action_attempts"], 12)

    def test_reset_saves_old_run_before_clearing_memory_and_gets_unique_id(self):
        self.play([Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT])
        first_id = self.game.recorder.run_id
        saved = self.game.restart()
        data = self.load(saved)
        self.assertEqual(data["termination"]["reason"], "reset")
        self.assertTrue(data["final_observation"]["has_key"])
        self.assertEqual(data["summary"]["moves"], 4)
        self.assertNotEqual(first_id, self.game.recorder.run_id)
        self.assertFalse(self.game.observation.has_key)
        self.assertEqual(self.game.observation.moves, 0)
        self.assertEqual(self.game.memory.travel_history, [(1, 1)])
        self.assertIsNone(self.game.last_saved)
        second = self.game.finish(EndReason.QUIT)
        self.assertNotEqual(saved.json_path, second.json_path)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 2)
        self.assertEqual(len(list(self.directory.glob("*.txt"))), 2)
        self.assertEqual(self.load(saved), data)

    def test_patrol_collisions_on_entry_crossing_and_wait_preserve_key_and_real_path(self):
        layout = MazeLayout(("#########", "#S....E.#", "#K......#", "#......T#", "#########"))
        game = ManualGame(layout, log_dir=self.directory)
        previous = game.change_ghost_patrol(((1, 3), (1, 2)))
        self.assertEqual(previous.reason, EndReason.GHOST_MODE_CHANGED)
        initial = game.observation
        self.assertIsNone(initial.ghost_position)
        self.assertEqual(game.move(Action.UP).observation, initial)

        pickup = game.move(Action.DOWN)
        self.assertTrue(pickup.observation.has_key)
        self.assertEqual(pickup.observation.ghost_position, (1, 2))
        game.move(Action.UP)
        entry = game.move(Action.RIGHT)
        self.assertEqual(entry.observation.position, initial.position)
        self.assertEqual(entry.observation.ghost_hits, 1)
        self.assertEqual(entry.events[-1].kind, EventKind.GHOST_COLLISION)

        self.assertEqual(game.move(Action.RIGHT).observation.position, (1, 2))
        crossing = game.move(Action.RIGHT)
        self.assertEqual(crossing.observation.position, initial.position)
        self.assertEqual(crossing.observation.ghost_hits, 2)
        game.move(Action.RIGHT)
        waiting = game.move(Action.WAIT)
        self.assertEqual(waiting.observation.position, initial.position)
        self.assertEqual([event.kind for event in waiting.events], [EventKind.WAITED, EventKind.GHOST_COLLISION])
        self.assertTrue(waiting.observation.has_key)
        self.assertEqual(game.memory.known_traps, set())
        self.assertEqual(game.recorder.travel_edges, (
            ((1, 1), (2, 1)), ((2, 1), (1, 1)), ((1, 1), (1, 2)),
            ((1, 1), (1, 2)), ((1, 2), (1, 3)), ((1, 1), (1, 2)),
        ))

        saved = game.finish(EndReason.QUIT)
        data = self.load(saved)
        self.assertTrue(data["settings"]["ghost_enabled"])
        summary = data["summary"]
        self.assertEqual((summary["moves"], summary["turns"], summary["waits"]), (6, 7, 1))
        self.assertEqual((summary["trap_hits"], summary["ghost_hits"]), (0, 3))
        self.assertIn("유령 접촉 3회", saved.text_path.read_text(encoding="utf-8"))

    def test_escape_saves_immediately_and_later_quit_does_not_duplicate(self):
        self.play([Action.DOWN, Action.DOWN, Action.RIGHT, Action.RIGHT, Action.UP, Action.UP, Action.RIGHT])
        self.assertTrue(self.game.observation.done)
        self.assertIsNotNone(self.game.last_saved)
        first = self.game.last_saved
        data = self.load(first)
        self.assertEqual(data["termination"], {"reason": "escaped", "status": "success"})
        self.assertEqual(data["summary"]["moves"], 7)
        self.assertEqual(data["summary"]["moves_to_key"], 4)
        self.assertEqual(data["summary"]["moves_after_key"], 3)
        self.game.move(Action.LEFT)
        second = self.game.finish(EndReason.QUIT)
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)
        self.assertEqual(self.load(second), data)

    def test_failed_save_does_not_reset_or_seal_the_current_run(self):
        self.play([Action.RIGHT])
        blocker = self.directory / "not-a-directory"
        blocker.write_text("file", encoding="utf-8")
        self.game.log_dir = blocker
        before = self.game.observation
        run_id = self.game.recorder.run_id
        with self.assertRaises(OSError):
            self.game.restart()
        self.assertEqual(self.game.observation, before)
        self.assertEqual(self.game.recorder.run_id, run_id)
        self.game.move(Action.LEFT)
        self.game.log_dir = self.directory
        saved = self.game.finish(EndReason.QUIT)
        self.assertEqual(self.load(saved)["summary"]["moves"], 2)

    def test_interrupted_run_is_saved_as_aborted(self):
        self.game.move(Action.DOWN)
        saved = self.game.finish(EndReason.INTERRUPTED)
        data = self.load(saved)
        self.assertEqual(data["termination"], {"reason": "interrupted", "status": "aborted"})
        self.assertEqual(data["summary"]["moves"], 1)

    def test_end_of_input_with_save_error_exits_instead_of_retrying_forever(self):
        blocker = self.directory / "not-a-directory"
        blocker.write_text("file", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--line-input", "--log-dir", str(blocker)],
            cwd=Path(__file__).resolve().parents[1], input="", capture_output=True,
            text=True, encoding="utf-8", timeout=10,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("기록 저장에 실패", result.stderr)


if __name__ == "__main__":
    unittest.main()
