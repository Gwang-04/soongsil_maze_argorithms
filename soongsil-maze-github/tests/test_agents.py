"""Public-information search, trap recovery, experiment limits and CLI checks."""

from collections import deque
from contextlib import redirect_stdout
import csv
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

from agents import (
    ACTION_ORDER, Decision, Knowledge, RoutingTask, SearchAgent, SearchStats,
    astar_route, bfs_route,
)
from comparison import benchmark_task, run_comparison
from experiment_log import EndReason
from game import ExperimentGame
from levels import BASIC_KEY_PLACEMENTS, MAPS
from main import main
from maze_env import Action, AgentMemory, DELTAS, MazeEnv, MazeLayout, Tile


ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = (ROOT / "tmp" / "test-runs").resolve()
FIXTURE = MazeLayout((
    "#########", "#S..E...#", "#.......#", "#..K.T..#",
    "#.......#", "#.......#", "#########",
))
RECOVERY = MazeLayout((
    "###########", "#S....T..E#", "#####.#.#.#", "#K........#", "###########",
))
ISOLATED = MazeLayout(("#######", "#S#..E#", "###K.T#", "#######"))


def test_knowledge(layout):
    # Full terrain is permitted in planner unit fixtures, never in actual agents.
    tiles = {(r, c): Tile(tile) for r, row in enumerate(layout.rows) for c, tile in enumerate(row)}
    return Knowledge(layout.shape, MappingProxyType(tiles), frozenset(layout.find_all(Tile.TRAP)))


def reference_distance(layout, start, goal):
    """Independent unit-cost state search, including automatic key pickup."""
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (row, col, key), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for dr, dc in DELTAS.values():
            r, c = row + dr, col + dc
            tile = layout.rows[r][c]
            if tile in "#T" or (tile == "E" and not key):
                continue
            child = (r, c, key or tile == "K")
            if child not in seen:
                seen.add(child)
                queue.append((child, distance + 1))
    return None


class PlannerTests(unittest.TestCase):
    def test_routes_handle_key_state_and_match_independent_shortest_distance(self):
        layout = MazeLayout(("#########", "#S.E....#", "#.#.###.#", "#K...T..#", "#########"))
        knowledge = test_knowledge(layout)
        goal = layout.find(Tile.EXIT)
        lengths = []
        for has_key in (False, True):
            start = (*layout.find(Tile.START), has_key)
            expected = reference_distance(layout, start, goal)
            lengths.append(expected)
            for route in (bfs_route(knowledge, start, goal=goal), astar_route(knowledge, start, goal)):
                self.assertEqual(len(route.actions), expected)
                state = start
                for action in route.actions:
                    children = dict(knowledge.successors(state))
                    self.assertIn(action, children)
                    state = children[action]
                self.assertEqual(state[:2], goal)
                self.assertTrue(state[2])
        self.assertGreater(lengths[0], lengths[1])

    def test_unknown_cells_do_not_become_imaginary_routes_to_known_exit(self):
        env, memory = MazeEnv(FIXTURE), AgentMemory()
        observation = env.get_observation()
        memory.update(observation)
        knowledge = Knowledge.from_public(observation, memory)
        start = (*observation.position, True)
        self.assertIsNone(bfs_route(knowledge, start, goal=observation.exit_position).actions)
        self.assertIsNone(astar_route(knowledge, start, observation.exit_position).actions)
        self.assertNotIn(FIXTURE.find(Tile.KEY), knowledge.tiles)
        self.assertNotIn(FIXTURE.find(Tile.TRAP), knowledge.tiles)

    def test_known_trap_is_avoided_by_both_planners(self):
        knowledge = test_knowledge(FIXTURE)
        start, goal = (3, 4, True), (3, 6)
        for route in (bfs_route(knowledge, start, goal=goal), astar_route(knowledge, start, goal)):
            self.assertEqual(len(route.actions), reference_distance(FIXTURE, start, goal))
            position = start[:2]
            for action in route.actions:
                dr, dc = DELTAS[action]
                position = position[0] + dr, position[1] + dc
                self.assertNotEqual(position, FIXTURE.find(Tile.TRAP))

    def test_heuristic_reduces_expansions_on_a_direct_known_route(self):
        knowledge = test_knowledge(FIXTURE)
        start, goal = (1, 1, True), (1, 7)
        bfs = bfs_route(knowledge, start, goal=goal)
        astar = astar_route(knowledge, start, goal)
        self.assertEqual(len(bfs.actions), len(astar.actions))
        self.assertLess(astar.expanded_nodes, bfs.expanded_nodes)

    def test_snapshot_is_immutable_and_independent_of_future_memory(self):
        env, memory = MazeEnv(FIXTURE), AgentMemory()
        observation = env.get_observation()
        memory.update(observation)
        knowledge = Knowledge.from_public(observation, memory)
        before = dict(knowledge.tiles)
        memory.known_tiles[(3, 3)] = Tile.KEY
        memory.known_traps.add((3, 5))
        self.assertEqual(dict(knowledge.tiles), before)
        self.assertEqual(knowledge.traps, frozenset())
        with self.assertRaises(TypeError):
            knowledge.tiles[(3, 3)] = Tile.KEY

    def test_bfs_and_astar_share_exploration_when_key_is_unknown(self):
        env, memory = MazeEnv(FIXTURE), AgentMemory()
        observation = env.get_observation()
        memory.update(observation)
        visited, known = set(memory.visited_positions), dict(memory.known_tiles)
        decisions = [SearchAgent(algorithm).choose(observation, memory) for algorithm in ("bfs", "astar")]
        self.assertEqual(decisions[0].action, decisions[1].action)
        self.assertEqual(decisions[0].target, decisions[1].target)
        self.assertEqual(decisions[0].phase, "explore")
        self.assertEqual(memory.visited_positions, visited)
        self.assertEqual(memory.known_tiles, known)

    def test_paired_benchmark_uses_one_snapshot_and_equal_route_lengths(self):
        knowledge = test_knowledge(FIXTURE)
        row = benchmark_task("fixture", 1, RoutingTask(knowledge, (1, 1, False), (1, 4), "exit"))
        self.assertTrue(row["same_path_length"])
        self.assertEqual(row["bfs_path_length"], row["astar_path_length"])
        self.assertEqual(len(row["snapshot_sha256"]), 64)


class AgentExperimentTests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="maze-agent-test-", dir=TEMP_ROOT)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.assertTrue(self.directory.resolve().is_relative_to(TEMP_ROOT))
        self.temporary.cleanup()

    def game(self, algorithm="bfs", layout=FIXTURE, **kwargs):
        return ExperimentGame(layout, controller=algorithm, log_dir=self.directory, **kwargs)

    def finish_agent(self, game):
        while not game.finished:
            game.automatic_step()
        return json.loads(game.last_saved.json_path.read_text(encoding="utf-8"))

    def test_three_agents_escape_all_three_maps_using_only_public_observations(self):
        for map_id, definition in MAPS.items():
            for algorithm in ("dfs", "bfs", "astar"):
                with self.subTest(map=map_id, algorithm=algorithm):
                    game = self.game(algorithm, definition.layout, map_id=map_id, map_name=definition.name)
                    self.assertIsNone(game.last_saved)
                    self.assertNotIn(definition.layout.find(Tile.KEY), game.memory.known_tiles)
                    self.assertFalse(hasattr(game.agent, "env"))
                    self.assertFalse(hasattr(game.agent, "layout"))
                    data = self.finish_agent(game)
                    self.assertTrue(game.observation.done)
                    self.assertEqual(data["termination"], {"reason": "escaped", "status": "success"})
                    self.assertEqual(data["controller"], algorithm)
                    self.assertEqual(data["settings"]["neighbor_order"], [action.value for action in ACTION_ORDER])
                    self.assertEqual(data["summary"]["wall_collisions"], 0)
                    self.assertEqual(data["summary"]["locked_exit_attempts"], 0)
                    self.assertEqual(data["summary"]["repeated_trap_entries"], 0)
                    self.assertGreater(data["computation"]["expanded_nodes"], 0)
                    self.assertEqual(data["computation"]["expanded_nodes"],
                                     data["computation"]["route_expanded_nodes"] + data["computation"]["exploration_expanded_nodes"])
                    self.assertEqual(len(data["steps"]), data["summary"]["action_attempts"])
                    self.assertTrue(all(step["decision"] is not None for step in data["steps"]))

    def test_return_with_key_replans_for_exit_and_preserves_memory(self):
        for algorithm in ("dfs", "bfs", "astar"):
            with self.subTest(algorithm=algorithm):
                game = self.game(algorithm)
                for action in [Action.DOWN, Action.DOWN] + [Action.RIGHT] * 4:
                    game.move(action)
                self.assertEqual(game.observation.trap_hits, 1)
                self.assertTrue(game.observation.has_key)
                self.assertEqual(game.memory.known_tiles[(3, 3)], Tile.FLOOR)
                self.assertIn((3, 3), game.memory.visited_positions)
                known = set(game.memory.known_tiles)
                game.automatic_step()
                self.assertEqual(game.last_decision.phase, "exit")
                self.assertEqual(game.last_decision.target, game.observation.exit_position)
                data = self.finish_agent(game)
                self.assertTrue(game.observation.done)
                self.assertTrue(known <= game.memory.known_tiles.keys())
                self.assertEqual(data["summary"]["key_pickups"], 1)
                self.assertEqual(data["summary"]["key_losses"], 0)
                self.assertEqual(data["summary"]["trap_hits"], 1)

    def test_dfs_resumes_a_distant_pending_junction_after_a_trap_return(self):
        game = self.game("dfs", RECOVERY)
        for _ in range(5):
            game.move(Action.RIGHT)
        self.assertEqual(game.observation.position, (1, 1))
        self.assertEqual(game.observation.trap_hits, 1)
        self.assertNotIn(RECOVERY.find(Tile.KEY), game.memory.known_tiles)
        for col in range(2, 6):
            game.automatic_step()
            self.assertEqual(game.observation.position, (1, col))
            self.assertEqual(game.last_decision.phase, "dfs_recover")
        game.automatic_step()
        self.assertEqual(game.observation.position, (2, 5))
        self.assertTrue(self.finish_agent(game)["summary"]["success"])
        self.assertEqual(game.observation.trap_hits, 1)

    def test_action_limit_counts_zero_move_wall_attempts_and_seals_once(self):
        game = self.game(max_actions=3)
        game.agent = SimpleNamespace(
            choose=lambda *args: Decision(Action.UP, "explore", (0, 1)), observe=lambda result: None,
        )
        data = self.finish_agent(game)
        self.assertEqual(data["termination"], {"reason": "action_limit", "status": "limit_reached"})
        self.assertEqual(data["summary"]["action_attempts"], 3)
        self.assertEqual(data["summary"]["moves"], 0)
        self.assertEqual(data["summary"]["wall_collisions"], 3)
        game.automatic_step()
        game.move(Action.RIGHT)
        game.finish(EndReason.QUIT)
        self.assertEqual(game.observation.moves, 0)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)

    def test_success_on_last_allowed_attempt_takes_precedence_over_limit(self):
        layout = MazeLayout(("#######", "#SKE..#", "#..T..#", "#######"))
        game = self.game(layout=layout, max_actions=2)
        data = self.finish_agent(game)
        self.assertTrue(data["summary"]["success"])
        self.assertEqual(data["termination"]["reason"], "escaped")
        self.assertEqual(data["summary"]["action_attempts"], 2)

    def test_exhaustion_is_distinct_from_user_quit_and_action_limit(self):
        for algorithm in ("dfs", "bfs", "astar"):
            with self.subTest(algorithm=algorithm):
                game = self.game(algorithm, ISOLATED)
                data = self.finish_agent(game)
                self.assertEqual(data["termination"], {"reason": "exhausted", "status": "exhausted"})
                self.assertEqual(data["summary"]["action_attempts"], 0)
                self.assertEqual(data["computation"]["decision_calls"], 1)

    def test_manual_play_has_no_automatic_action_limit(self):
        game = self.game("manual", max_actions=1)
        for _ in range(3):
            game.move(Action.UP)
        self.assertFalse(game.finished)
        self.assertEqual(game.recorder.action_attempts, 3)
        self.assertEqual(game.recorder.computation["decision_calls"], 0)

    def test_restart_is_fresh_and_reproduces_deterministic_actions(self):
        game = self.game("dfs")
        first = self.finish_agent(game)
        game.restart()
        self.assertEqual(game.memory.travel_history, [(1, 1)])
        self.assertEqual(game.recorder.computation["expanded_nodes"], 0)
        self.assertEqual(game.memory.known_traps, set())
        second = self.finish_agent(game)
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["summary"], second["summary"])
        self.assertEqual([step["action"] for step in first["steps"]], [step["action"] for step in second["steps"]])
        self.assertEqual(first["computation"]["expanded_nodes"], second["computation"]["expanded_nodes"])

    def test_controller_switch_saves_previous_run_and_clears_memory_and_computation(self):
        game = self.game("dfs")
        game.automatic_step()
        saved = game.change_controller("astar")
        data = json.loads(saved.json_path.read_text(encoding="utf-8"))
        self.assertEqual(data["controller"], "dfs")
        self.assertEqual(data["termination"]["reason"], "controller_changed")
        self.assertEqual(game.controller, "astar")
        self.assertEqual(game.recorder.action_attempts, 0)
        self.assertEqual(game.recorder.computation["decision_calls"], 0)
        self.assertEqual(game.memory.travel_history, [(1, 1)])
        game.change_controller("manual")
        self.assertIsNone(game.agent)

    def test_failed_limit_save_can_retry_without_a_new_action_or_losing_state(self):
        blocker = self.directory / "not-a-directory"
        blocker.write_text("blocked", encoding="utf-8")
        game = self.game(max_actions=1)
        game.log_dir = blocker
        run_id = game.recorder.run_id
        with self.assertRaises(OSError):
            game.automatic_step()
        self.assertEqual(game.recorder.action_attempts, 1)
        self.assertFalse(game.finished)
        game.log_dir = self.directory
        game.automatic_step()
        self.assertEqual(game.recorder.run_id, run_id)
        self.assertEqual(game.recorder.action_attempts, 1)
        self.assertEqual(game.end_reason, EndReason.ACTION_LIMIT)

    def test_computation_timing_excludes_logging_and_external_wait(self):
        game = self.game(max_actions=1)
        game.agent = SimpleNamespace(
            choose=lambda *args: Decision(Action.UP, "explore", None, SearchStats(path_planning_seconds=0.012)),
            observe=lambda result: None,
        )
        with patch("game.perf_counter", side_effect=[10.0, 10.02]):
            game.automatic_step()
        data = json.loads(game.last_saved.json_path.read_text(encoding="utf-8"))
        self.assertAlmostEqual(data["computation"]["decision_time_seconds"], 0.02)
        self.assertAlmostEqual(data["computation"]["path_planning_seconds"], 0.012)

    def cli(self, *arguments, input="", timeout=20):
        return subprocess.run(
            [sys.executable, "-X", "utf8", "main.py", "--log-dir", str(self.directory), *arguments],
            cwd=ROOT, input=input, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        )

    def test_cli_single_step_controller_switch_map_switch_and_reset(self):
        result = self.cli("--line-input", "--no-clear", input="4\nn\n\n5\nn\nr\n6\nn\n3\nn\nm\nd\nq\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*.json")]
        self.assertEqual(len(records), 7)
        self.assertEqual({record["controller"] for record in records}, {"manual", "dfs", "bfs", "astar"})
        self.assertEqual(sum(record["termination"]["reason"] == "controller_changed" for record in records), 4)
        dfs = next(record for record in records if record["controller"] == "dfs")
        self.assertEqual(dfs["summary"]["action_attempts"], 2)
        self.assertEqual(dfs["computation"]["decision_calls"], 2)
        self.assertIn("일시정지", result.stdout)

    def test_cli_headless_and_invalid_automatic_arguments(self):
        result = self.cli("--agent", "astar", "--map", "hard21", "--headless")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(data["controller"], "astar")
        self.assertTrue(data["summary"]["success"])
        limited = self.cli("--agent", "bfs", "--headless", "--max-actions", "1")
        self.assertEqual(limited.returncode, 2, limited.stderr)
        for arguments in (("--headless",), ("--agent", "bfs", "--max-actions", "0"), ("--delay", "nan"), ("--key-placements",)):
            with self.subTest(arguments=arguments):
                invalid = self.cli(*arguments)
                self.assertEqual(invalid.returncode, 2)
                self.assertIn("error:", invalid.stderr)

    def test_headless_keyboard_interrupt_saves_the_active_experiment(self):
        with patch("game.ExperimentGame.automatic_step", side_effect=KeyboardInterrupt), redirect_stdout(StringIO()):
            code = main(["--agent", "bfs", "--headless", "--log-dir", str(self.directory)])
        self.assertEqual(code, 2)
        data = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(data["termination"], {"reason": "interrupted", "status": "aborted"})

    def test_batch_comparison_resets_every_run_and_writes_paired_route_csv(self):
        result = self.cli("--compare", "--repeat", "2", timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        summaries = [path for path in self.directory.glob("comparison_*.csv") if not path.stem.endswith("_routes")]
        self.assertEqual(len(summaries), 1)
        with summaries[0].open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(MAPS) * 3 * 2)
        self.assertTrue(all(row["status"] == "success" for row in rows))
        self.assertEqual(len(list(self.directory.glob("*.json"))), len(MAPS) * 3 * 2)
        for map_id in MAPS:
            for algorithm in ("dfs", "bfs", "astar"):
                repeated = [row for row in rows if row["map_id"] == map_id and row["algorithm"] == algorithm]
                self.assertEqual(len(repeated), 2)
                self.assertEqual(repeated[0]["moves"], repeated[1]["moves"])
                self.assertEqual(repeated[0]["expanded_nodes"], repeated[1]["expanded_nodes"])
                self.assertNotEqual(repeated[0]["run_id"], repeated[1]["run_id"])
        route_path = next(self.directory.glob("*_routes.csv"))
        with route_path.open(encoding="utf-8-sig", newline="") as handle:
            tasks = list(csv.DictReader(handle))
        self.assertGreater(len(tasks), 0)
        self.assertTrue(all(row["bfs_path_length"] == row["astar_path_length"] for row in tasks))
        self.assertTrue(all(row["timing_repeats"] == "2" for row in tasks))
        self.assertEqual({row["map_id"] for row in tasks}, set(MAPS))

    def test_batch_interrupt_preserves_partial_summary_and_aborted_record(self):
        with patch("game.ExperimentGame.automatic_step", side_effect=KeyboardInterrupt), redirect_stdout(StringIO()):
            code = run_comparison(self.directory)
        self.assertEqual(code, 130)
        self.assertEqual(len(list(self.directory.glob("*.json"))), 1)
        self.assertEqual(len(list(self.directory.glob("comparison_*.csv"))), 1)
        data = json.loads(next(self.directory.glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(data["termination"]["status"], "aborted")

    def test_key_placement_comparison_reports_moves_at_the_pickup_event(self):
        result = self.cli("--compare", "--key-placements", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = next(path for path in self.directory.glob("comparison_keys_*.csv") if not path.stem.endswith("_routes"))
        with summary.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = {
            (definition.map_id, algorithm)
            for definition in BASIC_KEY_PLACEMENTS
            for algorithm in ("dfs", "bfs", "astar")
        }
        self.assertEqual(len(rows), 9)
        self.assertEqual({(row["map_id"], row["algorithm"]) for row in rows}, expected)
        for row in rows:
            with self.subTest(map_id=row["map_id"], algorithm=row["algorithm"]):
                data = json.loads((self.directory / row["json_file"]).read_text(encoding="utf-8"))
                pickup = next(
                    step for step in data["steps"]
                    if any(event["kind"] == "key_picked_up" for event in step["events"])
                )
                self.assertEqual(int(row["moves_to_key"]), pickup["after"]["moves"])
                self.assertLess(int(row["moves_to_key"]), int(row["moves"]))


if __name__ == "__main__":
    unittest.main()
