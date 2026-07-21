from __future__ import annotations

import csv
from itertools import permutations
import random
import unittest
from pathlib import Path

from aestrela import AStarPlanner, PathCache, astar_search
from algoritmogenetico import (
    Chromosome,
    GeneticConfig,
    GeneticPlanner,
    mutate_chromosome,
    ordered_crossover,
)
from ambiente import generate_scenario
from metricas import calculate_route_metrics, compare_plans
from modelos import AgentRuntimeState
from resultados import save_results


class EnvironmentTests(unittest.TestCase):
    def test_seed_reproduces_maze_products_and_docks(self) -> None:
        first = generate_scenario(15, 21, 4, seed=12345)
        second = generate_scenario(15, 21, 4, seed=12345)
        self.assertEqual(first.products, second.products)
        self.assertEqual(first.dock_slots, second.dock_slots)
        for cell in first.maze.cells:
            self.assertEqual(first.maze.neighbors(cell), second.maze.neighbors(cell))

    def test_all_cells_are_reachable_from_start(self) -> None:
        scenario = generate_scenario(15, 21, 6, seed=42)
        pending = [scenario.start]
        visited = {scenario.start}
        while pending:
            cell = pending.pop()
            for neighbor in scenario.maze.neighbors(cell):
                if neighbor not in visited:
                    visited.add(neighbor)
                    pending.append(neighbor)
        self.assertEqual(len(visited), len(scenario.maze.cells))

class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = generate_scenario(13, 19, 4, seed=801)

    def test_astar_path_contains_only_valid_steps(self) -> None:
        goal = self.scenario.products[0].position
        result = astar_search(self.scenario.maze, self.scenario.start, goal)
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], self.scenario.start)
        self.assertEqual(result.path[-1], goal)
        for first, second in zip(result.path, result.path[1:]):
            self.assertTrue(self.scenario.maze.connected(first, second))

    def test_astar_finds_the_optimal_complete_operation(self) -> None:
        planner_result = AStarPlanner().plan(self.scenario)
        cache = PathCache(self.scenario.maze)
        products = {product.id: product.position for product in self.scenario.products}
        slots = {slot.id: slot.position for slot in self.scenario.dock_slots}
        best_cost = 10**18
        for product_order in permutations(products):
            for slot_order in permutations(slots):
                current = self.scenario.start
                cost = 0
                for product_id, slot_id in zip(product_order, slot_order):
                    cost += int(cache.distance(current, products[product_id]))
                    cost += int(cache.distance(products[product_id], slots[slot_id]))
                    current = slots[slot_id]
                best_cost = min(best_cost, cost)
        self.assertEqual(planner_result.metrics["passos"], best_cost)

    def test_genetic_operators_preserve_permutations(self) -> None:
        rng = random.Random(5)
        first = (1, 2, 3, 4, 5, 6, 7, 8)
        second = tuple(reversed(first))
        child = ordered_crossover(first, second, rng)
        self.assertEqual(set(child), set(first))
        self.assertEqual(len(child), len(set(child)))

        chromosome = Chromosome(first, second)
        mutated = mutate_chromosome(chromosome, 1.0, rng)
        self.assertEqual(set(mutated.products), set(first))
        self.assertEqual(set(mutated.slots), set(second))

    def test_both_planners_complete_all_deliveries(self) -> None:
        astar_plan = AStarPlanner().plan(self.scenario)
        genetic_plan = GeneticPlanner(
            GeneticConfig(population_size=36, generations=30, stagnation_limit=10)
        ).plan(self.scenario)
        self.assertTrue(astar_plan.success)
        self.assertTrue(genetic_plan.success)
        self.assertEqual(len(astar_plan.deliveries), len(self.scenario.products))
        self.assertEqual(len(genetic_plan.deliveries), len(self.scenario.products))
        comparison = compare_plans(astar_plan, genetic_plan)
        self.assertGreaterEqual(comparison.optimality_gap_percent, 0)
        self.assertGreaterEqual(comparison.route_overlap_percent, 0)
        self.assertLessEqual(comparison.route_overlap_percent, 100)

    def test_route_metrics_count_turns_and_revisits(self) -> None:
        route = [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]
        metrics = calculate_route_metrics(route, delivery_count=1)
        self.assertEqual(metrics.steps, 4)
        self.assertEqual(metrics.turns, 3)
        self.assertEqual(metrics.unique_cells, 4)
        self.assertEqual(metrics.revisits, 1)
        self.assertEqual(metrics.revisit_percent, 25)

    def test_runtime_product_copies_are_independent(self) -> None:
        plan = AStarPlanner().plan(self.scenario)
        ids = tuple(product.id for product in self.scenario.products)
        first = AgentRuntimeState(plan)
        second = AgentRuntimeState(plan)
        first.begin()
        second.begin()

        pickup_step = plan.deliveries[0].pickup_step
        for tick in range(pickup_step):
            first.advance(tick / 35)
        product_id = plan.deliveries[0].product_id
        self.assertEqual(first.product_status[product_id], "carrying")
        self.assertEqual(second.product_status[product_id], "pending")

        while not first.finished:
            first.advance(first.steps / 35)
        self.assertEqual(first.delivered_count, len(ids))
        self.assertEqual(second.delivered_count, 0)

    def test_results_are_written_with_new_schema(self) -> None:
        plan = AStarPlanner().plan(self.scenario)
        ids = tuple(product.id for product in self.scenario.products)
        runtime = AgentRuntimeState(plan)
        runtime.begin()
        while not runtime.finished:
            runtime.advance(runtime.steps / 35)

        output = Path("tests") / "_temporary_results.csv"
        try:
            save_results(self.scenario, {"astar": runtime}, output)
            with output.open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["algoritmo"], "A*")
            self.assertEqual(int(rows[0]["produtos_entregues"]), len(ids))
            self.assertIn("curvas", rows[0])
            self.assertIn("percentual_revisita", rows[0])
            self.assertIn("gap_otimo_percentual", rows[0])
        finally:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
