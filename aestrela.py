from __future__ import annotations

import heapq
import tracemalloc
from dataclasses import dataclass
from itertools import count
from time import perf_counter
from typing import Iterator, Sequence

from ambiente import MazeGrid
from modelos import Cell, DeliveryPlan, MetricValue, PlannerResult
from modelos import ProgressCallback, Scenario


INF = 10**18
LogisticsState = tuple[int, int, int]


def _active_indices(mask: int) -> Iterator[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


@dataclass(frozen=True)
class SearchResult:
    path: tuple[Cell, ...]
    found: bool
    nodes_expanded: int
    nodes_generated: int
    max_frontier: int


def h_score(cell: Cell, destination: Cell) -> int:
    return abs(cell[0] - destination[0]) + abs(cell[1] - destination[1])


def astar_search(maze: MazeGrid, start: Cell, destination: Cell) -> SearchResult:
    if not maze.contains(start) or not maze.contains(destination):
        return SearchResult((), False, 0, 0, 0)
    if start == destination:
        return SearchResult((start,), True, 1, 1, 1)

    sequence = count()
    frontier: list[tuple[int, int, int, Cell]] = []
    initial_h = h_score(start, destination)
    heapq.heappush(frontier, (initial_h, initial_h, next(sequence), start))
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, int] = {start: 0}
    closed: set[Cell] = set()
    nodes_generated = 1
    max_frontier = 1

    while frontier:
        _, _, _, cell = heapq.heappop(frontier)
        if cell in closed:
            continue
        closed.add(cell)

        if cell == destination:
            reverse_path = [cell]
            while reverse_path[-1] != start:
                reverse_path.append(came_from[reverse_path[-1]])
            reverse_path.reverse()
            return SearchResult(
                tuple(reverse_path),
                True,
                len(closed),
                nodes_generated,
                max_frontier,
            )

        for neighbor in maze.neighbors(cell):
            tentative = g_score[cell] + 1
            if tentative >= g_score.get(neighbor, INF):
                continue
            came_from[neighbor] = cell
            g_score[neighbor] = tentative
            heuristic = h_score(neighbor, destination)
            heapq.heappush(
                frontier,
                (tentative + heuristic, heuristic, next(sequence), neighbor),
            )
            nodes_generated += 1
        max_frontier = max(max_frontier, len(frontier))

    return SearchResult((), False, len(closed), nodes_generated, max_frontier)


class PathCache:

    def __init__(self, maze: MazeGrid) -> None:
        self.maze = maze
        self._paths: dict[tuple[Cell, Cell], tuple[Cell, ...] | None] = {}
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.max_frontier = 0
        self.search_count = 0

    def path(self, start: Cell, destination: Cell) -> tuple[Cell, ...] | None:
        key = (start, destination)
        if key in self._paths:
            return self._paths[key]

        result = astar_search(self.maze, start, destination)
        self.search_count += 1
        self.nodes_expanded += result.nodes_expanded
        self.nodes_generated += result.nodes_generated
        self.max_frontier = max(self.max_frontier, result.max_frontier)
        value = result.path if result.found else None
        self._paths[key] = value
        self._paths[(destination, start)] = tuple(reversed(value)) if value else None
        return value

    def distance(self, start: Cell, destination: Cell) -> int | None:
        path = self.path(start, destination)
        return len(path) - 1 if path else None

    def metrics(self) -> dict[str, int]:
        return {
            "nós_expandidos": self.nodes_expanded,
            "nós_gerados": self.nodes_generated,
            "fronteira_máxima": self.max_frontier,
            "buscas_a_estrela": self.search_count,
        }


def build_route(
    scenario: Scenario,
    product_order: Sequence[int],
    slot_order: Sequence[int],
    cache: PathCache,
) -> tuple[list[Cell], list[DeliveryPlan], bool]:
    products = {product.id: product for product in scenario.products}
    slots = {slot.id: slot for slot in scenario.dock_slots}
    route = [scenario.start]
    deliveries: list[DeliveryPlan] = []
    current = scenario.start

    for product_id, slot_id in zip(product_order, slot_order):
        product_path = cache.path(current, products[product_id].position)
        dock_path = cache.path(products[product_id].position, slots[slot_id].position)
        if not product_path or not dock_path:
            return route, deliveries, False

        route.extend(product_path[1:])
        pickup_step = len(route) - 1
        route.extend(dock_path[1:])
        delivery_step = len(route) - 1
        deliveries.append(
            DeliveryPlan(
                product_id=product_id,
                dock_id=slot_id,
                pickup_step=pickup_step,
                delivery_step=delivery_step,
            )
        )
        current = slots[slot_id].position

    return route, deliveries, True


class AStarPlanner:
    algorithm = "A*"

    def plan(
        self,
        scenario: Scenario,
        progress_callback: ProgressCallback | None = None,
    ) -> PlannerResult:
        tracemalloc.start()
        started_at = perf_counter()
        cache = PathCache(scenario.maze)
        products = tuple(sorted(scenario.products, key=lambda item: item.id))
        slots = tuple(sorted(scenario.dock_slots, key=lambda item: item.id))
        item_count = len(products)

        start_to_product = [
            cache.distance(scenario.start, product.position) for product in products
        ]
        product_to_slot: list[list[int | None]] = [
            [None] * item_count for _ in range(item_count)
        ]
        for slot_index, slot in enumerate(slots):
            for product_index, product in enumerate(products):
                product_to_slot[product_index][slot_index] = cache.distance(
                    slot.position, product.position
                )

        distances_valid = all(distance is not None for distance in start_to_product)
        distances_valid = distances_valid and all(
            distance is not None for row in product_to_slot for distance in row
        )

        full_mask = (1 << item_count) - 1
        initial_state: LogisticsState = (-1, full_mask, full_mask)
        heuristic_cache: dict[LogisticsState, int] = {}

        def heuristic(state: LogisticsState) -> int:
            cached = heuristic_cache.get(state)
            if cached is not None:
                return cached
            current_slot, product_mask, slot_mask = state
            if product_mask == 0:
                return 0
            product_indices = tuple(_active_indices(product_mask))
            slot_indices = tuple(_active_indices(slot_mask))
            if current_slot == -1:
                first_leg = min(int(start_to_product[index]) for index in product_indices)
            else:
                first_leg = min(
                    int(product_to_slot[index][current_slot])
                    for index in product_indices
                )

            delivery_by_product = sum(
                min(int(product_to_slot[product][slot]) for slot in slot_indices)
                for product in product_indices
            )
            delivery_by_slot = sum(
                min(int(product_to_slot[product][slot]) for product in product_indices)
                for slot in slot_indices
            )
            delivery_legs = max(delivery_by_product, delivery_by_slot)


            incoming_minimums = [
                min(int(product_to_slot[product][slot]) for slot in slot_indices)
                for product in product_indices
            ]
            between_trips = (
                sum(incoming_minimums) - max(incoming_minimums)
                if len(incoming_minimums) > 1
                else 0
            )
            value = first_leg + delivery_legs + between_trips
            heuristic_cache[state] = value
            return value

        product_order: list[int] = []
        slot_order: list[int] = []
        high_level_expanded = 0
        high_level_generated = 1
        high_level_frontier = 1
        goal_state: LogisticsState | None = None

        if distances_valid:
            sequence = count()
            initial_h = heuristic(initial_state)
            frontier: list[tuple[int, int, int, int, LogisticsState]] = [
                (initial_h, initial_h, 0, next(sequence), initial_state)
            ]
            g_score: dict[LogisticsState, int] = {initial_state: 0}
            expanded_cost: dict[LogisticsState, int] = {}
            came_from: dict[LogisticsState, tuple[LogisticsState, int, int]] = {}

            while frontier:
                _, _, queued_g, _, state = heapq.heappop(frontier)
                if queued_g != g_score.get(state):
                    continue
                if queued_g >= expanded_cost.get(state, INF):
                    continue
                expanded_cost[state] = queued_g
                high_level_expanded += 1
                current_slot, product_mask, slot_mask = state

                if product_mask == 0:
                    goal_state = state
                    break

                for product_index in _active_indices(product_mask):
                    if current_slot == -1:
                        first_leg = int(start_to_product[product_index])
                    else:
                        first_leg = int(product_to_slot[product_index][current_slot])
                    for slot_index in _active_indices(slot_mask):
                        step_cost = first_leg + int(product_to_slot[product_index][slot_index])
                        next_state: LogisticsState = (
                            slot_index,
                            product_mask ^ (1 << product_index),
                            slot_mask ^ (1 << slot_index),
                        )
                        tentative = queued_g + step_cost
                        if tentative >= g_score.get(next_state, INF):
                            continue
                        g_score[next_state] = tentative
                        came_from[next_state] = (
                            state,
                            products[product_index].id,
                            slots[slot_index].id,
                        )
                        next_h = heuristic(next_state)
                        heapq.heappush(
                            frontier,
                            (
                                tentative + next_h,
                                next_h,
                                tentative,
                                next(sequence),
                                next_state,
                            ),
                        )
                        high_level_generated += 1
                high_level_frontier = max(high_level_frontier, len(frontier))
                if progress_callback and high_level_expanded % 250 == 0:
                    progress_callback(
                        {
                            "fase": "A* global",
                            "estados_expandidos": high_level_expanded,
                            "fronteira": len(frontier),
                        }
                    )

            if goal_state is not None:
                transitions: list[tuple[int, int]] = []
                state = goal_state
                while state != initial_state:
                    previous, product_id, slot_id = came_from[state]
                    transitions.append((product_id, slot_id))
                    state = previous
                transitions.reverse()
                product_order = [product_id for product_id, _ in transitions]
                slot_order = [slot_id for _, slot_id in transitions]

        route, deliveries, success = build_route(
            scenario, product_order, slot_order, cache
        )
        planning_ms = (perf_counter() - started_at) * 1000
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        success = success and goal_state is not None and len(deliveries) == len(products)
        navigation_metrics = cache.metrics()
        metrics: dict[str, MetricValue] = {
            "nos_expandidos": high_level_expanded,
            "nos_gerados": high_level_generated,
            "fronteira_maxima": high_level_frontier,
            "nos_expandidos_navegacao": navigation_metrics["nos_expandidos"],
            "buscas_a_estrela_navegacao": navigation_metrics["buscas_a_estrela"],
            "passos": max(0, len(route) - 1),
            "ordem_produtos": "-".join(map(str, product_order)),
            "ordem_docas": "-".join(map(str, slot_order)),
            "memoria_pico_kib": peak_bytes / 1024,
        }
        return PlannerResult(
            algorithm=self.algorithm,
            route=route,
            deliveries=deliveries,
            planning_ms=planning_ms,
            metrics=metrics,
            success=success,
        )
