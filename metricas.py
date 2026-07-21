from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from modelos import Cell, PlannerResult


@dataclass(frozen=True)
class RouteMetrics:
    steps: int
    turns: int
    unique_cells: int
    revisits: int
    revisit_percent: float
    steps_per_delivery: float


@dataclass(frozen=True)
class ComparisonMetrics:
    astar: RouteMetrics
    genetic: RouteMetrics
    optimality_gap_percent: float
    route_overlap_percent: float
    product_order_agreement_percent: float
    dock_order_agreement_percent: float
    ga_reached_optimum: bool
    faster_planner: str
    planning_speed_ratio: float


def calculate_route_metrics(
    route: Sequence[Cell],
    delivery_count: int,
) -> RouteMetrics:
    steps = max(0, len(route) - 1)
    directions = [
        (second[0] - first[0], second[1] - first[1])
        for first, second in zip(route, route[1:])
    ]
    turns = sum(first != second for first, second in zip(directions, directions[1:]))
    unique_cells = len(set(route))
    revisits = max(0, len(route) - unique_cells)
    return RouteMetrics(
        steps=steps,
        turns=turns,
        unique_cells=unique_cells,
        revisits=revisits,
        revisit_percent=(revisits / max(1, steps)) * 100,
        steps_per_delivery=steps / max(1, delivery_count),
    )


def _agreement(first: Sequence[int], second: Sequence[int]) -> float:
    total = max(1, len(first), len(second))
    return sum(left == right for left, right in zip(first, second)) / total * 100


def compare_plans(astar: PlannerResult, genetic: PlannerResult) -> ComparisonMetrics:
    astar_metrics = calculate_route_metrics(astar.route, len(astar.deliveries))
    genetic_metrics = calculate_route_metrics(genetic.route, len(genetic.deliveries))
    astar_cells = set(astar.route)
    genetic_cells = set(genetic.route)
    union = astar_cells | genetic_cells
    overlap = len(astar_cells & genetic_cells) / max(1, len(union)) * 100

    astar_products = [delivery.product_id for delivery in astar.deliveries]
    genetic_products = [delivery.product_id for delivery in genetic.deliveries]
    astar_docks = [delivery.dock_id for delivery in astar.deliveries]
    genetic_docks = [delivery.dock_id for delivery in genetic.deliveries]
    optimality_gap = (genetic_metrics.steps - astar_metrics.steps) / max(
        1, astar_metrics.steps
    ) * 100
    astar_time = max(0.000001, astar.planning_ms)
    genetic_time = max(0.000001, genetic.planning_ms)
    faster = "A*" if astar_time < genetic_time else "Genético"
    ratio = max(astar_time, genetic_time) / min(astar_time, genetic_time)
    return ComparisonMetrics(
        astar=astar_metrics,
        genetic=genetic_metrics,
        optimality_gap_percent=optimality_gap,
        route_overlap_percent=overlap,
        product_order_agreement_percent=_agreement(astar_products, genetic_products),
        dock_order_agreement_percent=_agreement(astar_docks, genetic_docks),
        ga_reached_optimum=genetic_metrics.steps == astar_metrics.steps,
        faster_planner=faster,
        planning_speed_ratio=ratio,
    )
