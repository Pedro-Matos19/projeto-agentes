from __future__ import annotations

import random

from modelos import Cell, DockSlot, Product, Scenario


GA_SEED_MASK = 0x5F3759DF

"cria a doca, o labirinto, os produtos e o ponto inicial"
class MazeGrid:

    def __init__(self, rows: int, warehouse_cols: int, dock_width: int) -> None:
        self.rows = rows
        self.warehouse_cols = warehouse_cols
        self.dock_width = dock_width
        self.total_cols = warehouse_cols + dock_width
        self._passages: dict[Cell, set[Cell]] = {
            (row, col): set()
            for row in range(rows)
            for col in range(self.total_cols)
        }

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple(self._passages)

    def contains(self, cell: Cell) -> bool:
        return cell in self._passages

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        return tuple(sorted(self._passages.get(cell, ())))

    def connected(self, first: Cell, second: Cell) -> bool:
        return second in self._passages.get(first, ())

    def connect(self, first: Cell, second: Cell) -> None:
        if first not in self._passages or second not in self._passages:
            raise ValueError(f"Passagem fora da grade: {first} -> {second}")
        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            raise ValueError(f"Células não adjacentes: {first} -> {second}")
        self._passages[first].add(second)
        self._passages[second].add(first)


def _dock_width(warehouse_cols: int) -> int:
    return max(4, min(7, warehouse_cols // 9))


def _spaced_rows(count: int, rows: int) -> list[int]:
    candidates = list(range(1, rows - 1))
    if count > len(candidates):
        raise ValueError("Não há linhas suficientes para todas as vagas de doca.")
    return [candidates[round(index * (len(candidates) - 1) / max(count - 1, 1))] for index in range(count)]


def _build_dock(maze: MazeGrid, product_count: int) -> tuple[Cell, list[DockSlot]]:
    for row in range(maze.rows):
        for col in range(maze.dock_width):
            if row + 1 < maze.rows:
                maze.connect((row, col), (row + 1, col))
            if col + 1 < maze.dock_width:
                maze.connect((row, col), (row, col + 1))

    gate_rows = _spaced_rows(product_count, maze.rows)
    slots = []
    slot_col = max(1, maze.dock_width - 2)
    for slot_id, row in enumerate(gate_rows, start=1):
        maze.connect((row, maze.dock_width - 1), (row, maze.dock_width))
        slots.append(DockSlot(slot_id, (row, slot_col)))

    start = (maze.rows // 2, max(1, maze.dock_width // 2))
    return start, slots


def _generate_warehouse(maze: MazeGrid, rng: random.Random, loop_percent: float) -> None:
    start = (rng.randrange(maze.rows), maze.dock_width)
    visited = {start}
    stack = [start]

    while stack:
        row, col = stack[-1]
        candidates = []
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + d_row, col + d_col)
            if (
                0 <= neighbor[0] < maze.rows
                and maze.dock_width <= neighbor[1] < maze.total_cols
                and neighbor not in visited
            ):
                candidates.append(neighbor)

        if not candidates:
            stack.pop()
            continue

        next_cell = rng.choice(candidates)
        maze.connect(stack[-1], next_cell)
        visited.add(next_cell)
        stack.append(next_cell)

    closed_edges: list[tuple[Cell, Cell]] = []
    for row in range(maze.rows):
        for col in range(maze.dock_width, maze.total_cols):
            cell = (row, col)
            for neighbor in ((row + 1, col), (row, col + 1)):
                if (
                    neighbor[0] < maze.rows
                    and neighbor[1] < maze.total_cols
                    and not maze.connected(cell, neighbor)
                ):
                    closed_edges.append((cell, neighbor))

    rng.shuffle(closed_edges)
    extra_openings = int(maze.rows * maze.warehouse_cols * loop_percent)
    for first, second in closed_edges[:extra_openings]:
        maze.connect(first, second)


def _place_products(
    maze: MazeGrid,
    count: int,
    rng: random.Random,
) -> tuple[Product, ...]:
    minimum_col = maze.dock_width + max(2, maze.warehouse_cols // 5)
    candidates = [
        (row, col)
        for row in range(1, maze.rows - 1)
        for col in range(minimum_col, maze.total_cols - 1)
    ]
    rng.shuffle(candidates)
    minimum_spacing = max(2, min(maze.rows, maze.warehouse_cols) // 10)
    selected: list[Cell] = []

    for candidate in candidates:
        if all(
            abs(candidate[0] - other[0]) + abs(candidate[1] - other[1]) >= minimum_spacing
            for other in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break

    if len(selected) < count:
        remaining = [cell for cell in candidates if cell not in selected]
        selected.extend(remaining[: count - len(selected)])

    if len(selected) != count:
        raise ValueError("Não foi possível posicionar todos os produtos.")
    return tuple(Product(index, cell) for index, cell in enumerate(selected, start=1))


def generate_scenario(
    rows: int = 35,
    cols: int = 55,
    product_count: int = 8,
    seed: int = 20260721,
    loop_percent: float = 0.08,
) -> Scenario:

    rng = random.Random(seed)
    maze = MazeGrid(rows, cols, _dock_width(cols))
    _generate_warehouse(maze, rng, loop_percent)
    start, dock_slots = _build_dock(maze, product_count)
    return Scenario(
        maze,
        start,
        _place_products(maze, product_count, rng),
        tuple(dock_slots),
        seed,
        seed ^ GA_SEED_MASK,
    )
