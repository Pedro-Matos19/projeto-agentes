from __future__ import annotations

import argparse

from interface import WarehouseApp


"parse -> lê as opções passadas pelo terminal"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulador visual de coleta em armazém usando labirinto."
    )
    parser.add_argument(
        "--size",
        choices=("standard", "large"),
        default="standard",
        help="define o tamanho do tabuleiro -> standard usa 35x55; large usa 100x100.",
    )
    parser.add_argument("--products", type=int, default=8, choices=range(4, 9))
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


"aqui na main essas opções vem do parse e viram variáveis reais"
def main() -> None:
    args = parse_args()
    rows, cols = (100, 100) if args.size == "large" else (35, 55)
    app = WarehouseApp(
        rows=rows,
        cols=cols,
        product_count=args.products,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
