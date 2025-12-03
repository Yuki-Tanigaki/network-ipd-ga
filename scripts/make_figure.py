import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yaml

COLOR_MAP = {
    "000": "black",
    "001": "tab:blue",
    "010": "tab:orange",
    "011": "tab:green",
    "100": "tab:red",
    "101": "tab:purple",
    "110": "tab:brown",
    "111": "tab:pink",
}

def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load result CSV automatically from config.yml + seed"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (same as run_single_experiment).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed of the experiment to visualize.",
    )
    return parser.parse_args()

def save_metric_plot(df, x, y, save_path: Path):
    """1つの指標について折れ線グラフを作り保存"""
    plt.figure(figsize=(8, 5))
    plt.plot(df[x], df[y], label=y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y} over {x}")
    plt.grid(True)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] {save_path}")

def save_strategy_distribution_plot(df: pd.DataFrame, save_path: Path):
    """
    8戦略(000〜111)の割合推移を1枚のグラフに描画して保存
    """
    strategy_cols = ["000", "001", "010", "011", "100", "101", "110", "111"]

    # 世代ごとの合計（エージェント数）で割って割合に
    total = df[strategy_cols].sum(axis=1)

    plt.figure(figsize=(10, 6))
    for s in strategy_cols:
        proportion = df[s] / total
        plt.plot(
            df["generation"],
            proportion,
            label=s,
            color=COLOR_MAP.get(s, None),  # 保険で get を使う
        )

    plt.xlabel("Generation")
    plt.ylabel("Proportion")
    plt.title("Strategy Distribution Over Generations")
    plt.grid(True)
    plt.legend(title="Strategy")
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] {save_path}")

def main():
    args = parse_args()
    cfg = load_config(Path(args.config))

    output_dir = Path(cfg["output_dir"])
    output_base = cfg["output_base"]
    seed = args.seed

    # --- CSV ファイルパス決定 ---
    csv_name = f"{output_base}_seed{seed}.csv"
    csv_path = output_dir / "csvs" / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"[INFO] Loading: {csv_path}")

    # --- CSV 読み込み ---
    df = pd.read_csv(csv_path)

    # --- 保存先: output_dir / figs / ... ---
    figs_dir = output_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # --- 個別のメトリクス ---
    metrics = [
        "realized_coop_rate",
        "strategy_coop_rate",
        "diversity",
        "avg_payoff",
    ]
    for m in metrics:
        save_path = figs_dir / f"{output_base}_seed{seed}_{m}.png"
        save_metric_plot(df, "generation", m, save_path)

    # --- 8戦略の割合推移グラフ ---
    strat_fig_path = figs_dir / f"{output_base}_seed{seed}_strategy_distribution.png"
    save_strategy_distribution_plot(df, strat_fig_path)


if __name__ == "__main__":
    main()