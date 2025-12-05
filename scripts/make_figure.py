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

STRATEGY_COLS = ["000", "001", "010", "011", "100", "101", "110", "111"]

def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def parse_seed_range(s: str) -> list[int]:
    """
    seed 指定文字列をパースする:
        "0-9"      -> [0,1,2,3,4,5,6,7,8,9]
        "0,3,7"    -> [0,3,7]
        "0-3,7-9"  -> [0,1,2,3,7,8,9]
    """
    result = []
    parts = s.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            start, end = part.split("-")
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))

    # 重複排除 & ソート
    return sorted(set(result))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load multiple seeds and compute averaged metrics."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--seeds",
        type=str,
        required=True,
        help="Seed range, e.g., '0-9' or '0,2,5-7'"
    )
    return parser.parse_args()

# -------------------------------------------------------------

def load_all_seed_data(cfg, seeds):
    """複数seedのCSVを読み込み、1つのDataFrameに縦結合する."""
    dfs = []

    for seed in seeds:
        csv_name = f"{cfg['output_base']}_seed{seed:04d}.csv"
        csv_path = Path(cfg["output_dir"]) / "csvs" / csv_name

        if not csv_path.exists():
            print(f"[WARN] Not found: {csv_path}")
            continue

        print(f"[LOAD] {csv_path}")
        df = pd.read_csv(csv_path)
        df["seed"] = seed  # どのシードか識別したいときに使える
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSV files loaded.")

    return pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------

def aggregate_over_seeds(df_all: pd.DataFrame):
    """
    世代ごとに「平均」と「標準偏差」の DataFrame (メトリクス用) を返す.
    戻り値:
        df_mean, df_std
    """
    df_num = df_all.drop(columns=["seed"], errors="ignore")
    grouped = df_num.groupby("generation")
    df_mean = grouped.mean(numeric_only=True).reset_index()
    df_std  = grouped.std(ddof=0, numeric_only=True).reset_index()
    return df_mean, df_std

def aggregate_strategy_proportions(df_all: pd.DataFrame):
    """
    戦略比率に対して、世代ごとの「平均」と「標準偏差」を計算する.

    各行（1世代・1シード）について：
        p_s = count_s / sum(count_all_strategies)
    を計算し、その p_s を seed 方向に平均・標準偏差.
    """
    # 戦略列がなければ何もしない
    missing = [c for c in STRATEGY_COLS if c not in df_all.columns]
    if missing:
        print(f"[WARN] Strategy columns not found in df_all: {missing}. "
              f"Skip strategy proportion aggregation.")
        return None, None

    # 各世代・各シードごとの「戦略比率」を計算
    cols = ["generation", "seed"] + STRATEGY_COLS
    df_prop = df_all[cols].copy()

    total = df_prop[STRATEGY_COLS].sum(axis=1)
    # total == 0 の行は NaN になるので、そのまま groupby.mean で無視される
    for s in STRATEGY_COLS:
        df_prop[s] = df_prop[s] / total

    grouped = df_prop.groupby("generation")
    df_prop_mean = grouped.mean(numeric_only=True).reset_index()
    df_prop_std  = grouped.std(ddof=0, numeric_only=True).reset_index()

    return df_prop_mean, df_prop_std

# -------------------------------------------------------------

def save_metric_plot(df_mean, df_std, x, y, save_path: Path):
    """
    各世代の平均に対して、±1σ の帯を付けた折れ線グラフを保存.
    """
    plt.figure(figsize=(8, 5))

    x_vals = df_mean[x]
    y_mean = df_mean[y]
    y_std  = df_std[y]

    # 平均の折れ線
    plt.plot(x_vals, y_mean, label=f"avg_{y}")

    # ±1σ の帯
    upper = y_mean + y_std
    lower = y_mean - y_std
    plt.fill_between(x_vals, lower, upper, alpha=0.3, label="±1 std")

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"Average {y} over {x} (with std band)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] {save_path}")

def save_strategy_distribution_plot_with_band(
    df_prop_mean: pd.DataFrame,
    df_prop_std: pd.DataFrame,
    save_path: Path,
):
    """
    戦略ごとの比率の平均と ±1σ の帯を描画する.
    """
    if df_prop_mean is None or df_prop_std is None:
        print("[INFO] Skip strategy distribution plot (no data).")
        return

    plt.figure(figsize=(10, 6))

    x_vals = df_prop_mean["generation"]

    for s in STRATEGY_COLS:
        if s not in df_prop_mean.columns:
            continue

        mean = df_prop_mean[s]
        std  = df_prop_std[s]

        # 平均の線
        plt.plot(x_vals, mean, color=COLOR_MAP[s], label=s)

        # ±1σ の帯
        upper = mean + std
        lower = mean - std
        plt.fill_between(x_vals, lower, upper,
                         alpha=0.15, color=COLOR_MAP[s])

    plt.xlabel("Generation")
    plt.ylabel("Proportion")
    plt.title("Average Strategy Distribution (with std bands)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVE] {save_path}")

# -------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    seeds = parse_seed_range(args.seeds)
    print("[INFO] Seeds to load:", seeds)

    # 全シードのデータ読み込み
    df_all = load_all_seed_data(cfg, seeds)

    # 1) メトリクスの平均 & 標準偏差
    df_mean, df_std = aggregate_over_seeds(df_all)

    # 2) 戦略比率の平均 & 標準偏差
    df_prop_mean, df_prop_std = aggregate_strategy_proportions(df_all)

    # --- 出力先 ---
    figs_dir = Path(cfg["output_dir"]) / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # メトリクス各種を平均＋分散帯付きで保存
    metrics = ["realized_coop_rate", "strategy_coop_rate", "diversity", "avg_payoff"]

    for m in metrics:
        p = figs_dir / f"{cfg['output_base']}_avg_{m}_with_std.png"
        save_metric_plot(df_mean, df_std, "generation", m, p)

    # 戦略分布の平均＋帯
    p = figs_dir / f"{cfg['output_base']}_avg_strategy_distribution_with_std.png"
    save_strategy_distribution_plot_with_band(df_prop_mean, df_prop_std, p)


if __name__ == "__main__":
    main()
