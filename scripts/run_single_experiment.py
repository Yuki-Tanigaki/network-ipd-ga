# scripts/run_single_experiment.py
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pickle
from typing import List

from network_ipd_ga.config_loader import load_config
from network_ipd_ga.simulation import run_simulation

# ---------------------------------------
# ログレベル文字列を logging レベルに変換
# ---------------------------------------
def to_loglevel(level_str: str) -> int:
    level_str = level_str.upper()
    if level_str in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        return getattr(logging, level_str)
    raise ValueError(f"Invalid log level: {level_str}")

def setup_logging(output_dir: Path, output_base: str, seed: int, enable_file: bool, log_level: int) -> None:
    """
    ログ設定：
    - 標準出力への出力は常に有効
    - enable_file=True のときだけファイルにも出力
    """
    logger = logging.getLogger()  # root logger
    logger.setLevel(log_level)

    # 既存ハンドラをクリア（basicConfig 依存を避ける）
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # --- 標準出力ハンドラ ---
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if enable_file:
        log_dir = output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{output_base}_seed{seed:04d}.log"
        log_file = log_dir / fname

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log will be saved to: {log_file}")
    else:
        logger.info("File logging disabled.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single IPD evolution experiment using a YAML config and a seed."
    )

    # 設定ファイルのパス
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    # 乱数シード（必須）
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for the simulation.",
    )

    # ログファイル出力 ON/OFF を選択
    parser.add_argument(
        "--log-file",
        action="store_true",
        help="If set, save logs to a file (default: only stdout).",
    )

    # ログレベル指定 (DEBUG, INFO, WARNING, ERROR)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 設定ファイル読み込み
    cfg = load_config(Path(args.config))

    # シミュレーション実行
    df, graph, agents, node_df = run_simulation(
        topology=cfg.topology,
        num_agents=cfg.num_agents,
        generations=cfg.generations,
        T=cfg.T,
        mutation_rate=cfg.mutation_rate,
        small_world_k=cfg.small_world_k,
        small_world_p=cfg.small_world_p,
        scale_free_m=cfg.scale_free_m,
        meta_influence=cfg.meta_influence,
        seed=args.seed,
    )

    # メイン出力ファイル名決定（世代サマリ）
    summary_fname = f"{cfg.output_base}_seed{args.seed:04d}.csv"
    summary_path = cfg.output_dir / "csvs" / summary_fname

    # ノード履歴とグラフのファイル名
    node_fname = f"{cfg.output_base}_seed{args.seed:04d}_nodes.csv"
    node_path = cfg.output_dir / "csvs" / node_fname

    graph_fname = f"{cfg.output_base}_seed{args.seed:04d}_graph.pickle"
    graph_path = cfg.output_dir / "pickles" / graph_fname

    # 親ディレクトリ作成
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    # CSV 保存（世代サマリ）
    df.to_csv(summary_path, index=False)

    # CSV 保存（ノード履歴）
    node_df.to_csv(node_path, index=False)

    # グラフ保存（ネットワーク構造）
    with graph_path.open("wb") as f:
        pickle.dump(graph, f)

    logging.info(f"Saved summary to: {summary_path.resolve()}")
    logging.info(f"Saved node history to: {node_path.resolve()}")
    logging.info(f"Saved graph to: {graph_path.resolve()}")

    for k, v in cfg.as_dict().items():
        print(f"{k}: {v}")
    print(df.head())


if __name__ == "__main__":
    # ロギング初期化
    args = parse_args()
    cfg = load_config(Path(args.config))

    # ログレベルを整数値に変換
    log_level = to_loglevel(args.log_level)

    setup_logging(
        Path(cfg.output_dir),
        cfg.output_base,
        args.seed,
        enable_file=args.log_file,
        log_level=log_level,
    )
    logger = logging.getLogger(__name__)
    logger.info("===== Simulation Config =====")
    for k, v in cfg.as_dict().items():
        logger.info(f"{k}: {v}")
    logger.info("=================================")

    main()
