#!/usr/bin/env python
from pathlib import Path
import subprocess


def main():
    # 設定ファイルが入っているディレクトリ
    configs_dir = Path("configs", "exp")  # 必要ならここを "settings" などに変更

    # 0〜99 の seed を回す
    seeds = range(10)

    # *.yml を全部取得（ソートしておくと順番が安定）
    config_files = sorted(configs_dir.glob("*.yml"))

    if not config_files:
        print(f"[WARN] No .yml files found in {configs_dir.resolve()}")
        return

    for cfg in config_files:
        for seed in seeds:
            cmd = [
                "uv",
                "run",
                "scripts/run_single_experiment.py",
                "--config",
                str(cfg),
                "--seed",
                str(seed),
                "--log-level",
                "ERROR"
            ]
            print(f"[INFO] Running: {' '.join(cmd)}")
            # エラー時に止めたい場合は check=True
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()