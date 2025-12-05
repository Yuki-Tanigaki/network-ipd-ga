from pathlib import Path
import copy
import yaml

# ---- ベースとなる設定（質問に書いてくれた内容をそのまま） ----
BASE_CONFIG = {
    "num_agents": 1000,
    "generations": 100,
    "T": 50,
    "mutation_rate": 0.01,

    # 上書き対象：
    # "topology": "small_world",
    # "meta_influence": 0.0,

    # Network parameters
    "small_world_k": 4,
    "small_world_p": 0.1,
    "scale_free_m": 2,

    # Logging / output
    "logs_dir": "results/logs",
    "output_dir": "results",
    # "output_base": "smallworld_meta00",  # ここも後で上書き
}

# topology の3種
TOPOLOGIES = ["cycle", "small_world", "scale_free"]

# meta_influence を 0.0 から 0.2刻みで 1.0 まで
META_VALUES = [i * 0.2 for i in range(6)]  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def meta_suffix(value: float) -> str:
    """
    0.0 → "00", 0.1 → "01", ..., 1.0 → "10"
    という2桁の文字列を返す。
    """
    # 0.0〜1.0 を 0〜10 にマッピングし、2桁ゼロ埋め
    return f"{int(round(value * 10)) :02d}"


def main():
    # 出力先ディレクトリ（必要に応じて変更）
    out_dir = Path("configs", "exp")
    out_dir.mkdir(parents=True, exist_ok=True)

    for topo in TOPOLOGIES:
        for m in META_VALUES:
            suffix = meta_suffix(m)
            base_name = f"{topo}_meta{suffix}"      
            filename = out_dir / f"{base_name}.yml"

            cfg = copy.deepcopy(BASE_CONFIG)
            cfg["topology"] = topo
            cfg["meta_influence"] = float(f"{m:.1f}")  # 0.2 などの float に
            cfg["output_base"] = base_name

            with filename.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

            print(f"Generated: {filename}")


if __name__ == "__main__":
    main()