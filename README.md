# network-ipd-ga
**Network-structured Iterated Prisoner’s Dilemma (IPD) with Genetic Algorithm (GA) Evolution**  
本プロジェクトは、格子・小世界・スケールフリーといった **ネットワーク構造上**で、 エージェントの **3ビット戦略**を遺伝的アルゴリズムで進化させるシミュレータです。 
また、世代内の最頻戦略を外部環境として利用する **Meta-GA モデル**にも対応しています。  
開始 → 対戦 → 再生産（交叉・突然変異） → 次世代 を繰り返し、協力率・戦略多様性・平均利得の推移を解析できます。

# Installation 
Ubuntuでの実行を想定
## Step 0 — Install **uv** and **ffmpeg**
```bash 
sudo apt update
sudo apt install uv 
sudo apt install ffmpeg
``` 
または公式インストール： 
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh 
``` 
確認： 
```bash 
uv --version 
```

## Step 1 — Clone project 
```bash 
git clone https://github.com/yourname/network-ipd-ga.git 
cd network-ipd-ga 
```
## Step 2 — Create virtual environment & install dependencies
```bash 
uv venv source .venv/bin/activate uv sync 
```

# How to Run Experiments 
`scripts/run_single_experiment.py` を使って実行します。 
## Basic usage 
```bash 
uv run scripts/run_single_experiment.py --config configs/sample.yml --seed 42 
```
実行後、結果 CSV が `results/csvs/` に保存されます。

# Configuration (sample.yml) 
`configs/sample.yml` はシミュレーションの全パラメータを記述する YAML ファイルです。 
例：
```yaml 
num_agents: 100 
generations: 200 
T: 50 
mutation_rate: 0.01 
topology: "small_world" 
small_world_k: 4 
small_world_p: 0.1 
scale_free_m: 2 
meta_influence: 0.3 
logs_dir: "results/logs" 
output_dir: "results/csvs" 
output_base: "smallworld_meta00"
```
## Parameter description 
| パラメータ | 説明 |
|-----------|------|
| **num_agents** | ネットワークのノード数（エージェント数） |
| **generations** | GA の進化世代数 |
| **T** | IPD の繰り返しラウンド数 |
| **mutation_rate** | 突然変異率（ビット反転確率） |
| **topology** | ネットワーク構造（`lattice` / `small_world` / `scale_free`） |
| **small_world_k, small_world_p** | Watts–Strogatz 小世界ネットワークのパラメータ |
| **scale_free_m** | Barabási–Albert スケールフリーの接続数 |
| **meta_influence** | Meta-GA：最頻戦略と交叉する確率 |
| **logs_dir, output_dir, output_base** | 出力ディレクトリ, ファイル名 |

# Output

実行後、`results/csvs/` に CSV が生成されます。

| 列名                   | 意味                                                       | プログラム                                    |
| -------------------- | -------------------------------------------------------- | ------------------------------------- |
| `generation`         | 世代番号（0,1,2,…）                                            | `history_records.append`より            |
| `realized_coop_rate` | **実際の対戦結果から測定した協調率**（全エッジで T ラウンド対戦した結果、協調(C)の回数 / 行動総数） | `play_ipd()` の結果を集計                   |
| `strategy_coop_rate` | **戦略ビット列の 1 の割合**（全ノードの戦略を集計した協力傾向）                             | `cooperation_rate_from_strategies()`  |
| `diversity`          | **戦略分布のシャノンエントロピー**（多様性指標）                               | `strategy_diversity_entropy()`        |
| `avg_payoff`         | **エージェントの平均利得**（その世代での対戦後の payoff の平均）                   | `sum(a.payoff)/len(agents)`           |
| 000         | **各戦略数**（その世代において該当する戦略を持った個体の数）                   | `strategy_counts`で集計           |

# Make Figure
統計データのグラフ化
```bash 
uv run scripts/make_figure.py --config configs/sample.yml --seeds 0-9
```

ノードの戦略変化を動画化
```bash 
uv run scripts/make_video.py --config configs/sample.yml --seed 42
```