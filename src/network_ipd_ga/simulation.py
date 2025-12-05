# simulation.py
from __future__ import annotations
from typing import Literal, Tuple, List
import random
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import networkx as nx

from network_ipd_ga.network import make_cycle_graph, make_small_world_graph, make_scale_free_graph
from network_ipd_ga.agent import Agent
from network_ipd_ga.strategy import random_strategy, strategy_to_int, int_to_strategy
from network_ipd_ga.game import play_ipd
from network_ipd_ga.ga import reproduce_population
from network_ipd_ga.metrics import strategy_diversity_entropy, cooperation_rate_from_strategies
from collections import Counter


Topology = Literal["cycle", "small_world", "scale_free"]
ModelType = Literal["ga", "meta_ga"]


def _build_graph(
    topology: Topology,
    num_agents: int,
    seed: int,
    small_world_k: int,
    small_world_p: float,
    scale_free_m: int,
) -> nx.Graph:
    if topology == "cycle":
        return make_cycle_graph(num_agents)
    elif topology == "small_world":
        return make_small_world_graph(num_agents, k=small_world_k, p=small_world_p, seed=seed)
    elif topology == "scale_free":
        return make_scale_free_graph(num_agents, m=scale_free_m, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

def run_simulation(
    topology: Topology = "cycle",
    num_agents: int = 100,
    generations: int = 100,
    T: int = 50,
    mutation_rate: float = 0.01,
    small_world_k: int = 4,
    small_world_p: float = 0.1,
    scale_free_m: int = 2,
    seed: int = 0,
    meta_influence: float = 0.3,
) -> Tuple[pd.DataFrame, nx.Graph, List[Agent], pd.DataFrame]:
    """
    1つのネットワーク上で、指定したモデルタイプ（標準GA or メタ環境GA）
    による戦略進化をシミュレーションする。

    戻り値:
        df: 世代ごとの協力率・多様性などの時系列 DataFrame
        graph: 使用したネットワークグラフ
        agents: 最終世代のエージェントリスト
    """
    rng = random.Random(seed)

    graph = _build_graph(
        topology=topology,
        num_agents=num_agents,
        seed=seed,
        small_world_k=small_world_k,
        small_world_p=small_world_p,
        scale_free_m=scale_free_m,
    )

    # エージェント初期化
    agents: List[Agent] = [
        Agent(id=node, strategy=random_strategy(rng)) for node in graph.nodes
    ]

    history_records = []
    node_history: List[dict] = []

    for gen in range(generations):
        # 利得リセット
        for a in agents:
            a.reset_payoff()

        # 実際の対戦から協力率を測るためのカウンタ
        coop_actions_total = 0
        total_actions = 0

        # 各エッジで繰り返しゲームを実行
        id_to_agent = {a.id: a for a in agents}
        for i, j in graph.edges:
            ai = id_to_agent[i]
            aj = id_to_agent[j]
            coop, acts = play_ipd(ai, aj, T, rng)
            coop_actions_total += coop
            total_actions += acts

        realized_coop_rate = (
            coop_actions_total / total_actions if total_actions > 0 else 0.0
        )

        # 戦略分布から見た協調ポテンシャル
        strategy_coop_rate = cooperation_rate_from_strategies(agents)

        # 戦略多様性（エントロピー）
        diversity = strategy_diversity_entropy(agents)

        # --- 各戦略（000〜111）の個体数をカウント ---
        counter = Counter(strategy_to_int(a.strategy) for a in agents)
        strategy_counts = {}
        for i in range(8):
            bits = "".join(str(b) for b in int_to_strategy(i))
            strategy_counts[f"{bits}"] = counter.get(i, 0)

        history_records.append(
            {
                "generation": gen,
                "realized_coop_rate": realized_coop_rate,
                "strategy_coop_rate": strategy_coop_rate,
                "diversity": diversity,
                "avg_payoff": sum(a.payoff for a in agents) / len(agents),
                **strategy_counts,
            }
        )

        # --- 各ノードの戦略と利得を履歴に保存 ---
        for a in agents:
            node_history.append(
                {
                    "generation": gen,
                    "node_id": a.id,
                    "strategy_int": strategy_to_int(a.strategy),
                    "strategy_bits": "".join(str(b) for b in a.strategy),
                    "payoff": a.payoff,
                }
            )

        # 次世代の戦略を生成（GA + メタ環境）
        reproduce_population(
            agents=agents,
            graph=graph,
            rng=rng,
            mutation_rate=mutation_rate,
            meta_influence=meta_influence,
        )
        logger.info(
            f"Gen {gen}: coop(real={realized_coop_rate:.3f}, "
            f"strategy={strategy_coop_rate:.3f}), "
            f"div={diversity:.3f}, avg_payoff={history_records[-1]['avg_payoff']:.3f}"
        )

    df = pd.DataFrame(history_records)
    node_df = pd.DataFrame(node_history)

    return df, graph, agents, node_df
