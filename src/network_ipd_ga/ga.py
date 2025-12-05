# ga.py
from __future__ import annotations
from typing import List, Dict
from collections import Counter
import random

import networkx as nx

from network_ipd_ga.agent import Agent
from network_ipd_ga.strategy import Strategy


def uniform_crossover(s1: Strategy, s2: Strategy, rng: random.Random) -> Strategy:
    """一様交叉：各ビットを 50% で親1/親2 からとる。"""
    b0 = s1[0] if rng.random() < 0.5 else s2[0]
    b1 = s1[1] if rng.random() < 0.5 else s2[1]
    b2 = s1[2] if rng.random() < 0.5 else s2[2]
    return (b0, b1, b2)


def mutate(strategy: Strategy, pm: float, rng: random.Random) -> Strategy:
    """ビット反転突然変異。各ビットを確率 pm で反転。"""
    b = list(strategy)
    for i in range(3):
        if rng.random() < pm:
            b[i] = 1 - b[i]
    return (b[0], b[1], b[2])


def _tournament_select(
    candidates: List[Agent],
    rng: random.Random,
    k: int = 3,
) -> Agent:
    """トーナメント選択：候補から k 個無作為に取り、その中で payoff 最大の個体を選ぶ。"""
    if len(candidates) == 1:
        return candidates[0]
    k = min(k, len(candidates))
    sampled = [rng.choice(candidates) for _ in range(k)]
    return max(sampled, key=lambda a: a.payoff)


def reproduce_population(
    agents: List[Agent],
    graph: nx.Graph,
    rng: random.Random,
    mutation_rate: float = 0.01,
    meta_influence: float = 0.3,
) -> None:
    """
    ネットワーク構造と GA に基づき、各エージェントの戦略を更新する。

    - 標準GA:
        各ノードは自分＋隣接ノードから親をトーナメント選択し、
        一様交叉＋突然変異により子戦略を生成。
    - メタ環境GA:
        上に加えて、現世代で最頻な戦略を「メタ戦略」とし、
        meta_influence の確率で子戦略とメタ戦略の交叉を行う。
    """
    id_to_agent: Dict[int, Agent] = {a.id: a for a in agents}

    # メタ戦略（最頻戦略）の算出
    meta_strategy: Strategy | None = None
    counter = Counter(a.strategy for a in agents)
    meta_strategy = counter.most_common(1)[0][0]

    new_strategies: Dict[int, Strategy] = {}

    for node in graph.nodes:
        agent = id_to_agent[node]
        neighbors = list(graph.neighbors(node))
        candidate_agents = [agent] + [id_to_agent[n] for n in neighbors]

        parent1 = _tournament_select(candidate_agents, rng)
        parent2 = _tournament_select(candidate_agents, rng)

        child = uniform_crossover(parent1.strategy, parent2.strategy, rng)
        child = mutate(child, mutation_rate, rng)

        if meta_strategy is not None and rng.random() < meta_influence:
            # メタ戦略との一様交叉を追加で行う
            child = uniform_crossover(child, meta_strategy, rng)

        new_strategies[node] = child

    # 新しい戦略をエージェントに反映
    for agent in agents:
        agent.strategy = new_strategies[agent.id]
