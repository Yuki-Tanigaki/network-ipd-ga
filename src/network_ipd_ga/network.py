# network.py
from __future__ import annotations
import networkx as nx


def make_cycle_graph(num_agents: int) -> nx.Graph:
    """
    1次元（環状）のネットワークを生成する。
    各ノードは左右1つずつ、計2つの隣接ノードを持つ。
    """
    return nx.cycle_graph(num_agents)


def make_small_world_graph(
    num_agents: int,
    k: int = 4,
    p: float = 0.1,
    seed: int | None = None,
) -> nx.Graph:
    """
    Watts-Strogatz 型の小世界ネットワークを生成する。
    num_agents: ノード数
    k: 1ノードあたりの最近傍ノード数（偶数推奨）
    p: 再結線確率
    """
    if k >= num_agents:
        raise ValueError("k must be smaller than num_agents.")
    if k % 2 == 1:
        k -= 1  # 偶数にそろえる

    return nx.watts_strogatz_graph(num_agents, k, p, seed=seed)


def make_scale_free_graph(
    num_agents: int,
    m: int = 2,
    seed: int | None = None,
) -> nx.Graph:
    """
    Barabási-Albert 型のスケールフリーネットワークを生成する。
    num_agents: ノード数
    m: 1ノード追加時に付与するエッジ数
    """
    if m < 1:
        raise ValueError("m must be >= 1.")
    if m >= num_agents:
        raise ValueError("m must be smaller than num_agents.")

    return nx.barabasi_albert_graph(num_agents, m, seed=seed)
