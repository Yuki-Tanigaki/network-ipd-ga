# strategy.py
from __future__ import annotations
from typing import Tuple
import random

# 行動の定義：1 = 協調(C), 0 = 裏切り(D)
C = 1
D = 0

Strategy = Tuple[int, int, int]  # (b0, b1, b2)


def random_strategy(rng: random.Random) -> Strategy:
    """ランダムに 3ビット戦略を生成する。"""
    return tuple(rng.randint(0, 1) for _ in range(3))  # type: ignore[return-value]


def decide_action(strategy: Strategy, round_index: int, opponent_prev_action: int | None) -> int:
    """
    3ビット戦略に従って行動を決める。

    strategy = (b0, b1, b2)
    - t == 0: b0 を用いる
    - t >= 1: 直前の相手行動に応じて
        - 相手が D(0) のとき -> b1
        - 相手が C(1) のとき -> b2
    """
    b0, b1, b2 = strategy

    if round_index == 0 or opponent_prev_action is None:
        return b0

    if opponent_prev_action == D:
        return b1
    else:
        return b2


def strategy_to_int(strategy: Strategy) -> int:
    """(b0, b1, b2) を 0〜7 の整数に変換。"""
    b0, b1, b2 = strategy
    return (b0 << 2) | (b1 << 1) | b2


def int_to_strategy(x: int) -> Strategy:
    """0〜7 の整数を 3ビット戦略に変換。"""
    if not (0 <= x < 8):
        raise ValueError("Strategy index must be between 0 and 7.")
    b0 = (x >> 2) & 1
    b1 = (x >> 1) & 1
    b2 = x & 1
    return (b0, b1, b2)


def cooperation_potential(strategy: Strategy) -> float:
    """
    戦略そのものが持つ「協調傾向」を簡単に定義。
    3ビット中の1の割合を協調ポテンシャルとみなす。
    """
    return sum(strategy) / 3.0
