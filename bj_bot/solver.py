# -*- coding: utf-8 -*-
"""着法求解器：枚举所有相邻交换，模拟消除与（确定性的）连锁下落，
按分值排序 —— 5 连 > L/T 形 > 4 连 > 3 连，连锁深度指数加权。
"""
from __future__ import annotations

import random

import numpy as np

from .board import EMPTY, HYPER, SIZE

# 基础分值（相对权重，非游戏真实分数）：刻意拉大 5 连与普通消除的差距
RUN_SCORE = {3: 10.0, 4: 45.0, 5: 320.0}
CASCADE_FACTOR = 1.8     # 每深一层连锁，权重乘 1.8
HYPER_BASE = 150.0       # 超立方体交换的基础分
HYPER_PER_GEM = 35.0     # 超立方体每消除一个同色宝石的加分


def _runs(board: np.ndarray):
    """返回 (mask, runs)。runs: [(cells, length, orient)]，orient: 'h'/'v'。"""
    mask = np.zeros((SIZE, SIZE), bool)
    runs = []
    for r in range(SIZE):
        c = 0
        while c < SIZE:
            v = board[r, c]
            if v >= 7:
                c += 1
                continue
            c2 = c
            while c2 + 1 < SIZE and board[r, c2 + 1] == v:
                c2 += 1
            if c2 - c + 1 >= 3:
                cells = [(r, cc) for cc in range(c, c2 + 1)]
                runs.append((cells, c2 - c + 1, "h"))
                for cc in range(c, c2 + 1):
                    mask[r, cc] = True
                c = c2 + 1
            else:
                c += 1
    for c in range(SIZE):
        r = 0
        while r < SIZE:
            v = board[r, c]
            if v >= 7:
                r += 1
                continue
            r2 = r
            while r2 + 1 < SIZE and board[r2 + 1, c] == v:
                r2 += 1
            if r2 - r + 1 >= 3:
                cells = [(rr, c) for rr in range(r, r2 + 1)]
                runs.append((cells, r2 - r + 1, "v"))
                for rr in range(r, r2 + 1):
                    mask[rr, c] = True
                r = r2 + 1
            else:
                r += 1
    return mask, runs


def _run_value(lengths, lt_cells: set) -> float:
    """一组同次消除的分值：考虑 5 连加成与 L/T 形加成。"""
    total = 0.0
    for ln in lengths:
        if ln >= 6:
            total += 700.0
        else:
            total += RUN_SCORE[ln]
    if lt_cells:                     # 横竖同时消 → L/T 形（星星宝石）
        total += 70.0 + 20.0 * len(lt_cells)
    return total


def _gravity(board: np.ndarray) -> np.ndarray:
    """每列非空下落，顶部补 EMPTY（不模拟随机补牌）。"""
    out = np.full_like(board, EMPTY)
    for c in range(SIZE):
        col = board[:, c]
        vals = col[col < 7]
        k = len(vals)
        if k:
            out[SIZE - k:, c] = vals
    return out


def simulate_swap(board: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> float | None:
    """模拟一次交换，返回该着法的估计分值；不产生消除则返回 None。"""
    a, b = board[r1, c1], board[r2, c2]
    # 超立方体：与任意相邻宝石交换都合法，清除全场该色
    if a == HYPER or b == HYPER:
        target = b if a == HYPER else a
        if target == HYPER:
            cleared = int(((board >= 0) & (board < 7)).sum())
        else:
            cleared = int((board == target).sum())
        return HYPER_BASE + HYPER_PER_GEM * cleared

    g = board.copy()
    g[r1, c1], g[r2, c2] = g[r2, c2], g[r1, c1]
    mask, runs = _runs(g)
    if not runs:
        return None

    score = 0.0
    depth = 0
    while runs:
        lengths = [ln for _, ln, _ in runs]
        # L/T 交叉格
        hcells = set()
        vcells = set()
        for cells, _, orient in runs:
            (hcells if orient == "h" else vcells).update(cells)
        lt = (hcells & vcells)
        score += _run_value(lengths, lt) * (CASCADE_FACTOR ** depth)
        g[mask] = EMPTY
        g = _gravity(g)
        depth += 1
        mask, runs = _runs(g)
        if depth > 8:
            break
    return score


def best_move(board: np.ndarray, exclude=None) -> tuple[float, tuple[int, int, int, int]] | None:
    """返回 (分值, (r1,c1,r2,c2))；无着法时 None。exclude: 屏蔽已失败着法。"""
    exclude = exclude or set()
    cands = []
    for r in range(SIZE):
        for c in range(SIZE):
            v = board[r, c]
            if v == EMPTY:          # 未知格不可操作；超立方体(HYPER)可以
                continue
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if r2 >= SIZE or c2 >= SIZE:
                    continue
                if (r, c, r2, c2) in exclude:
                    continue
                v2 = board[r2, c2]
                if v2 == EMPTY:
                    continue
                if v == v2 and v != HYPER:
                    continue        # 同色普通交换必无消除（超立方体除外）
                s = simulate_swap(board, r, c, r2, c2)
                if s is not None:
                    cands.append((s, (r, c, r2, c2)))
    if not cands:
        return None
    cands.sort(key=lambda t: -t[0])
    top = cands[0][0]
    ties = [m for s, m in cands if s >= top - 1e-6]
    return top, random.choice(ties)
