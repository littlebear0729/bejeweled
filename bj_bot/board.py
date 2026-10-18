# -*- coding: utf-8 -*-
"""棋盘识别：从窗口截图中读出 8x8 宝石矩阵。

颜色编码（与像素分类一致）:
    0=R 红  1=O 橙  2=Y 黄  3=G 绿  4=B 蓝  5=P 紫  6=W 白
    7='.' 未知/动画中    8=超立方体（彩虹宝石）
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = 8
EMPTY = 7
HYPER = 8
LETTERS = "ROYGBPW."
GEM_NAMES = ["红", "橙", "黄", "绿", "蓝", "紫", "白"]


def classify_pixels(img_bgr: np.ndarray) -> np.ndarray:
    """逐像素分类，返回 letter 索引图（H, W），uint8。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    letter = np.full(h_.shape, EMPTY, np.uint8)
    white = (s_ < 55) & (v_ >= 55)
    letter[white] = 6                                    # W
    colored = (~white) & (v_ >= 55)
    hh = h_
    letter[colored & ((hh < 8) | (hh >= 172))] = 0       # R
    letter[colored & (hh >= 8) & (hh < 22)] = 1          # O
    letter[colored & (hh >= 22) & (hh < 34)] = 2         # Y
    letter[colored & (hh >= 34) & (hh < 90)] = 3         # G
    letter[colored & (hh >= 90) & (hh < 130)] = 4        # B
    letter[colored & (hh >= 130) & (hh < 172)] = 5       # P
    return letter


def read_board(img_bgr: np.ndarray, cell: int, ox: int, oy: int) -> tuple[np.ndarray, np.ndarray]:
    """读取 8x8 棋盘。

    返回 (board, conf):
        board: 8x8 uint8，元素为 0-8 编码
        conf:  8x8 float，每个格子的多数派占比（0~1）
    """
    lt = classify_pixels(img_bgr)
    h = max(4, cell // 5)
    board = np.full((SIZE, SIZE), EMPTY, np.uint8)
    conf = np.zeros((SIZE, SIZE), np.float32)
    for r in range(SIZE):
        for c in range(SIZE):
            cy = int(oy + (r + 0.5) * cell)
            cx = int(ox + (c + 0.5) * cell)
            patch = lt[cy - h:cy + h + 1, cx - h:cx + h + 1].ravel()
            if patch.size == 0:
                continue
            counts = np.bincount(patch, minlength=8)
            idx = int(counts.argmax())
            frac = counts[idx] / patch.size
            conf[r, c] = frac
            # 多数派占比过低 → 动画/遮挡/半透明覆盖
            if frac < 0.45:
                board[r, c] = EMPTY
                continue
            if idx == EMPTY:
                board[r, c] = EMPTY
                continue
            if idx == 6:  # 白色：可能是白钻石，也可能是超立方体（白球+彩色光环）
                colored_share = counts[:6].sum() / patch.size
                # 白钻只有零星彩色闪光；超立方体的彩色环占比大
                if colored_share >= 0.22:
                    board[r, c] = HYPER
                    continue
                board[r, c] = 6
                continue
            board[r, c] = idx
    return board, conf


def board_is_settled(board: np.ndarray) -> bool:
    """落定棋盘：无未知格、无已存在的三连。"""
    if (board == EMPTY).any():
        return False
    b = board
    for r in range(SIZE):
        for c in range(SIZE - 2):
            if b[r, c] == b[r, c + 1] == b[r, c + 2] and b[r, c] not in (EMPTY, HYPER):
                return False
    for c in range(SIZE):
        for r in range(SIZE - 2):
            if b[r, c] == b[r + 1, c] == b[r + 2, c] and b[r, c] not in (EMPTY, HYPER):
                return False
    return True


def pretty(board: np.ndarray) -> str:
    return "\n".join(" ".join(LETTERS[v] for v in row) for row in board)
