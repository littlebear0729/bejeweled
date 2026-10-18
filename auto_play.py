# -*- coding: utf-8 -*-
"""Bejeweled 3 自动玩家 —— 命令行入口。

常用命令：
    python auto_play.py                 # 使用 config.json 开始自动玩
    python auto_play.py --calibrate     # 重新校准棋盘（窗口移动/缩放后）
    python auto_play.py --rounds 20     # 只玩 20 步
    python auto_play.py --dry           # 演练：只打印棋盘和着法，不点击
    python auto_play.py --selftest      # 求解器单元自测

紧急停止：按住 F8，或 Ctrl+C。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from bj_bot import winio
from bj_bot.board import HYPER, pretty
from bj_bot.calibrate import calibrate
from bj_bot.player import Bot
from bj_bot.solver import _gravity, _runs, best_move, simulate_swap

CONFIG = "config.json"


def _configure_console() -> None:
    """避免 Windows GBK 控制台遇到箭头等字符时中断程序。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            # 保留控制台当前编码，仅让无法编码的字符降级为替代字符。
            reconfigure(errors="replace")
        except (OSError, ValueError):
            pass


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"未找到 {path}，请先运行: python auto_play.py --calibrate")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("cell", "ox", "oy", "win_w", "win_h")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise SystemExit(
            f"{path} 缺少校准字段 {', '.join(missing)}，"
            "请重新运行: python auto_play.py --calibrate"
        )
    try:
        cell = int(cfg["cell"])
        win_w = int(cfg["win_w"])
        win_h = int(cfg["win_h"])
        int(cfg["ox"]), int(cfg["oy"])
    except (TypeError, ValueError):
        raise SystemExit(f"{path} 的校准数据格式无效，请重新校准") from None
    if cell <= 0 or win_w <= 0 or win_h <= 0 or 8 * cell > min(win_w, win_h):
        raise SystemExit(f"{path} 的棋盘尺寸不合理，请重新校准")
    return cfg


# ---------------------------------------------------------------- 自测 --
def _mk_board(fn) -> np.ndarray:
    b = np.zeros((8, 8), np.uint8)
    for r in range(8):
        for c in range(8):
            b[r, c] = (3 * r + 5 * c) % 6   # 无三连的底版
    for r in range(8):
        for c in range(8):
            b[r, c] = fn(r, c, b[r, c])
    return b


def _no_runs(b: np.ndarray) -> bool:
    mask, runs = _runs(b)
    return len(runs) == 0


def selftest() -> int:
    ok = True

    # 1) 5 连优先：行4 为 G G ? G G，缺口的 G 在 (3,3)，交换补成五连
    def five(r, c, v):
        if r == 4 and 1 <= c <= 5:
            return 3 if c != 3 else 2      # G G Y G G
        if (r, c) == (3, 3):
            return 3                        # 待换入的 G
        return v
    b = _mk_board(five)
    assert _no_runs(b), "底版不应存在三连"
    res = best_move(b)
    assert res is not None, "应存在可行着法"
    score, (r1, c1, r2, c2) = res
    five_score = simulate_swap(b, 3, 3, 4, 3)
    print(f"[1] 五连场景: 最优分={score:.0f}, 五连着法分={five_score:.0f}, "
          f"最优着法=({r1},{c1})<->({r2},{c2})")
    ok &= five_score is not None and five_score >= 300
    ok &= abs(score - five_score) < 1e-6 and {r1, r2} == {3, 4} and c1 == c2 == 3

    # 2) 四连：行6 为 W W ? W，缺口的 W 在 (5,2)，交换补成四连
    def four(r, c, v):
        if (r, c) in {(6, 0), (6, 1), (6, 3)}:
            return 6          # W W ? W
        if (r, c) == (5, 2):
            return 6          # 上方补位 W
        return v
    b = _mk_board(four)
    assert _no_runs(b)
    res = best_move(b)
    score, mv = res
    three_only = simulate_swap(b, 0, 0, 0, 1)   # 底版任意交换（多数无效→None）
    print(f"[2] 四连场景: 最优分={score:.0f}, 着法={mv}")
    ok &= score >= 40  # 四连 45 或带连锁更高

    # 3) 重力下落
    g = np.full((8, 8), 7, np.uint8)
    g[7, 0], g[5, 0], g[3, 0] = 1, 2, 4
    g2 = _gravity(g)
    ok &= list(g2[:, 0]) == [7, 7, 7, 7, 7, 4, 2, 1]

    # 4) 超立方体：与相邻格子中"该色全场数量最多"的颜色交换
    def hyper(r, c, v):
        if (r, c) == (3, 3):
            return HYPER
        if r in (0, 7) and c % 2 == 0:
            return 0          # 隔一格放红，避免三连
        if (r, c) == (7, 3):
            return 3          # 打断行7底部的红三连
        return v
    b = _mk_board(hyper)
    assert _no_runs(b)
    res = best_move(b)
    score, mv = res
    neigh = [(2, 3), (4, 3), (3, 2), (3, 4)]
    expected = max(150 + 35 * int((b == b[r, c]).sum()) for r, c in neigh)
    print(f"[4] 超立方体: 最优分={score:.0f}, 期望={expected:.0f}, 着法={mv}")
    ok &= mv is not None and (3, 3) in {(mv[0], mv[1]), (mv[2], mv[3])}
    ok &= abs(score - expected) < 1e-6

    # 5) 无效交换返回 None；exclude 生效
    b2 = _mk_board(lambda r, c, v: v)
    ok &= simulate_swap(b2, 0, 0, 0, 1) is None      # 同色交换
    res = best_move(b2)
    if res:
        _s, m = res
        res2 = best_move(b2, exclude={m, m[::-1]})
        ok &= res2 is None or res2[1] != m

    print("自测结果:", "全部通过 OK" if ok else "存在失败 FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------- main --
def main() -> int:
    _configure_console()
    ap = argparse.ArgumentParser(description="Bejeweled 3 自动玩家")
    ap.add_argument("--window", default=None, help="游戏窗口标题（默认取配置/Bejeweled 3）")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--calibrate", action="store_true", help="重新校准棋盘位置")
    ap.add_argument("--rounds", type=int, default=None, help="最多交换次数")
    ap.add_argument("--seconds", type=float, default=None, help="最长运行秒数")
    ap.add_argument("--delay", type=float, default=0.35, help="两次交换的最小间隔秒")
    ap.add_argument("--dry", action="store_true", help="演练模式：不点击鼠标")
    ap.add_argument("--selftest", action="store_true", help="求解器自测")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    title = args.window or cfg.get("window_title", "Bejeweled 3")
    try:
        hwnd = winio.find_window(title)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None

    if args.calibrate or not cfg:
        cfg = calibrate(hwnd, args.config, window_title=title)

    bot = Bot(cfg, max_rounds=args.rounds, max_seconds=args.seconds,
              min_delay=args.delay, dry=args.dry)
    return 0 if bot.run() >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
