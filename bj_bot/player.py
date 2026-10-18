# -*- coding: utf-8 -*-
"""自动玩家主循环：读盘 → 求解 → 点击 → 等待动画 → 循环。

紧急停止：按住 F8；或在控制台 Ctrl+C。
"""
from __future__ import annotations

import time

import numpy as np

from . import winio
from .board import EMPTY, HYPER, SIZE, board_is_settled, pretty, read_board
from .solver import best_move


class Bot:
    def __init__(self, cfg: dict, max_rounds: int | None = None,
                 max_seconds: float | None = None, min_delay: float = 0.35,
                 dry: bool = False, log=print):
        self.cfg = cfg
        self.title = cfg.get("window_title", "Bejeweled 3")
        self.max_rounds = max_rounds
        self.max_seconds = max_seconds
        self.min_delay = min_delay
        self.dry = dry
        self.log = log
        self.hwnd = winio.find_window(self.title)
        self._hyper_cells: set = set()      # 已确认的超立方体位置
        self._hyper_blacklist: set = set()  # 被游戏拒绝过 hyper 判定的格子
        self._flip_votes: dict = {}
        self._forbidden: set = set()
        self._board_key = None
        self._unsettled = 0

    # ---------- 基础动作 ----------
    def _read_once(self):
        l, t, r, b = winio.get_rect(self.hwnd)
        w, h = r - l, b - t
        if abs(w - self.cfg.get("win_w", w)) > 3 or abs(h - self.cfg.get("win_h", h)) > 3:
            raise RuntimeError("游戏窗口尺寸与校准时不一致，请重新运行: python auto_play.py --calibrate")
        img = winio.capture(l, t, r, b)
        board, conf = read_board(img, self.cfg["cell"], self.cfg["ox"], self.cfg["oy"])
        return board, conf, (l, t, r, b)

    def _apply_hyper_policy(self, board: np.ndarray) -> np.ndarray:
        """应用超立方体黑名单：被拒绝过的格子还原为普通白宝石。"""
        for (r, c) in set(self._hyper_blacklist):
            if board[r, c] == EMPTY:
                self._hyper_blacklist.discard((r, c))   # 格子已被消掉，解除黑名单
            elif board[r, c] == HYPER:
                board[r, c] = 6                          # 还原为白宝石
        for (r, c) in self._hyper_cells:
            if (r, c) not in self._hyper_blacklist and board[r, c] != EMPTY:
                board[r, c] = HYPER
        return board

    def _read(self):
        """双帧合成读盘：两次采样逐格取多数/高置信者，抵抗闪光与高光干扰。"""
        board, conf, rect = self._read_once()
        time.sleep(0.13)
        board2, conf2, _ = self._read_once()
        if not np.array_equal(board, board2):
            merge = board.copy()
            flip = conf2 > conf
            merge[flip] = board2[flip]
            board = merge
        board = self._apply_hyper_policy(board)
        return board, conf, rect

    def _cell_xy(self, rect, r: int, c: int):
        l, t, _, _ = rect
        return (l + self.cfg["ox"] + int((c + 0.5) * self.cfg["cell"]),
                t + self.cfg["oy"] + int((r + 0.5) * self.cfg["cell"]))

    def _wait_settled(self, timeout: float = 9.0):
        """等待棋盘动画结束（连续两次读盘一致）。"""
        prev, rect = None, None
        t0 = time.time()
        time.sleep(0.55)
        while time.time() - t0 < timeout:
            if winio.panic_pressed():
                raise KeyboardInterrupt
            board, _conf, rect = self._read()
            if prev is not None and np.array_equal(board, prev):
                return board, rect
            prev = board
            time.sleep(0.22)
        return prev, rect

    # ---------- 单步 ----------
    def _step(self) -> tuple[bool, float]:
        """执行一步。返回 (是否完成有效交换, 预估分)。"""
        if winio.user32.GetForegroundWindow() != self.hwnd:
            winio.focus_window(self.hwnd)

        board, conf, rect = self._read()

        # 未落定（动画中 / 暂停遮挡 / 弹窗）
        if not board_is_settled(board):
            self._unsettled += 1
            if (board == EMPTY).sum() > 36 and self._unsettled >= 3:
                self.log("· 检测到暂停/遮挡覆盖，发送 ESC 恢复游戏…")
                winio.key_escape()
                time.sleep(1.3)
            else:
                time.sleep(0.35)
            return False, 0.0
        self._unsettled = 0

        # 超立方体确认：静止棋盘上仍在变色的格子（投票制）
        time.sleep(0.4)
        board2, _c2, rect = self._read()
        diffs = [(r, c) for r in range(SIZE) for c in range(SIZE)
                 if board2[r, c] != board[r, c]]
        if np.array_equal(board2, board):
            pass                                    # 完全静止
        elif 0 < len(diffs) <= 3:
            for (r, c) in diffs:
                if (r, c) in self._hyper_blacklist:
                    continue
                a, b2 = int(board[r, c]), int(board2[r, c])
                if not (0 <= a <= 6 and 0 <= b2 <= 6):
                    continue
                # 饱和色之间的翻转 = 光环换色（权重2）；涉及白色 = 可能是脉冲高光（权重1）
                w = 2 if (a <= 5 and b2 <= 5) else 1
                self._flip_votes[(r, c)] = self._flip_votes.get((r, c), 0) + w
            confirmed = [cell for cell, v in self._flip_votes.items() if v >= 2][:2]
            if confirmed:
                for (r, c) in confirmed:
                    board2[r, c] = HYPER
                    self._hyper_cells.add((r, c))
                self.log(f"  · 确认超立方体 @ {confirmed}")
            board = board2
        else:
            board = board2                          # 大变化：按新盘面处理

        key = board.tobytes()
        if key != self._board_key:
            if self._board_key is not None:
                # 自动洗牌后，旧格子上的超立方体判定已经失效。
                self._hyper_cells.clear()
                self._hyper_blacklist.clear()
            self._board_key = key
            self._forbidden = set()
            self._flip_votes = {}

        mv = best_move(board, exclude=self._forbidden)
        if mv is None:
            self.log("· 本盘面无可行交换，等待游戏自动洗牌/提示……")
            time.sleep(2.0)
            return False, 0.0
        score, (r1, c1, r2, c2) = mv

        self.log(f"-> 交换 ({r1},{c1})<->({r2},{c2})  预估分 {score:.0f}")
        if self.dry:
            self.log(pretty(board))
            return True, score

        pre = board.copy()
        x1, y1 = self._cell_xy(rect, r1, c1)
        x2, y2 = self._cell_xy(rect, r2, c2)
        winio.click(x1, y1)
        time.sleep(0.12)
        winio.click(x2, y2)

        settled, _rect = self._wait_settled()
        if settled is None or np.array_equal(settled, pre):
            # 交换被游戏拒绝（识别误差或棋盘刚刷新）
            self._forbidden.add((r1, c1, r2, c2))
            self._forbidden.add((r2, c2, r1, c1))
            for cell in ((r1, c1), (r2, c2)):
                if pre[cell] == HYPER:
                    self._hyper_cells.discard(cell)
                    self._hyper_blacklist.add(cell)
            self.log("  x 交换未生效（无消除），此着法本轮屏蔽")
            return False, 0.0

        self.log("  OK 消除完成，棋盘已更新")
        self._board_key = None
        self._hyper_cells = set()
        self._hyper_blacklist = set()
        self._flip_votes = {}
        return True, score

    # ---------- 主循环 ----------
    def run(self) -> int:
        self.log(f"自动玩家启动：{self.title} | 按住 F8 紧急停止"
                 + (" | 演练模式(不点击)" if self.dry else ""))
        winio.focus_window(self.hwnd)
        time.sleep(1.2)                     # 等游戏完全恢复前台/退出暂停动画
        t0 = time.time()
        rounds = 0
        est_total = 0.0
        try:
            while True:
                if winio.panic_pressed():
                    self.log("已按 F8，紧急停止。")
                    break
                if self.max_rounds is not None and rounds >= self.max_rounds:
                    self.log(f"已达到轮数上限 {self.max_rounds}。")
                    break
                if self.max_seconds is not None and time.time() - t0 >= self.max_seconds:
                    self.log(f"已运行 {self.max_seconds:.0f}s，自动停止。")
                    break
                try:
                    done, score = self._step()
                except KeyboardInterrupt:
                    raise
                except RuntimeError as e:
                    self.log(f"!! {e}")
                    break
                except Exception as e:            # 单步异常不中断整体
                    self.log(f"· 单步异常（忽略继续）: {type(e).__name__}: {e}")
                    time.sleep(1.0)
                    continue
                if done:
                    rounds += 1
                    est_total += score
                    time.sleep(self.min_delay)
        except KeyboardInterrupt:
            self.log("Ctrl+C，退出。")
        dt = max(1e-6, time.time() - t0)
        self.log(f"运行统计: 有效交换 {rounds} 次 / {dt:.0f}s"
                 f"（{rounds / dt * 60:.1f} 次/分钟），累计预估分 {est_total:.0f}")
        return rounds
