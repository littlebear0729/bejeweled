# -*- coding: utf-8 -*-
"""棋盘自动校准：粗搜 + 1px 精搜 + 双帧稳定性验证，结果存 config.json。"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from . import winio
from .board import classify_pixels

LETTERS = "ROYGBPW."


def _grid_fit(onehot, grad_unused, cell, oxs, oys, H, W):
    """对候选 (cell, ox, oy) 批量评估：返回 (valid, score, letters(N,8,8))。"""
    k = max(3, cell // 4) | 1
    blurred = np.stack([cv2.blur(onehot[i], (k, k)) for i in range(8)])
    CX = oxs[:, None] + (np.arange(8) + 0.5) * cell
    CY = oys[:, None] + (np.arange(8) + 0.5) * cell
    cxi = np.clip(CX.astype(int), 0, W - 1)
    cyi = np.clip(CY.astype(int), 0, H - 1)
    g = blurred[:, cyi[:, None, :, None], cxi[None, :, None, :]]   # (8, NOy, NOx, 8, 8)
    letters = g.argmax(0).astype(np.int8)
    purity = (g.max(0) - np.sort(g, axis=0)[-2]).astype(np.float32)
    del g
    NOy, NOx = letters.shape[:2]
    N = NOy * NOx
    flat = letters.reshape(N, 64)
    lm = letters.reshape(N, 8, 8)
    bad = np.zeros(N, np.int32)
    for r in range(8):
        for c in range(6):
            bad += ((lm[:, r, c] == lm[:, r, c + 1]) & (lm[:, r, c] == lm[:, r, c + 2]))
    for c in range(8):
        for r in range(6):
            bad += ((lm[:, r, c] == lm[:, r + 1, c]) & (lm[:, r, c] == lm[:, r + 2, c]))
    hist = np.stack([(flat == i).sum(1) for i in range(7)], 1).astype(float)
    nhues = (hist >= 4).sum(1)
    maxfrac = hist.max(1) / 64.0
    pur = purity.reshape(N, 64).mean(1)
    valid = (bad == 0) & (nhues >= 5) & (maxfrac <= 0.40)
    return valid, np.where(valid, pur, -1.0), lm


def _read_cells(img, cell, ox, oy):
    lt = classify_pixels(img)
    h = max(4, cell // 5)
    out = np.zeros((8, 8), np.uint8)
    for r in range(8):
        for c in range(8):
            cy = int(oy + (r + 0.5) * cell)
            cx = int(ox + (c + 0.5) * cell)
            patch = lt[cy - h:cy + h + 1, cx - h:cx + h + 1].ravel()
            out[r, c] = np.bincount(patch, minlength=8).argmax() if patch.size else 7
    return out


def calibrate(hwnd: int, save_path: str = "config.json", log=print,
              window_title: str = "Bejeweled 3") -> dict:
    winio.focus_window(hwnd)
    time.sleep(1.0)
    l, t, r, b = winio.get_rect(hwnd)
    imgA = winio.capture(l, t, r, b)
    time.sleep(1.2)
    imgB = winio.capture(l, t, r, b)
    H, W = imgA.shape[:2]
    log(f"窗口尺寸 {W}x{H}，开始校准（请确保棋盘完整可见且处于落定状态）……")

    letterA = classify_pixels(imgA)
    onehot = np.ascontiguousarray(np.eye(8, dtype=np.float32)[letterA].transpose(2, 0, 1))

    coarse = []
    for cell in range(60, 131, 2):
        ox_max, oy_max = W - 8 * cell, H - 8 * cell
        if ox_max < 0 or oy_max < 0:
            continue
        oxs = np.arange(0, ox_max + 1, 3)
        oys = np.arange(0, oy_max + 1, 3)
        valid, sc, lm = _grid_fit(onehot, None, cell, oxs, oys, H, W)
        i = int(np.argmax(sc))
        if sc[i] > 0:
            iy, ix = divmod(i, len(oxs))
            coarse.append((float(sc[i]), cell, int(oxs[ix]), int(oys[iy])))
    if not coarse:
        raise RuntimeError("未找到棋盘：请进入对局界面（棋盘完整、无弹窗、动画结束）后重试")
    coarse.sort(reverse=True)
    log("粗搜最佳: " + ", ".join(f"(cell={c[1]}, ox={c[2]}, oy={c[3]}, 纯度={c[0]:.3f})"
                                 for c in coarse[:3]))

    _, cell0, ox0, oy0 = coarse[0]
    best = None
    for cell in range(max(60, cell0 - 3), cell0 + 4):
        if W - 8 * cell < 0 or H - 8 * cell < 0:
            continue
        oxs = np.arange(max(0, ox0 - 8), min(W - 8 * cell, ox0 + 8) + 1)
        oys = np.arange(max(0, oy0 - 8), min(H - 8 * cell, oy0 + 8) + 1)
        valid, sc, lm = _grid_fit(onehot, None, cell, oxs, oys, H, W)
        i = int(np.argmax(sc))
        if best is None or sc[i] > best[0]:
            iy, ix = divmod(i, len(oxs))
            best = (float(sc[i]), cell, int(oxs[ix]), int(oys[iy]), lm[i].copy())
    score, cell, ox, oy, letters = best
    log(f"精搜: cell={cell} ox={ox} oy={oy} 纯度={score:.3f}")

    lA = _read_cells(imgA, cell, ox, oy)
    lB = _read_cells(imgB, cell, ox, oy)
    diff = int((lA != lB).sum())
    log(f"双帧稳定性: {64 - diff}/64 格一致")
    for row in lA:
        log("   " + " ".join(LETTERS[i] for i in row))
    if diff > 4:
        raise RuntimeError("棋盘不稳定（可能在动画/菜单中），请在对局落定状态重新校准")

    cfg = {
        "window_title": window_title,
        "cell": cell, "ox": ox, "oy": oy,
        "win_w": W, "win_h": H,
        "purity": round(score, 3),
        "calibrated_on": time.strftime("%F %T"),
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log(f"配置已保存到 {save_path}")
    return cfg
