# -*- coding: utf-8 -*-
"""Windows 窗口交互：找窗口 / 置前 / 截屏 / 鼠标点击 / 按键状态。

全部坐标均为物理像素（进程启动即开启 Per-Monitor v2 DPI 感知）。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time

import mss
import numpy as np

# ---------- DPI 感知：保证 GetWindowRect / SetCursorPos / mss 坐标一致 ----------
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)   # PER_MONITOR_AWARE_V2
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

VK_F8 = 0x77          # 紧急停止键
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080

WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)


def find_window(title: str) -> int:
    """按精确标题查找；找不到时再匹配标题包含关系。

    某些启动器/覆盖层会给标题追加内容，因此只依赖 FindWindowW 的精确
    匹配容易误报“游戏未启动”。
    """
    needle = title.strip().casefold()
    if not needle:
        raise RuntimeError("窗口标题不能为空")

    hwnd = user32.FindWindowW(None, title)
    if hwnd and user32.IsWindow(hwnd):
        return hwnd

    candidates: list[tuple[int, int, int, str]] = []

    @WNDENUMPROC
    def collect(candidate: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(candidate):
            return True
        if user32.GetWindowLongW(candidate, GWL_EXSTYLE) & WS_EX_TOOLWINDOW:
            return True
        length = user32.GetWindowTextLengthW(candidate)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(candidate, buf, length + 1)
        window_title = buf.value.strip()
        if not window_title:
            return True
        folded = window_title.casefold()
        exact = int(folded == needle)
        if not exact and needle not in folded:
            return True
        rect = wt.RECT()
        if not user32.GetWindowRect(candidate, ctypes.byref(rect)):
            return True
        area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
        candidates.append((exact, area, candidate, window_title))
        return True

    user32.EnumWindows(collect, 0)
    if not candidates:
        raise RuntimeError(
            f"未找到标题包含 {title!r} 的游戏窗口。"
            "请先启动游戏并解除锁屏，再确认窗口未最小化。"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def get_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.left, rect.top, rect.right, rect.bottom


def focus_window(hwnd: int) -> None:
    """把窗口带到前台（用 Alt 键技巧绕过 SetForegroundWindow 限制）。"""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        time.sleep(0.3)
    user32.keybd_event(0xA4, 0, 0, 0)          # Alt down
    try:
        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)
    finally:
        user32.keybd_event(0xA4, 0, 2, 0)      # Alt up
    time.sleep(0.15)


def capture(l: int, t: int, r: int, b: int) -> np.ndarray:
    """截取屏幕区域，返回 BGR ndarray。"""
    try:
        with mss.mss() as sct:
            img = np.array(sct.grab({"left": l, "top": t,
                                     "width": r - l, "height": b - t}))
    except Exception as exc:
        raise RuntimeError(
            "截取游戏画面失败。请确认桌面已解锁、游戏窗口未最小化，"
            "并优先使用窗口模式运行游戏"
        ) from exc
    return img[:, :, :3]


def click(x: int, y: int, settle: float = 0.06) -> None:
    """移动鼠标并左键单击（SetCursorPos + mouse_event，最兼容游戏窗口）。"""
    user32.SetCursorPos(int(x), int(y))
    time.sleep(settle)
    user32.mouse_event(2, 0, 0, 0, 0)   # LEFTDOWN
    time.sleep(0.035)
    user32.mouse_event(4, 0, 0, 0, 0)   # LEFTUP
    time.sleep(0.02)


def key_escape() -> None:
    """向当前焦点窗口发送 ESC（用于关闭暂停/对话框）。"""
    user32.keybd_event(0x1B, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(0x1B, 0, 2, 0)


def panic_pressed() -> bool:
    """F8 是否被按住（紧急停止）。"""
    return bool(user32.GetAsyncKeyState(VK_F8) & 0x8000)
