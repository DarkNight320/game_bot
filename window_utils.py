import win32gui
import win32con
import win32api
import ctypes
import time

def force_foreground(hwnd, timeout=2.0):
    """Robustly try to bring a window to the foreground.

    Strategy (best-effort):
      1. Restore/Show the window if minimized.
      2. Try SetForegroundWindow.
      3. If that fails, attach thread input and retry SetForegroundWindow/SetActiveWindow/BringWindowToTop.
      4. As a fallback, briefly make the window topmost then remove topmost.
      5. Try the "Alt" key trick (synthesise VK_MENU) which can help on some Windows versions.

    Returns True if the requested window became foreground within `timeout`, False otherwise.
    """
    try:
        if not hwnd:
            return False

        # Try to restore or show the window first
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        except Exception:
            pass

        user32 = ctypes.windll.user32

        # Quick attempt
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception:
            pass

        # If already foreground, done
        if user32.GetForegroundWindow() == hwnd:
            return True

        # Try attaching thread input and set foreground
        attached = False
        try:
            fg = user32.GetForegroundWindow()
            fg_thread = user32.GetWindowThreadProcessId(fg, 0)
            target_thread = user32.GetWindowThreadProcessId(hwnd, 0)
            # Attach threads
            attached = bool(user32.AttachThreadInput(fg_thread, target_thread, True))
            try:
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetActiveWindow(hwnd)
                win32gui.BringWindowToTop(hwnd)
            except Exception:
                pass
        except Exception:
            attached = False

        # Detach if attached
        try:
            if attached:
                user32.AttachThreadInput(fg_thread, target_thread, False)
        except Exception:
            pass

        # Small wait loop to see if foreground changed
        end = time.time() + timeout
        while time.time() < end:
            if user32.GetForegroundWindow() == hwnd:
                return True
            time.sleep(0.03)

        # Fallback: temporarily make window topmost and try again
        try:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.05)
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        except Exception:
            pass

        # Quick check again
        if user32.GetForegroundWindow() == hwnd:
            return True

        # Final attempt: synthesize ALT key press (VK_MENU) which can allow SetForegroundWindow
        try:
            # keybd_event is deprecated but still works for this purpose
            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            win32gui.SetForegroundWindow(hwnd)
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        except Exception:
            pass

        # Final wait
        end = time.time() + 0.5
        while time.time() < end:
            if user32.GetForegroundWindow() == hwnd:
                return True
            time.sleep(0.02)

        return False
    except Exception as e:
        print(f"window_utils.force_foreground error: {e}")
        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return False
