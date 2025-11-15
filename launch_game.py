import os
import subprocess
import time
import win32gui
import win32con
from window_utils import force_foreground

# Define possible paths for the game executable
POSSIBLE_PATHS = [
    r"C:\Program Files (x86)\NCSOFT\Purple\PurpleLauncher.exe",  # Main launcher
    r"C:\Program Files (x86)\NCSOFT\Purple\2.25.1029.8\Purple.exe",  # Direct executable
    os.path.expandvars(r"%LOCALAPPDATA%\NCSOFT\Purple\PurpleLauncher.exe"),
    os.path.expandvars(r"%PROGRAMFILES(X86)%\NCSOFT\Purple\Purple.exe"),
]

def find_game_window():
    """Find the game window by trying different possible titles."""
    window_titles = ["Purple", "Lineage2M", "PURPLE", "라인지M"]
    for title in window_titles:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd and win32gui.IsWindowVisible(hwnd):
            return hwnd
    return None

def is_game_running():
    """Check if Lineage2M is already running."""
    return bool(find_game_window())

def bring_to_foreground(hwnd):
    """Attempt to bring a window to the foreground."""
    if not hwnd:
        return False
    try:
        # Try to restore if minimized
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        # Force to foreground
        force_foreground(hwnd)
        return True
    except Exception as e:
        print(f"⚠️ Could not focus window: {e}")
        return False

def launch_lineage2m():
    """Launch Lineage2M and wait for window to appear."""
    # First check if game is already running
    hwnd = find_game_window()
    if hwnd:
        print("✅ Game is already running")
        if bring_to_foreground(hwnd):
            return hwnd
        # Even if focusing fails, return the handle
        return hwnd

    # Try to launch the game
    print("🎮 Launching game...")
    for path in POSSIBLE_PATHS:
        if not os.path.exists(path):
            continue
            
        try:
            subprocess.Popen([path])
            print(f"✅ Launched from: {path}")
            
            # Wait for window to appear
            for _ in range(60):  # 60 second timeout
                time.sleep(1)
                hwnd = find_game_window()
                if hwnd:
                    bring_to_foreground(hwnd)
                    return hwnd
                    
            print("❌ Window did not appear after launch")
            break  # Exit after first failed launch attempt
                
        except Exception as e:
            print(f"❌ Failed to launch {path}: {e}")
            continue
    
    # No installation found or launch failed
    print("❌ Could not find or launch game. Please check:")
    print("1. Game is installed")
    print("2. Installation location matches one of:")
    for path in POSSIBLE_PATHS:
        print(f"   - {path}")
    print("3. You have necessary permissions")
    return None

def main():
    """Launch game and bring window to foreground."""
    try:
        hwnd = launch_lineage2m()
        if hwnd:
            print("✅ Game window ready")
            return hwnd
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()