# launcher.py - New game launcher implementation
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
            print(f"Found window: {title}")
            return hwnd
    return None

def is_game_running():
    """Check if the game is already running."""
    return bool(find_game_window())

def launch_game():
    """Launch the game and wait for the window to appear."""
    # First try to find existing window
    hwnd = find_game_window()
    if hwnd:
        print("✅ Game is already running")
        try:
            force_foreground(hwnd)
            return hwnd
        except Exception as e:
            print(f"⚠️ Could not focus window: {e}")
            return hwnd

    # Try launching the game
    print("🎮 Launching game...")
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            try:
                subprocess.Popen([path])
                print(f"✅ Launched from: {path}")
                
                # Wait for window
                for _ in range(60):
                    time.sleep(1)
                    hwnd = find_game_window()
                    if hwnd:
                        try:
                            force_foreground(hwnd)
                        except Exception as e:
                            print(f"⚠️ Could not focus window: {e}")
                        return hwnd
                
                print("❌ Window did not appear after launch")
                return None
                
            except Exception as e:
                print(f"❌ Failed to launch {path}: {e}")
                continue
    
    # No working installation found
    print("❌ Could not find game. Please check:")
    print("1. Game is installed")
    print("2. Installation location matches one of:")
    for path in POSSIBLE_PATHS:
        print(f"   - {path}")
    return None

def main():
    try:
        return launch_game()
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    main()