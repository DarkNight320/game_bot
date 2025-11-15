import sys
import os

def get_base_path():
    """Get the base path for resources.
    
    When bundled with PyInstaller, sys._MEIPASS contains the path to the bundle.
    When running as script, uses the script directory.
    """
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(relative_path):
    """Get absolute path for a resource file.
    
    Args:
        relative_path: Path relative to the bot directory (e.g., 'buttons/skip_button.png')
    
    Returns:
        Absolute path to the resource
    """
    base_path = get_base_path()
    return os.path.join(base_path, relative_path)

def ensure_resources_exist():
    """Verify that critical resource directories exist."""
    base_path = get_base_path()
    required_dirs = ['buttons', 'region']
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            print(f"⚠️ Warning: Resource directory not found: {dir_path}")
            return False
    
    return True
