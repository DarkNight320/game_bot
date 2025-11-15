import win32gui, win32api, win32con, cv2, numpy as np, dxcam, ctypes, time, os, re, pyautogui, json
import pytesseract
from datetime import datetime
from window_utils import force_foreground
from collections import deque
from win32api import MessageBox
import pydirectinput
from stop_event import STOP_EVENT
from skimage.metrics import structural_similarity as ssim
from resource_path import get_resource_path, get_base_path

# Disable PyAutoGUI fail-safe to prevent errors when clicking at screen edges
# WARNING: This disables the safety feature that stops scripts when mouse moves to corners
pyautogui.FAILSAFE = False

# Helper: safe image write to avoid OpenCV assertion when image is empty

# ========== CONFIGURATION ==========
def load_config():
    config_file = get_resource_path('bot_config.json')
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        return {
            "WINDOW_SUBSTRING": "Lineage2M",
            "HEAL_KEY": "1",
            "HEAL_THRESHOLD": 40.0,
            "CAPTURE_INTERVAL": 2.0,
            "MOTION_THRESHOLD": 10000,
            "MISSION_DIFF_THRESHOLD": 20,
            "CLICK_OFFSET_X": 0,
            "CLICK_OFFSET_Y": 0,
            "USE_REAL_MOUSE": True,
            "MOTION_RATIO": 0.3
        }

# Load configuration
config = load_config()
WINDOW_SUBSTRING = config.get("WINDOW_SUBSTRING", "Lineage2M")
HEAL_KEY = config.get("HEAL_KEY", "1")
HEAL_THRESHOLD = float(config.get("HEAL_THRESHOLD", 40.0))
CAPTURE_INTERVAL = float(config.get("CAPTURE_INTERVAL", 2.0))
MOTION_THRESHOLD = int(config.get("MOTION_THRESHOLD", 10000))
MISSION_DIFF_THRESHOLD = int(config.get("MISSION_DIFF_THRESHOLD", 20))
BUTTONS_FOLDER = get_resource_path("buttons")
CLICK_OFFSET_X = int(config.get("CLICK_OFFSET_X", 0))
CLICK_OFFSET_Y = int(config.get("CLICK_OFFSET_Y", 5))
USE_REAL_MOUSE = bool(config.get("USE_REAL_MOUSE", True))
MOTION_RATIO = float(config.get("MOTION_RATIO", 0.3))

# Template cache for button detection performance optimization
BUTTON_TEMPLATES_CACHE = {}
BUTTON_TEMPLATES_LOADED = False

def load_button_templates():
    """Load all button templates into cache for faster detection."""
    global BUTTON_TEMPLATES_CACHE, BUTTON_TEMPLATES_LOADED
    if BUTTON_TEMPLATES_LOADED:
        return BUTTON_TEMPLATES_CACHE
    
    if not os.path.exists(BUTTONS_FOLDER):
        print(f"⚠️ Folder '{BUTTONS_FOLDER}' not found.")
        BUTTON_TEMPLATES_LOADED = True
        return {}
    
    templates = {}
    for filename in os.listdir(BUTTONS_FOLDER):
        if not filename.lower().endswith(".png"):
            continue
        tpl_path = os.path.join(BUTTONS_FOLDER, filename)
        template = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates[filename] = template
            print(f"✅ Loaded template: {filename}")
        else:
            print(f"⚠️ Failed to load template: {tpl_path}")
    
    BUTTON_TEMPLATES_CACHE = templates
    BUTTON_TEMPLATES_LOADED = True
    print(f"📦 Cached {len(templates)} button templates")
    return templates

# Load templates at module initialization
load_button_templates()
# Fraction of changed pixels (changed_pixels / area) used to detect motion.
# This is now configurable via the CONFIG_FILE as `MOTION_RATIO` (float).
# Keep MOTION_THRESHOLD as a backward-compatible absolute fallback.
MOTION_RATIO = 0.3
# EMA smoothing alpha for motion ratio (0..1). Higher -> more responsive, lower -> smoother
MOTION_EMA_ALPHA = 0.5
# Number of consecutive frames required to declare 'moving'
MOTION_STREAK_TO_MOVE = 2
# Number of consecutive stable frames required to declare 'not moving'
MOTION_STABLE_TO_STOP = 3
# Reference resolution used for saving regions in the GUI. Regions saved by
# the GUI are stored in this coordinate space (x,y,w,h) with a marker
# '_coord':'ref'. At runtime we convert reference coords to the current
# captured frame / window size.
REF_WIDTH = 1920
REF_HEIGHT = 1080
# Small configurable offset to apply to clicks (useful if clicks land slightly off)
CLICK_OFFSET_X = 0
CLICK_OFFSET_Y = 15
# If True, use the real OS mouse (move -> click -> restore) by default.
# If False, attempt window message (invisible) clicks first.
USE_REAL_MOUSE = True
# Movement flag timeout - timeout for detecting when movement flag becomes true (in seconds)
MOVEMENT_FLAG_TIMEOUT = 300

# Configuration file path - GUI writes to this file
CONFIG_FILE = get_resource_path('bot_config.json')
LAST_NOTIFY_FILE = get_resource_path('.last_notification')

# Saved regions loaded from config (optional)
SAVED_REGIONS = {}
ACTIVE_REGION_NAME = None

def should_notify_today():
    """Check if we should show notification today."""
    try:
        if os.path.exists(LAST_NOTIFY_FILE):
            with open(LAST_NOTIFY_FILE, 'r') as f:
                last_date = datetime.strptime(f.read().strip(), '%Y-%m-%d').date()
                return last_date < datetime.now().date()
        return True
    except Exception:
        return True

def update_notification_date():
    """Update the last notification date to today."""
    try:
        with open(LAST_NOTIFY_FILE, 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"Error updating notification date: {e}")

def show_daily_notification(hwnd):
    """Show daily notification with game stats."""
    try:
        message = "Daily Bot Notification\\n\\n"
        message += "Your bot is running! Remember to:\\n"
        message += "1. Check your blood bottle count\\n"
        message += "2. Verify your character's position\\n"
        message += "3. Make sure auto-features are enabled"
        
        MessageBox(hwnd, message, "Daily Bot Reminder", 0x40)  # 0x40 is MB_ICONINFORMATION

        update_notification_date()
    except Exception as e:
        print(f"Error showing notification: {e}")


def load_config_from_file():
    """Load configuration overrides from CONFIG_FILE. Values present will
    replace the module-level defaults. This function is safe to call at
    import-time and will silently ignore parse errors.
    """
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to read config file {CONFIG_FILE}: {e}")
        return

    global WINDOW_SUBSTRING, HEAL_KEY, HEAL_THRESHOLD, CAPTURE_INTERVAL
    global MOTION_THRESHOLD, MISSION_DIFF_THRESHOLD, CLICK_OFFSET_X, CLICK_OFFSET_Y, USE_REAL_MOUSE
    global MOTION_RATIO

    WINDOW_SUBSTRING = cfg.get('WINDOW_SUBSTRING', WINDOW_SUBSTRING)
    HEAL_KEY = cfg.get('HEAL_KEY', HEAL_KEY)
    try:
        HEAL_THRESHOLD = float(cfg.get('HEAL_THRESHOLD', HEAL_THRESHOLD))
    except Exception:
        pass
    try:
        CAPTURE_INTERVAL = float(cfg.get('CAPTURE_INTERVAL', CAPTURE_INTERVAL))
    except Exception:
        pass
    try:
        MOTION_THRESHOLD = int(cfg.get('MOTION_THRESHOLD', MOTION_THRESHOLD))
    except Exception:
        pass
    # New config: MOTION_RATIO (fraction of changed pixels per-area used to
    # determine motion). If provided, this takes precedence for motion
    # detection. MOTION_THRESHOLD remains available as a backward-compatible
    # absolute pixel-count fallback.
    try:
        MOTION_RATIO = float(cfg.get('MOTION_RATIO', MOTION_RATIO))
    except Exception:
        pass
    try:
        MOTION_EMA_ALPHA = float(cfg.get('MOTION_EMA_ALPHA', MOTION_EMA_ALPHA))
    except Exception:
        pass
    try:
        MOTION_STREAK_TO_MOVE = int(cfg.get('MOTION_STREAK_TO_MOVE', MOTION_STREAK_TO_MOVE))
    except Exception:
        pass
    try:
        MOTION_STABLE_TO_STOP = int(cfg.get('MOTION_STABLE_TO_STOP', MOTION_STABLE_TO_STOP))
    except Exception:
        pass
    try:
        MISSION_DIFF_THRESHOLD = int(cfg.get('MISSION_DIFF_THRESHOLD', MISSION_DIFF_THRESHOLD))
    except Exception:
        pass
    try:
        CLICK_OFFSET_X = int(cfg.get('CLICK_OFFSET_X', CLICK_OFFSET_X))
    except Exception:
        pass
    try:
        CLICK_OFFSET_Y = int(cfg.get('CLICK_OFFSET_Y', CLICK_OFFSET_Y))
    except Exception:
        pass
    try:
        USE_REAL_MOUSE = bool(cfg.get('USE_REAL_MOUSE', USE_REAL_MOUSE))
    except Exception:
        pass
    # optional saved regions (absolute screen coords) and an active region name
    global SAVED_REGIONS, ACTIVE_REGION_NAME
    try:
        SAVED_REGIONS = cfg.get('SAVED_REGIONS', {}) or {}
    except Exception:
        SAVED_REGIONS = {}
    try:
        ACTIVE_REGION_NAME = cfg.get('ACTIVE_REGION', None)
    except Exception:
        ACTIVE_REGION_NAME = None

    # no GUI-controlled CAPTURE_SCALE in current config

# Load config file if present
load_config_from_file()

# ========== INITIALIZATION ==========
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
camera = dxcam.create()
prev_character, prev_prev_character = None, None
mission_area_state = [None, None, None]


# ========== UTILITY FUNCTIONS ==========
def validate_click_coordinates(x, y):
    """Validate that click coordinates are within screen bounds.
    
    Returns (valid_x, valid_y, is_valid) where is_valid is True if coordinates
    are within reasonable bounds (not at screen corners/edges that trigger failsafe).
    """
    try:
        screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        
        # Clamp coordinates to screen bounds
        valid_x = max(10, min(int(x), screen_width - 10))
        valid_y = max(10, min(int(y), screen_height - 10))
        
        # Check if coordinates are at screen corners (which trigger failsafe)
        # Allow some margin (10 pixels) from edges
        is_valid = (10 <= valid_x <= screen_width - 10 and 
                   10 <= valid_y <= screen_height - 10)
        
        if not is_valid:
            print(f"⚠️ Click coordinates ({x}, {y}) are out of bounds or at screen edge. Clamped to ({valid_x}, {valid_y})")
        
        return valid_x, valid_y, is_valid
    except Exception as e:
        print(f"⚠️ Error validating coordinates: {e}")
        # Return original coordinates if validation fails
        return int(x), int(y), True

def get_window_info():
    """Find window coordinates containing the target substring.
    Handles both windowed and fullscreen modes.
    Prefers game windows over controller/bot windows.
    Validates and clamps coordinates to ensure they're always valid."""
    candidates = []
    
    # Get screen dimensions once
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    def callback(hwnd, extra):
        if not win32gui.IsWindowVisible(hwnd):
            return
            
        title = win32gui.GetWindowText(hwnd)
        if WINDOW_SUBSTRING.lower() not in title.lower():
            return
            
        try:
            # Get window rect
            rect = win32gui.GetWindowRect(hwnd)
            
            # Validate and clamp coordinates to screen bounds (never negative)
            x = max(0, min(rect[0], screen_width - 1))
            y = max(0, min(rect[1], screen_height - 1))
            width = max(1, min(rect[2] - rect[0], screen_width - x))
            height = max(1, min(rect[3] - rect[1], screen_height - y))
            
            # Check if window is fullscreen
            is_fullscreen = (width >= screen_width * 0.95 and 
                           height >= screen_height * 0.95)
            
            # Calculate priority: prefer game windows over bot/controller windows
            # Higher priority = better match
            priority = 0
            title_lower = title.lower()
            
            # Prefer windows that are NOT bot/controller windows
            if "bot" not in title_lower and "controller" not in title_lower:
                priority += 10
            
            # Prefer larger windows (likely the game)
            priority += min(5, width // 200) + min(5, height // 200)
            
            # Prefer non-fullscreen windows (more likely to be the game window)
            if not is_fullscreen:
                priority += 2
                           
            candidate = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "name": title,
                "hwnd": hwnd,
                "is_fullscreen": is_fullscreen,
                "priority": priority
            }
            
            candidates.append(candidate)
            print(f"Found window candidate: {title} ({width}x{height}, priority={priority})")
            
        except Exception as e:
            print(f"⚠️ Error getting window info for {title}: {e}")
            
    win32gui.EnumWindows(callback, None)
    
    if not candidates:
        print("⚠️ Target window not found")
        return None
    
    # Sort by priority (highest first) and return the best match
    candidates.sort(key=lambda x: x.get("priority", 0), reverse=True)
    best = candidates[0]
    
    # Remove priority from the returned dict and ensure all values are valid
    info = {
        "x": max(0, best["x"]),
        "y": max(0, best["y"]),
        "width": max(1, best["width"]),
        "height": max(1, best["height"]),
        "name": best["name"],
        "hwnd": best["hwnd"],
        "is_fullscreen": best["is_fullscreen"]
    }
    
    # Final validation: ensure coordinates are within screen bounds
    info["x"] = max(0, min(info["x"], screen_width - 1))
    info["y"] = max(0, min(info["y"], screen_height - 1))
    info["width"] = max(1, min(info["width"], screen_width - info["x"]))
    info["height"] = max(1, min(info["height"], screen_height - info["y"]))
    
    # Warn if window is too small for accurate clicking
    MIN_WINDOW_WIDTH = 800
    MIN_WINDOW_HEIGHT = 600
    if info["width"] < MIN_WINDOW_WIDTH or info["height"] < MIN_WINDOW_HEIGHT:
        print(f"⚠️ Game window is small ({info['width']}x{info['height']}). Recommended minimum: {MIN_WINDOW_WIDTH}x{MIN_WINDOW_HEIGHT} for accurate clicking.")
    
    print(f"✅ Selected window: {info['name']} at ({info['x']}, {info['y']}) size {info['width']}x{info['height']}")
    
    return info


def detect_center(x, y, w, h, window_info):
    cx, cy = x + w // 2, y + h // 2
    print(f"✅ Selected region at (x={x}, y={y}, w={w}, h={h}) → Center=({cx}, {cy})")

    screen_x = window_info["x"] + cx
    screen_y = window_info["y"] + cy
    return screen_x, screen_y

def detect_center_from_image(img, x, y, window_info):
    h, w = img.shape[:2]
    return detect_center(x, y, w, h, window_info)


def map_region_to_screen(r, window_info=None):
    """Map a region dict (may contain '_coord'=='ref') to absolute screen coords.

    Returns (x, y, w, h) in screen coordinates or None on error.
    """
    try:
        if r is None:
            return None
        # Reference coords mapping
        if r.get('_coord') == 'ref':
            wi = window_info or get_window_info()
            if wi and 'width' in wi and 'height' in wi:
                win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                x = win_x + int(r.get('x', 0) * win_w / float(REF_WIDTH))
                y = win_y + int(r.get('y', 0) * win_h / float(REF_HEIGHT))
                w = int(r.get('w', 0) * win_w / float(REF_WIDTH))
                h = int(r.get('h', 0) * win_h / float(REF_HEIGHT))
            else:
                user32 = ctypes.windll.user32
                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
                x = int(r.get('x', 0) * sw / float(REF_WIDTH))
                y = int(r.get('y', 0) * sh / float(REF_HEIGHT))
                w = int(r.get('w', 0) * sw / float(REF_WIDTH))
                h = int(r.get('h', 0) * sh / float(REF_HEIGHT))
        else:
            # absolute screen coords
            x = int(r.get('x', 0))
            y = int(r.get('y', 0))
            w = int(r.get('w', 0))
            h = int(r.get('h', 0))
        return (x, y, w, h)
    except Exception:
        return None


def compute_nested_region_center(child_region, parent_region, window_info=None):
    """Compute the center screen coordinates of a small (child) region that
    may be defined relative to a parent region.

    Heuristic logic:
      - If child_region coordinates look like offsets inside parent (i.e. 0 <=
        child.x < parent.w and 0 <= child.y < parent.h), treat child as
        relative to parent and compute center = parent_origin + child_offset + child_size/2.
      - Otherwise treat child_region as absolute and return its center.

    Returns dict: {'center_screen': (sx,sy), 'center_frame': (fx,fy), 'screen_rect':(x,y,w,h)}
    or None on error.
    """
    try:
        if child_region is None or parent_region is None:
            return None

        # Map parent to screen coords
        p = map_region_to_screen(parent_region, window_info)
        if p is None:
            return None
        px, py, pw, ph = p

        # Preserve original child coords for heuristic (before mapping)
        try:
            c_x_raw = int(child_region.get('x', 0))
            c_y_raw = int(child_region.get('y', 0))
            c_w_raw = int(child_region.get('w', 0))
            c_h_raw = int(child_region.get('h', 0))
        except Exception:
            c_x_raw = c_y_raw = c_w_raw = c_h_raw = 0

        # Map child to screen coords too
        c = map_region_to_screen(child_region, window_info)
        if c is None:
            return None
        cx_abs, cy_abs, cw, ch = c

        # Heuristic: if child raw coords look like offsets inside parent, use relative math
        is_relative = (0 <= c_x_raw < pw) and (0 <= c_y_raw < ph) and (c_w_raw <= pw) and (c_h_raw <= ph)

        if is_relative:
            # child_region interpreted as offset inside parent
            sx = px + c_x_raw + cw // 2
            sy = py + c_y_raw + ch // 2
            screen_rect = (px + c_x_raw, py + c_y_raw, cw, ch)
        else:
            # child_region interpreted as absolute screen rect
            sx = cx_abs + cw // 2
            sy = cy_abs + ch // 2
            screen_rect = (cx_abs, cy_abs, cw, ch)

        # frame-relative center (if window_info provided)
        if window_info and isinstance(window_info, dict) and 'x' in window_info and 'y' in window_info:
            fx = sx - int(window_info['x'])
            fy = sy - int(window_info['y'])
            center_frame = (fx, fy)
        else:
            center_frame = None

        return {'center_screen': (int(sx), int(sy)), 'center_frame': center_frame, 'screen_rect': screen_rect}
    except Exception as e:
        print('⚠️ compute_nested_region_center failed:', e)
        return None


def convert_absolute_regions_to_ref():
    """Convert all saved regions from absolute screen coords to reference-scaled coords.
    
    This allows regions to work across different PCs with different resolutions.
    Assumes regions were captured on a 1920x1080 reference resolution.
    
    Returns: dict of converted regions or None if conversion fails
    """
    global SAVED_REGIONS
    if not SAVED_REGIONS:
        print("⚠️ No saved regions to convert")
        return None
    
    wi = get_window_info()
    if wi is None:
        print("⚠️ Window not found - cannot determine reference dimensions")
        return None
    
    converted = {}
    win_w, win_h = wi['width'], wi['height']
    
    print(f"Converting {len(SAVED_REGIONS)} regions from absolute to reference coords...")
    print(f"  Current window: {win_w}x{win_h}")
    print(f"  Reference resolution: {REF_WIDTH}x{REF_HEIGHT}")
    
    for name, region in SAVED_REGIONS.items():
        if region.get('_coord') in ['ref', 'rel']:
            converted[name] = region
            print(f"  ✓ {name}: already in {region.get('_coord')} format")
            continue
        
        try:
            x = int(region.get('x', 0))
            y = int(region.get('y', 0))
            w = int(region.get('w', region.get('width', 0)))
            h = int(region.get('h', region.get('height', 0)))
            
            win_x = int(wi['x'])
            win_y = int(wi['y'])
            rel_x = x - win_x
            rel_y = y - win_y
            
            scale_x = REF_WIDTH / float(win_w)
            scale_y = REF_HEIGHT / float(win_h)
            
            ref_x = int(rel_x * scale_x)
            ref_y = int(rel_y * scale_y)
            ref_w = int(w * scale_x)
            ref_h = int(h * scale_y)
            
            converted[name] = {
                'x': ref_x,
                'y': ref_y,
                'w': ref_w,
                'h': ref_h,
                '_coord': 'ref'
            }
            print(f"  ✓ {name}: ({x},{y},{w},{h}) → ref({ref_x},{ref_y},{ref_w},{ref_h})")
            
        except Exception as e:
            print(f"  ⚠️ {name}: conversion failed - {e}")
            converted[name] = region
    
    SAVED_REGIONS = converted
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except:
        config = {}
    
    config['SAVED_REGIONS'] = converted
    
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Converted {len(converted)} regions. Saved to {CONFIG_FILE}")
    return converted


def list_all_region_templates():
    """List all PNG template files in the region directory and its subdirectories.
    
    Returns: List of tuples (name, full_path) where:
        - name is the filename without extension (e.g., 'guide_button_region')
        - full_path is the absolute path to the PNG file
    """
    templates = []
    region_dir = get_resource_path('region')
    try:
        for root, _, files in os.walk(region_dir):
            for fn in files:
                if not fn.lower().endswith('.png'):
                    continue
                name = os.path.splitext(fn)[0]
                full_path = os.path.join(root, fn)
                # Include the subdirectory in the name if it's not the root region dir
                if root != region_dir:
                    subdir = os.path.basename(root)
                    name = f"{subdir}/{name}"
                templates.append((name, full_path))
    except Exception as e:
        print(f"⚠️ Error listing templates: {e}")
    return sorted(templates)

def search_region_in_frame(region, frame, threshold=0.65):
    """Search for a saved region image inside a captured frame.

    Parameters
    - region: either a path to an image file, a region name (looks in ./region/<name>.png),
              or a numpy array image (BGR or RGB).
    - frame: captured frame as a numpy array (BGR as returned by dxcam/cv2 is expected).
    - threshold: matching threshold (0..1) for normalized correlation.

    Returns: dict with keys {x,y,w,h,score,top_left,bottom_right} for best match,
             or None if no match above threshold.
    """
    print("******************************")
    tpl = None
    # region can be a numpy array already
    if isinstance(region, np.ndarray):
        tpl = region.copy()
    else:
        # try as file path first
        if os.path.exists(region):
            tpl = cv2.imread(region, cv2.IMREAD_UNCHANGED)
        else:
            # try region name under ./region/<name>.png or search recursively in subdirs
            region_dir = get_resource_path('region')
            alt = os.path.join(region_dir, f"{region}.png")
            if os.path.exists(alt):
                tpl = cv2.imread(alt, cv2.IMREAD_UNCHANGED)
                
            else:
                # Walk the region directory and subdirectories to find a matching png
                found = None
                try:
                    for root, _, files in os.walk(region_dir):
                        for fn in files:
                            if not fn.lower().endswith('.png'):
                                continue
                            name_no_ext = os.path.splitext(fn)[0]
                            # match either exact filename or basename (case-insensitive)
                            if fn.lower() == f"{region.lower()}.png" or name_no_ext.lower() == region.lower():
                                found = os.path.join(root, fn)
                                break
                        if found:
                            break
                except Exception:
                    found = None

                if found and os.path.exists(found):
                    tpl = cv2.imread(found, cv2.IMREAD_UNCHANGED)
                else:
                    # maybe region is already a filename without extension inside region/
                    alt2 = os.path.join(region_dir, region)
                    if os.path.exists(alt2):
                        tpl = cv2.imread(alt2, cv2.IMREAD_UNCHANGED)
                    else:
                        # Search through all available templates as a last resort
                        region_lower = region.lower()
                        found2 = None
                        try:
                            templates = list_all_region_templates()
                            for name, full_path in templates:
                                # Match either the full name (with subdirs) or just the base name
                                base_name = os.path.basename(name)
                                if name.lower() == region_lower or base_name.lower() == region_lower:
                                    found2 = full_path
                                    print(f"✅ Found template '{region}' at {found2}")
                                    break
                        except Exception as e:
                            print(f"⚠️ Template search error: {e}")
                            found2 = None

                        if found2 and os.path.exists(found2):
                            tpl = cv2.imread(found2, cv2.IMREAD_UNCHANGED)
 
    if tpl is None:
        return None
    # convert to grayscale for matching

    try:
        # Convert template to grayscale
        if tpl.ndim == 3:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        else:
            tpl_gray = tpl
        
        # Convert frame to grayscale
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
    except Exception as e:
        print(f"⚠️ Error converting images to grayscale: {e}")
        return None
    
    th, tw = tpl_gray.shape[:2]
    fh, fw = frame_gray.shape[:2]
    if th <= 0 or tw <= 0 or fh <= 0 or fw <= 0:
        return None

    # If template is larger than frame, no match
    if th > fh or tw > fw:
        return None

    try:

        res = cv2.matchTemplate(frame_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(res)
    except Exception:
        return None

    if max_val >= threshold:
        x, y = max_loc
        return {
            'x': int(x), 'y': int(y), 'w': int(tw), 'h': int(th),
            'score': float(max_val), 'top_left': (int(x), int(y)),
            'bottom_right': (int(x + tw), int(y + th))
        }
    return None

def find_region_position(region, frame, window_info=None, threshold=0.85):
    """
    Find a region (template path / image / region file) inside `frame`.

    Args:
      region: path to template or numpy image or region name (same as search_region_in_frame)
      frame: BGR numpy array where to search (as returned by capture_region)
      window_info: optional dict with keys 'x' and 'y' giving absolute screen origin of the frame;
                   when present the returned screen_* values will be absolute coordinates.
      threshold: matching threshold passed to search_region_in_frame (0..1)

    Returns:
      dict with keys:
        - frame_x, frame_y, w, h       : coordinates inside `frame`
        - screen_x, screen_y           : top-left absolute screen coords (if window_info given), otherwise same as frame_x/frame_y
        - center_frame, center_screen  : centers as tuples
        - score                        : match confidence
      or None if not found.
    """
    match = search_region_in_frame(region, frame, threshold=threshold)

    if not match:
        return None

    fx, fy, fw, fh = int(match['x']), int(match['y']), int(match['w']), int(match['h'])
    score = float(match.get('score', 0.0))

    if window_info and isinstance(window_info, dict) and 'x' in window_info and 'y' in window_info:
        sx = int(window_info['x']) + fx
        sy = int(window_info['y']) + fy
    else:
        sx, sy = fx, fy

    center_frame = (fx + fw // 2, fy + fh // 2)
    center_screen = (sx + fw // 2, sy + fh // 2)

    return {
        'frame_x': fx, 'frame_y': fy, 'w': fw, 'h': fh,
        'screen_x': sx, 'screen_y': sy,
        'center_frame': center_frame, 'center_screen': center_screen,
        'score': score
    }


# def search_and_click_template(template_key, frame, window_info=None, threshold=0.45, method='auto', debug=False):
#     """Search for a template (name or path) inside frame and click its center if found.

#     template_key: e.g. 'buttons/left_confirm_region' or absolute path to PNG
#     frame: captured window frame (BGR numpy array)
#     window_info: dict returned by get_window_info() (used to compute screen coords)
#     threshold: matching threshold passed to find_region_position/search_region_in_frame
#     method: click method passed to click helpers ('auto','window','sendinput','real_restore')
#     Returns True if a click was performed, False otherwise.
#     """
#     try:
#         res = find_region_position(template_key, frame, window_info=window_info, threshold=threshold)
#         if not res:
#             if debug:
#                 print(f"search_and_click_template: '{template_key}' not found (threshold={threshold})")
#             return False

#         sx, sy = res['center_screen']
#         sx = int(sx) + CLICK_OFFSET_X
#         sy = int(sy) + CLICK_OFFSET_Y
#         if debug:
#             print(f"search_and_click_template: clicking {template_key} at ({sx},{sy}) via method={method}")

#         hwnd = window_info.get('hwnd') if window_info and isinstance(window_info, dict) else None
#         if method == 'window' and hwnd:
#             ok = send_window_click(hwnd, sx, sy)
#         elif method == 'sendinput' and hwnd:
#             ok = send_input_click_abs(sx, sy, hwnd=hwnd)
#         elif method == 'real_restore':
#             ok = send_real_click_restore(sx, sy)
#         else:
#             # auto: try attach/sendinput then real restore fallback
#             ok = click_with_attach(sx, sy, hwnd=hwnd)
#             if not ok:
#                 ok = send_real_click_restore(sx, sy)

#         return bool(ok)
#     except Exception as e:
#         print(f"⚠️ search_and_click_template failed for '{template_key}': {e}")
#         return False


def mouse_move_click(r):
    """Move mouse to (x,y) and click, using configured method."""

    cx=int(r['x']+r['w']//2)
    cy=int(r['y']+r['h']//2)
    print('moving to',cx,cy)
    send_real_click_restore(cx, cy)

def  drag_mouse(r,start_x_offset=0,start_y_offset=30,end_x_offset=0,end_y_offset=0,duration=2):
    """Drag mouse from center of region r with optional offsets. Restores original cursor position after drag."""
    try:
        old_pos = win32api.GetCursorPos()
        
        try:
            wi = get_window_info()
            if not wi:
                print("⚠️ Could not find game window for drag")
                return False
        except Exception as e:
            print(f"⚠️ Error getting window info: {e}")
            return False
        
        win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
        win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
        
        reg_x = int(r.get('x', 0))
        reg_y = int(r.get('y', 0))
        reg_w = int(r.get('w', 0))
        reg_h = int(r.get('h', 0))
        
        if r.get('_coord') == 'rel':
            ref_w = r.get('_ref_width', 0)
            ref_h = r.get('_ref_height', 0)
            if ref_w > 0 and ref_h > 0:
                scale_x = win_w / float(ref_w)
                scale_y = win_h / float(ref_h)
                reg_x = int(reg_x * scale_x)
                reg_y = int(reg_y * scale_y)
                reg_w = int(reg_w * scale_x)
                reg_h = int(reg_h * scale_y)
            
            start_x = win_x + reg_x + reg_w // 2 + start_x_offset
            start_y = win_y + reg_y + start_y_offset
            end_x = win_x + reg_x + reg_w // 2 + end_x_offset
            end_y = win_y + reg_y + reg_h + end_y_offset
        else:
            start_x = reg_x + reg_w // 2 + start_x_offset
            start_y = reg_y + start_y_offset
            end_x = reg_x + reg_w // 2 + end_x_offset
            end_y = reg_y + reg_h + end_y_offset
        
        print(f'dragging from {start_x} {start_y} to {end_x} {end_y}')
        pydirectinput.moveTo(start_x, start_y)
        time.sleep(0.2)
        pydirectinput.mouseDown()
        
        dx = end_x - start_x
        dy = end_y - start_y
        distance = (dx**2 + dy**2)**0.5
        
        if distance > 0:
            steps = max(3, int(distance / 1000))
            step_duration = duration / steps
            for i in range(1, steps + 1):
                t = i / steps
                x = int(start_x + dx * t)
                y = int(start_y + dy * t)
                pydirectinput.moveTo(x, y)
                time.sleep(step_duration)
        
        pydirectinput.mouseUp()
        time.sleep(0.2)

        try:
            win32api.SetCursorPos(old_pos)
        except Exception:
            win32api.SetCursorPos((old_pos[0], old_pos[1]))
        print(f"✅ Restored cursor to original position {old_pos}")
        return True
    except Exception as e:
        print("⚠️ drag_mouse failed:", e)
        return False

def send_real_mouse_move(x, y):
    """Send a move to the virtual mouse without affecting the real cursor."""
    # Deprecated: move the real cursor. Prefer send_window_click(hwnd, x, y)
    user32 = ctypes.windll.user32
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    norm_x = int(x * 65535 / screen_w)
    norm_y = int(y * 65535 / screen_h)
    print(f"🧩 (fallback) Sending virtual move ({x},{y}) -> normalized ({norm_x},{norm_y})")
    try:
        # As a safer fallback: move system cursor, click, then restore previous cursor position
        send_real_click_restore(x, y)
        try:
            capture_click_frame(x, y, method='real_restore')
        except Exception:
            pass
    except Exception as e:
        print("⚠️ pydirectinput / restore click failed:", e)


def send_window_click(hwnd, x, y, button='left'):
    """Send mouse messages directly to a window (doesn't move system cursor).

    hwnd: target window handle
    x,y: screen coordinates where to click
    button: 'left' or 'right'

    Note: Some games read raw input and ignore window messages; this works for
    many windowed apps but may not work for all games.
    """
    try:
        # Force window to foreground first
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            win32gui.SetForegroundWindow(hwnd)
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            time.sleep(0.1)  # Small delay to let window activation complete
        except Exception as e:
            print("⚠️ Could not force foreground:", e)

        # Convert screen to client coordinates
        cx, cy = win32gui.ScreenToClient(hwnd, (int(x), int(y)))
        # lParam packs y<<16 | x (words)
        lparam = (cy << 16) | (cx & 0xFFFF)

        # send a mouse move first
        win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)

        if button == 'left':
            down, up = win32con.WM_LBUTTONDOWN, win32con.WM_LBUTTONUP
            wparam = win32con.MK_LBUTTON
        else:
            down, up = win32con.WM_RBUTTONDOWN, win32con.WM_RBUTTONUP
            wparam = win32con.MK_RBUTTON

        # Post down then up
        win32gui.PostMessage(hwnd, down, wparam, lparam)
        win32gui.PostMessage(hwnd, up, 0, lparam)
        print(f"✅ Posted {button} click to hwnd={hwnd} at client=({cx},{cy})")
        # capture and save a frame with the click marker for debugging/visual confirmation
        try:
            capture_click_frame(x, y, method='postmessage')
        except Exception:
            pass
        return True
    except Exception as e:
        print("⚠️ send_window_click failed:", e)
        return False


def send_real_click_restore(x, y, button='left', hwnd=None):
    """Move the real cursor, click, then restore the original cursor position.

    This minimizes visible cursor movement by returning the cursor to its prior
    position immediately after the click.
    """

    try:
        # Force game window to foreground if hwnd provided
        if hwnd:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                win32gui.SetForegroundWindow(hwnd)
                win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                time.sleep(0.1)  # Small delay to let window activation complete
            except Exception as e:
                print("⚠️ Could not force foreground:", e)

        # get current cursor pos
        old_pos = win32api.GetCursorPos()
        # move and click using pydirectinput (works on many setups)
        pydirectinput.moveTo(int(x), int(y))
        time.sleep(0.02)
        pydirectinput.click()
        time.sleep(0.02)
        # restore cursor
        try:
            win32api.SetCursorPos(old_pos)
        except Exception:
            # fallback: if SetCursorPos expects tuple
            win32api.SetCursorPos((old_pos[0], old_pos[1]))
        print(f"✅ Performed real click at ({x},{y}) and restored cursor to {old_pos}")
        try:
            capture_click_frame(x, y, method='real_restore')
        except Exception:
            pass
        return True
    except Exception as e:
        print("⚠️ send_real_click_restore failed:", e)
        return False


def send_input_click(x, y, button='left'):
    """Low-level click using SendInput. Moves cursor, sends down/up via SendInput, then restores cursor."""
    try:
        # Save current cursor
        old_pos = win32api.GetCursorPos()
        # Move cursor to target
        win32api.SetCursorPos((int(x), int(y)))
        time.sleep(0.01)

        # Prepare SendInput structures
        PUL = ctypes.POINTER(ctypes.c_ulong)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

        # flags
        if button == 'left':
            down_flag = 0x0002  # MOUSEEVENTF_LEFTDOWN
            up_flag = 0x0004    # MOUSEEVENTF_LEFTUP
        else:
            down_flag = 0x0008  # MOUSEEVENTF_RIGHTDOWN
            up_flag = 0x0010    # MOUSEEVENTF_RIGHTUP

        SendInput = ctypes.windll.user32.SendInput

        inp = INPUT()
        inp.type = 0
        inp.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, None)
        res = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)

        inp.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, None)
        res2 = SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)

        # restore cursor
        try:
            win32api.SetCursorPos(old_pos)
        except Exception:
            win32api.SetCursorPos((old_pos[0], old_pos[1]))

        try:
            capture_click_frame(x, y, method='sendinput')
        except Exception:
            pass

        return True
    except Exception as e:
        print("⚠️ send_input_click failed:", e)
        return False


def send_input_click_abs(x, y, hwnd=None, button='left'):
    """Send mouse input using SendInput with absolute coordinates and focus the window.

    Many games require the window to be foreground and absolute SendInput events
    for synthetic mouse input to be accepted. This helper focuses the hwnd if
    provided, normalizes coordinates to 0..65535, and sends move+down+up.
    """
    try:
        # Optionally bring target window to foreground
        if hwnd:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                win32gui.SetForegroundWindow(hwnd)
            except Exception as e:
                print("⚠️ Could not set foreground window:", e)

        # save and restore cursor
        old_pos = win32api.GetCursorPos()

        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)

        # normalize to 0..65535 (inclusive)
        nx = int(x * 65535 / float(sw - 1))
        ny = int(y * 65535 / float(sh - 1))

        PUL = ctypes.POINTER(ctypes.c_ulong)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]

        SendInput = ctypes.windll.user32.SendInput

        MOUSEEVENTF_MOVE = 0x0001
        MOUSEEVENTF_ABSOLUTE = 0x8000
        if button == 'left':
            down_flag = 0x0002  # LEFTDOWN
            up_flag = 0x0004    # LEFTUP
        else:
            down_flag = 0x0008  # RIGHTDOWN
            up_flag = 0x0010    # RIGHTUP

        # move (absolute)
        inp = INPUT()
        inp.type = 0
        inp.mi = MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, 0, None)
        SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)

        # down (absolute)
        inp.mi = MOUSEINPUT(nx, ny, 0, down_flag | MOUSEEVENTF_ABSOLUTE, 0, None)
        SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)

        # up (absolute)
        inp.mi = MOUSEINPUT(nx, ny, 0, up_flag | MOUSEEVENTF_ABSOLUTE, 0, None)
        SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        time.sleep(0.01)

        # restore cursor visually
        try:
            win32api.SetCursorPos(old_pos)
        except Exception:
            win32api.SetCursorPos((old_pos[0], old_pos[1]))

        try:
            capture_click_frame(x, y, method='sendinput_abs')
        except Exception:
            pass

        return True
    except Exception as e:
        print("⚠️ send_input_click_abs failed:", e)
        return False


def click_with_attach(x, y, hwnd=None, button='left'):
    """Try to attach our input thread to the target window's thread, set foreground,
    then send absolute SendInput. This helps with games that require focus or
    thread attachment to accept synthetic input.
    """
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    try:
        if not hwnd:
            return send_input_click_abs(x, y, hwnd, button)

        # Force to foreground with thread attachment
        pid = ctypes.c_ulong()
        target_thread = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        current_thread = kernel32.GetCurrentThreadId()
        
        # Try to attach input threads
        attached = False
        try:
            attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))
        except Exception:
            pass

        # Force foreground with multiple techniques
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            win32gui.SetForegroundWindow(hwnd)
            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            time.sleep(0.1)
        except Exception as e:
            print("⚠️ Could not force foreground:", e)

        # get current foreground to restore later
        try:
            prev_fore = win32gui.GetForegroundWindow()
        except Exception:
            prev_fore = None

        # get thread ids
        pid = ctypes.c_ulong()
        target_thread = user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        current_thread = kernel32.GetCurrentThreadId()

        attached = False
        try:
            attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))
        except Exception:
            attached = False

        # try to bring to front
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetForegroundWindow(hwnd)
        except Exception as e:
            print("⚠️ Could not SetForegroundWindow:", e)

        time.sleep(0.05)

        ok = send_input_click_abs(x, y, hwnd=hwnd, button=button)

        # detach and restore
        try:
            if attached:
                user32.AttachThreadInput(current_thread, target_thread, False)
        except Exception:
            pass
        try:
            if prev_fore:
                win32gui.SetForegroundWindow(prev_fore)
        except Exception:
            pass

        return ok
    except Exception as e:
        print("⚠️ click_with_attach failed:", e)
        return False

def capture_region(x1, y1, x2, y2):
    """Grab a screenshot of the given absolute screen rectangle and return a BGR numpy array.
    
    Strategy:
      1. Try DXCam (fast, GPU-backed) with a few quick retries.
      2. If DXCam fails, fall back to Pillow (ImageGrab) or pyautogui.screenshot.
      
    For fullscreen mode, coordinates are relative to the entire screen.
    For windowed mode, coordinates are relative to the window.

    This function is defensive because some systems / drivers block DX capture.
    Keeping a reliable Pillow fallback ensures the bot still works.
    """
    # Get screen dimensions
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
    
    # Ensure inputs are numbers and convert to integers
    try:
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    except (TypeError, ValueError) as e:
        print(f"⚠️ Invalid coordinates (must be numbers): {e}")
        return None
    
    # Clamp coordinates to screen bounds
    x1 = max(0, min(x1, screen_width - 1))
    y1 = max(0, min(y1, screen_height - 1))
    x2 = max(x1 + 1, min(x2, screen_width))
    y2 = max(y1 + 1, min(y2, screen_height))
    
    # Calculate dimensions
    w = x2 - x1
    h = y2 - y1
    
    # Validate minimum size
    MIN_SIZE = 30  # Minimum size in pixels
    if w < MIN_SIZE or h < MIN_SIZE:
        print(f"⚠️ Region too small: {w}x{h}. Please resize the game window to at least {MIN_SIZE}x{MIN_SIZE}.")
        return None
        
    # Validate maximum size
    if w > screen_width or h > screen_height:
        print(f"⚠️ Region too large: {w}x{h}. Maximum allowed is {screen_width}x{screen_height}")
        return None
        
    # Ensure region is valid for DXCam (must be within 1920x1080)
    if w > 1920 or h > 1080:
        print(f"⚠️ Region too large: {w}x{h}. Scaling down to fit DXCam limits.")
        aspect = w / h
        if aspect > (1920/1080):  # wider than 16:9
            w = 1920
            h = int(1920 / aspect)
        else:
            h = 1080
            w = int(1080 * aspect)
        # Adjust x2,y2 to maintain position while fitting size limits
        x2 = x1 + w
        y2 = y1 + h

    # Try DXCam first (fast). Retry a few times because DX capture can fail intermittently.
    frame = None
    dx_attempts = 3
    
    # Ensure region coordinates are integers for DXCam
    region = (int(x1), int(y1), int(w), int(h))
    
    for i in range(dx_attempts):
        try:
            # Validate region dimensions before capture
            if not (0 < region[2] <= 1920 and 0 < region[3] <= 1080):
                raise ValueError(f"Invalid region dimensions: {region[2]}x{region[3]}")
                
            frame = camera.grab(region=region)
            if frame is not None and frame.size > 0:
                # Validate captured frame
                if frame.shape[0] == 0 or frame.shape[1] == 0:
                    raise ValueError(f"Captured frame has invalid dimensions: {frame.shape}")
                    
                # DXCam returns RGB by default for some versions; normalize to BGR for OpenCV
                try:
                    if isinstance(frame, (np.ndarray,)) and frame.ndim == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"⚠️ Color conversion warning: {e}")
                return frame
        except Exception as e:
            print(f"⚠️ DXCam grab attempt {i+1}/{dx_attempts} failed: {e}")
            if "Invalid Region" in str(e):
                print(f"Region details: x={region[0]}, y={region[1]}, w={region[2]}, h={region[3]}")
        # small backoff between retries
        time.sleep(0.05)

    # DXCam failed repeatedly — try Pillow fallback (ImageGrab) or pyautogui
    print("⚠️ DXCam failed. Falling back to Pillow/pyautogui screenshot...")
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(bbox=(int(x1), int(y1), int(x2), int(y2)))
        if img is None:
            raise RuntimeError("Pillow ImageGrab.grab returned None")
        arr = np.array(img)
        # Pillow gives RGB; convert to BGR
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        return arr
    except Exception as e:
        print(f"⚠️ Pillow fallback failed: {e}")

    # Last resort: pyautogui (which uses Pillow internally but may be available differently)
    try:
        img = pyautogui.screenshot(region=(int(x1), int(y1), int(w), int(h)))
        if img is None:
            raise RuntimeError("pyautogui.screenshot returned None")
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        return arr
    except Exception as e:
        print(f"⚠️ pyautogui fallback failed: {e}")

    print("❌ Failed to capture region by any method.")
    return None

def get_weight(frame):
    img_crop_weight = crop_image(SAVED_REGIONS['weight_percent_region'], frame, offset=[15,15,2,15])

    print("Saved crop_debug.png")
    weight = extract_number_from_image(img_crop_weight, debug_save=True, region_label="weight", allow_decimals=True)

    if 0< weight <= 100:
        return weight
    else:
        print("Weight is not valid")
        return 80    
    


def screen_state(frame, game_state=None):
    """Analyze the screen state from the captured frame.

    Returns the name of the first matched region image found in the frame.
    """
    print("*************screen_state*****************")
    # Search for any saved region images in ./region folder
    try:
        print("screen_state: analyzing frame for regions...")
        if(search_region_in_frame('sleep_show_region', frame)):
            game_state["region_name"] = "sleep_show_region"
            print("This is a sleep state")
            return game_state
        if (search_region_in_frame(get_resource_path('region/buttons/general_mer_region.png'), frame)):

            click_saved_region_center("sales_address_region")
        elif info := detect_ui_buttons(frame, do_click=True, debug=False, debug_out=None, x_offset=0, y_offset=0, hwnd=None):
            game_state["button"] = info["filename"]
            return game_state
            
    except Exception as e:
        print(f"Error in screen_state: {str(e)}")
        pass

    # If no region/state was detected, return None so callers can ignore updates
    return None

def capture_click_frame(x, y, method='unknown', out_path=None, radius=18):
    """Capture the full screen and save an annotated image showing the click point and method.

    method: short label such as 'postmessage', 'real_restore', or 'virtual'
    """
    try:
        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        frame = camera.grab(region=(0, 0, sw, sh))
        if frame is None:
            return None
        out = frame.copy()

        # choose color by method
        method_lower = (method or '').lower()
        if 'post' in method_lower or 'window' in method_lower:
            color = (0, 255, 0)  # green
        elif 'real' in method_lower or 'restore' in method_lower:
            color = (0, 0, 255)  # red
        elif 'virtual' in method_lower or 'pydirect' in method_lower:
            color = (0, 255, 255)  # yellow
        else:
            color = (255, 0, 0)  # blue for unknown

        # Draw circle, crosshair and label
        cv2.circle(out, (int(x), int(y)), radius, color, 3)
        cv2.line(out, (int(x)-radius, int(y)), (int(x)+radius, int(y)), color, 2)
        cv2.line(out, (int(x), int(y)-radius), (int(x), int(y)+radius), color, 2)
        label = f"click:{method} ({x},{y})"
        cv2.putText(out, label, (max(10, int(x)-radius), max(20, int(y)-radius-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if out_path is None:
            ts = int(time.time())
            safe_method = ''.join(c if c.isalnum() else '_' for c in (method or 'unknown'))
            out_path = os.path.join("screenshots", f"click_debug_{safe_method}_{ts}.png")
        print(f"🖼 Click debug image saved to {out_path}")
        return out_path
    except Exception as e:
        print("⚠️ capture_click_frame failed:", e)
        return None


def show_saved_region_marker(name, out_path=None):
    """Draw a marker for the saved region center and save a debug screenshot.

    Use this to visually verify that the saved-region coordinates map to the
    expected screen location (helps diagnose DPI / scaling / offset issues).
    """
    try:
        if name not in SAVED_REGIONS:
            print(f"⚠️ Region '{name}' not found in SAVED_REGIONS.")
            return None
        r = SAVED_REGIONS[name]
        # Map region coordinates to screen coordinates based on coordinate type
        try:
            wi = get_window_info() or {}
            if r.get('_coord') == 'rel':
                # Window-relative coordinates: add window position to get absolute screen coords
                if wi and 'x' in wi and 'y' in wi:
                    win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                    rel_x = int(r.get('x', 0))
                    rel_y = int(r.get('y', 0))
                    rel_w = int(r.get('w', 0))
                    rel_h = int(r.get('h', 0))
                    x1 = win_x + rel_x
                    y1 = win_y + rel_y
                    x2 = x1 + rel_w
                    y2 = y1 + rel_h
                    cx = x1 + rel_w // 2
                    cy = y1 + rel_h // 2
                else:
                    # Fallback if window info not available
                    cx = int(r.get('x', 0) + r.get('w', 0) // 2)
                    cy = int(r.get('y', 0) + r.get('h', 0) // 2)
                    x1 = int(r.get('x', 0))
                    y1 = int(r.get('y', 0))
                    x2 = x1 + int(r.get('w', 0))
                    y2 = y1 + int(r.get('h', 0))
            elif r.get('_coord') == 'ref':
                # Reference coordinates: map to current window size
                if wi and 'width' in wi and 'height' in wi:
                    win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                    win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                    cx = win_x + int((r.get('x', 0) + r.get('w', 0) / 2.0) * win_w / float(REF_WIDTH))
                    cy = win_y + int((r.get('y', 0) + r.get('h', 0) / 2.0) * win_h / float(REF_HEIGHT))
                    x1 = win_x + int(r.get('x', 0) * win_w / float(REF_WIDTH))
                    y1 = win_y + int(r.get('y', 0) * win_h / float(REF_HEIGHT))
                    x2 = win_x + int((r.get('x', 0) + r.get('w', 0)) * win_w / float(REF_WIDTH))
                    y2 = win_y + int((r.get('y', 0) + r.get('h', 0)) * win_h / float(REF_HEIGHT))
                else:
                    # fallback: map against full screen
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)
                    cx = int((r.get('x', 0) + r.get('w', 0) / 2.0) * sw / float(REF_WIDTH))
                    cy = int((r.get('y', 0) + r.get('h', 0) / 2.0) * sh / float(REF_HEIGHT))
                    x1 = int(r.get('x', 0) * sw / float(REF_WIDTH))
                    y1 = int(r.get('y', 0) * sh / float(REF_HEIGHT))
                    x2 = int((r.get('x', 0) + r.get('w', 0)) * sw / float(REF_WIDTH))
                    y2 = int((r.get('y', 0) + r.get('h', 0)) * sh / float(REF_HEIGHT))
            else:
                # Absolute screen coordinates (legacy format)
                cx = int(r.get('x', 0) + r.get('w', 0) // 2)
                cy = int(r.get('y', 0) + r.get('h', 0) // 2)
                x1 = int(r.get('x', 0))
                y1 = int(r.get('y', 0))
                x2 = x1 + int(r.get('w', 0))
                y2 = y1 + int(r.get('h', 0))
        except Exception:
            cx = int(r.get('x', 0) + r.get('w', 0) // 2)
            cy = int(r.get('y', 0) + r.get('h', 0) // 2)
            x1 = int(r.get('x', 0))
            y1 = int(r.get('y', 0))
            x2 = x1 + int(r.get('w', 0))
            y2 = y1 + int(r.get('h', 0))

        user32 = ctypes.windll.user32
        sw = user32.GetSystemMetrics(0)
        sh = user32.GetSystemMetrics(1)
        frame = camera.grab(region=(0, 0, sw, sh))
        if frame is None:
            print('⚠️ camera.grab returned None')
            return None

        out = frame.copy()
        # draw region rect and center marker
        x1, y1 = int(r.get('x', 0)), int(r.get('y', 0))
        x2, y2 = x1 + int(r.get('w', 0)), y1 + int(r.get('h', 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(out, (cx, cy), 12, (0, 0, 255), 3)
        cv2.putText(out, f"{name} -> ({cx},{cy})", (max(10, cx-80), max(30, cy-20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        screenshots = os.path.join(os.path.dirname(__file__), 'screenshots')
        os.makedirs(screenshots, exist_ok=True)
        if out_path is None:
            ts = int(time.time())
            out_path = os.path.join(screenshots, f'region_marker_{name}_{ts}.png')
        print(f"🖼 Region marker saved to {out_path} (center at {cx},{cy})")
        return out_path
    except Exception as e:
        print('⚠️ show_saved_region_marker failed:', e)
        return None

def crop_image(r, frame, debug=False, offset=0):

    """Crop a region from a frame, handling both absolute and reference coordinates.

    Args:
        r: Region dict with x,y,w,h (and optional _coord='ref')
        frame: BGR image array
        debug: If True, save visualization showing coordinate mapping
        offset: int or tuple to expand/shrink the crop area. Supported shapes:
            - int: symmetric padding applied to all sides (positive to expand, negative to shrink)
            - (dx, dy): symmetric padding horizontally and vertically
            - (left, top, right, bottom): per-side padding

    Returns:
        BGR numpy array of the cropped region (may be empty if out of bounds)
    """
    # Get window info for coordinate mapping
    wi = get_window_info()
    if wi is None:
        print("⚠️ Window not found for coordinate mapping")
        return None
        
    win_x, win_y = int(wi['x']), int(wi['y'])

    # Handle window-relative coordinates (new format - works regardless of window position/size)
    if r.get('_coord') == 'rel':
        # Region coordinates are relative to window, but may need scaling if window size changed
        win_w, win_h = int(wi['width']), int(wi['height'])
        
        # Check if we have reference window size (saved when region was created)
        ref_w = r.get('_ref_width', 0)
        ref_h = r.get('_ref_height', 0)
        
        if ref_w > 0 and ref_h > 0:
            # Window size changed - scale coordinates proportionally
            scale_x = win_w / float(ref_w)
            scale_y = win_h / float(ref_h)
            # Scale the region coordinates
            x = int(r.get('x', 0) * scale_x)
            y = int(r.get('y', 0) * scale_y)
            w = int(r.get('w', r.get('width', 0)) * scale_x)
            h = int(r.get('h', r.get('height', 0)) * scale_y)
            
            # Ensure scaled coordinates don't exceed window bounds
            x = max(0, min(x, win_w - 1))
            y = max(0, min(y, win_h - 1))
            w = max(1, min(w, win_w - x))
            h = max(1, min(h, win_h - y))
            
            if debug:
                print(f"Window-relative coords scaled: ({r.get('x', 0)},{r.get('y', 0)},{r.get('w', 0)},{r.get('h', 0)}) @ {ref_w}x{ref_h} -> ({x},{y},{w},{h}) @ {win_w}x{win_h}, scale=({scale_x:.3f},{scale_y:.3f})")
        else:
            # No reference size - use coordinates as-is (assumes window size hasn't changed)
            x = int(r.get('x', 0))
            y = int(r.get('y', 0))
            w = int(r.get('w', r.get('width', 0)))
            h = int(r.get('h', r.get('height', 0)))
            
            # Ensure coordinates don't exceed window bounds
            x = max(0, min(x, win_w - 1))
            y = max(0, min(y, win_h - 1))
            w = max(1, min(w, win_w - x))
            h = max(1, min(h, win_h - y))
            
            if debug:
                print(f"Window-relative coords: ({x},{y},{w},{h}) [window at {win_x},{win_y}, size {win_w}x{win_h}]")
    # If region uses reference coordinates (old format), map to current window size
    elif r.get('_coord') == 'ref':
        win_w, win_h = int(wi['width']), int(wi['height'])
        # Map reference coords (1920x1030) to current window size
        # First convert to absolute screen coords, then to frame-relative
        abs_x = win_x + int(r['x'] * win_w / float(REF_WIDTH))
        abs_y = win_y + int(r['y'] * win_h / float(REF_HEIGHT))
        w = int(r['w'] * win_w / float(REF_WIDTH))
        h = int(r['h'] * win_h / float(REF_HEIGHT))
        # Convert to frame-relative
        x = abs_x - win_x
        y = abs_y - win_y
        if debug:
            print(f"Mapped ref coords ({r['x']},{r['y']},{r['w']},{r['h']}) → frame-relative ({x},{y},{w},{h})")
    else:
        # Region is in absolute screen coordinates (legacy format); support multiple saved formats
        # (x,y,w,h) or (x1,y1,x2,y2) or (x,y,width,height)
        if 'x' in r and ('w' in r or 'width' in r):
            x = int(r.get('x', 0))  # Absolute screen X
            y = int(r.get('y', 0))  # Absolute screen Y
            w = int(r.get('w', r.get('width', 0)))
            h = int(r.get('h', r.get('height', 0)))
        elif 'x1' in r and 'y1' in r:
            x1 = int(r.get('x1', 0)); y1 = int(r.get('y1', 0))
            x2 = int(r.get('x2', x1 + int(r.get('width', 0) or 1)))
            y2 = int(r.get('y2', y1 + int(r.get('height', 0) or 1)))
            x = x1; y = y1; w = max(1, x2 - x1); h = max(1, y2 - y1)
        else:
            # best-effort fallback
            x = int(r.get('x', 0))
            y = int(r.get('y', 0))
            w = int(r.get('w', r.get('width', 0)))
            h = int(r.get('h', r.get('height', 0)))
        if debug:
            print(f"Absolute coords: screen=({x},{y},{w},{h})")
        # Convert absolute screen coords to coordinates relative to frame origin
        x = x - win_x
        y = y - win_y
        if debug:
            print(f"Frame-relative: ({x},{y},{w},{h}) [origin: {win_x},{win_y}]")

    # Normalize offset into per-side integers
    try:
        if isinstance(offset, (list, tuple)):
            if len(offset) == 2:
                left_off = right_off = int(offset[0])
                top_off = bottom_off = int(offset[1])
            elif len(offset) == 4:
                left_off, top_off, right_off, bottom_off = map(int, offset)
            else:
                # unexpected shape, treat as scalar
                left_off = top_off = right_off = bottom_off = int(offset[0])
        else:
            left_off = top_off = right_off = bottom_off = int(offset)
    except Exception:
        left_off = top_off = right_off = bottom_off = 0

    # Apply offset (offset is in pixels, positive expands, negative shrinks)
    x = x - left_off
    y = y - top_off
    w = w + left_off + right_off
    h = h + top_off + bottom_off

    # Ensure crop region fits within frame bounds
    fh, fw = frame.shape[:2]
    x0 = max(0, min(fw-1, x))
    y0 = max(0, min(fh-1, y))
    x1 = max(0, min(fw, x0 + max(1, w)))
    y1 = max(0, min(fh, y0 + max(1, h)))

    if debug:
        # Save debug visualization
        try:
            debug_img = frame.copy()
            # Draw the computed crop region
            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            # Add text with coordinates
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            color = (0, 255, 0)
            thickness = 2
            
            info_lines = [
                f"Region type: {'reference-scaled' if r.get('_coord')=='ref' else 'absolute'}",
                f"Screen coords: ({x},{y}) {w}x{h}",
                f"Frame coords: ({x0},{y0}) {x1-x0}x{y1-y0}",
                f"Window origin: ({win_x},{win_y})",
                f"Offset applied (L,T,R,B): ({left_off},{top_off},{right_off},{bottom_off})"
            ]
            
            y_text = y0 - 10
            for i, line in enumerate(info_lines):
                y_text = max(20 + i*20, y0 - 10 + i*20)
                cv2.putText(debug_img, line, (x0, y_text), font, scale, (0,0,0), thickness+1)
                cv2.putText(debug_img, line, (x0, y_text), font, scale, color, thickness)
                
            # Save the debug image
            os.makedirs('debug_crops', exist_ok=True)
            ts = int(time.time())
            debug_path = os.path.join('debug_crops', f'crop_debug_{ts}.png')
            print(f"Saved debug visualization to {debug_path}")
        except Exception as e:
            print(f"⚠️ Debug visualization failed: {e}")
    # Validate frame and coordinates before cropping
    if frame is None:
        print("⚠️ Cannot crop: frame is None")
        return None
        
    try:
        if not isinstance(frame, np.ndarray):
            print("⚠️ Cannot crop: frame is not a numpy array")
            return None
            
        # Ensure coordinates are within bounds
        h, w = frame.shape[:2]
        x0 = max(0, min(x0, w-1))
        y0 = max(0, min(y0, h-1))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        
        # Check if resulting crop would be valid
        if x1 <= x0 or y1 <= y0:
            print(f"⚠️ Invalid crop dimensions: {x1-x0}x{y1-y0}")
            return None
            
        # Perform the crop
        crop = frame[y0:y1, x0:x1].copy()  # Make a copy to ensure we have our own data
        
        if crop.size == 0:
            print("⚠️ Crop resulted in empty array")
            return None
            
        return crop
        
    except Exception as e:
        print(f"⚠️ Error during crop: {e}")
        return None

def click_saved_region_center(name, method='auto', window_info=None):
    """Perform a test click at the saved-region center using the preferred method.

    method: 'auto' (respect USE_REAL_MOUSE), 'window', 'sendinput', 'real_restore'
    window_info: Optional window info dict (x, y, width, height, hwnd, name). 
                 If provided, uses this window instead of searching for one.
                 Critical for multi-instance support.
    """
    try:
        # Input validation
        if not name:
            print("⚠️ Invalid region name (empty or None)")
            return False
            
        if name not in SAVED_REGIONS:
            print(f"⚠️ Region '{name}' not found in SAVED_REGIONS.")
            return False
            
        # Get region info
        r = SAVED_REGIONS[name]
        if not r:
            print(f"⚠️ Empty region data for '{name}'")
            return False

        # Normalize region dict into coordinates (x,y,w,h)
        def normalize_region(reg):
            # Handle window-relative coordinates (new format - works regardless of window position/size)
            if reg.get('_coord') == 'rel':
                normalized = {
                    'x': int(reg.get('x', 0)),
                    'y': int(reg.get('y', 0)),
                    'w': int(reg.get('w', 0)),
                    'h': int(reg.get('h', 0)),
                    '_coord': 'rel'
                }
                # Preserve reference window size for scaling
                if '_ref_width' in reg:
                    normalized['_ref_width'] = int(reg.get('_ref_width', 0))
                if '_ref_height' in reg:
                    normalized['_ref_height'] = int(reg.get('_ref_height', 0))
                return normalized
            
            # Handle reference-coord regions (keeps x/y/w/h as reference units)
            if reg.get('_coord') == 'ref':
                return {
                    'x': int(reg.get('x', 0)),
                    'y': int(reg.get('y', 0)),
                    'w': int(reg.get('w', 0)),
                    'h': int(reg.get('h', 0)),
                    '_coord': 'ref'
                }

            # Common absolute formats (legacy - for backward compatibility)
            if 'x' in reg and 'w' in reg:
                return {
                    'x': int(reg.get('x', 0)), 'y': int(reg.get('y', 0)),
                    'w': int(reg.get('w', 0)), 'h': int(reg.get('h', 0)),
                    '_coord': 'abs'
                }

            # legacy: x1,y1,x2,y2
            if 'x1' in reg and 'y1' in reg and 'x2' in reg and 'y2' in reg:
                x1 = int(reg.get('x1', 0)); y1 = int(reg.get('y1', 0))
                x2 = int(reg.get('x2', x1 + 1)); y2 = int(reg.get('y2', y1 + 1))
                return {'x': x1, 'y': y1, 'w': max(1, x2 - x1), 'h': max(1, y2 - y1), '_coord': 'abs'}

            # alternative: x,y,width,height or width/height keys
            if 'width' in reg or 'height' in reg or 'w' in reg or 'h' in reg:
                x = int(reg.get('x', reg.get('x1', 0)))
                y = int(reg.get('y', reg.get('y1', 0)))
                w = int(reg.get('w', reg.get('width', 0)))
                h = int(reg.get('h', reg.get('height', 0)))
                return {'x': x, 'y': y, 'w': w, 'h': h, '_coord': 'abs'}

            # Fallback: attempt best-effort mapping
            x = int(reg.get('x', 0))
            y = int(reg.get('y', 0))
            w = int(reg.get('w', reg.get('width', 0)))
            h = int(reg.get('h', reg.get('height', 0)))
            return {'x': x, 'y': y, 'w': w, 'h': h, '_coord': reg.get('_coord', 'abs')}

        nr = normalize_region(r)

        # Use provided window_info if available (for multi-instance support), otherwise find window
        if window_info is not None and isinstance(window_info, dict):
            wi = window_info
        else:
            # Attempt to find the game window info (used for mapping reference coords)
            try:
                wi = get_window_info()
                if not wi:
                    print("⚠️ Could not find game window")
                    return False
            except Exception as e:
                print(f"⚠️ Error getting window info: {e}")
                return False

        hwnd = wi.get('hwnd')
        if not hwnd:
            print("⚠️ Invalid window handle")
            return False

        # Check if window is too small for accurate clicking
        win_w = int(wi.get('width', 0))
        win_h = int(wi.get('height', 0))
        MIN_WINDOW_WIDTH = 800
        MIN_WINDOW_HEIGHT = 600
        
        if win_w < MIN_WINDOW_WIDTH or win_h < MIN_WINDOW_HEIGHT:
            print(f"⚠️ WARNING: Game window is too small ({win_w}x{win_h}). Minimum recommended size is {MIN_WINDOW_WIDTH}x{MIN_WINDOW_HEIGHT}.")
            print(f"   Click accuracy may be reduced. Please resize the game window for better results.")
            # Continue anyway, but warn the user

        # Ensure width/height are present
        region_width = int(nr.get('w', 0))
        region_height = int(nr.get('h', 0))

        # Validate/expand region size if too small
        MIN_SIZE = 30
        if region_width < MIN_SIZE or region_height < MIN_SIZE:
            cx0 = nr.get('x', 0) + region_width / 2.0
            cy0 = nr.get('y', 0) + region_height / 2.0
            nr['w'] = MIN_SIZE
            nr['h'] = MIN_SIZE
            nr['x'] = int(cx0 - MIN_SIZE / 2.0)
            nr['y'] = int(cy0 - MIN_SIZE / 2.0)
            region_width = nr['w']
            region_height = nr['h']
            print(f"⚠️ Expanded small region '{name}' to minimum size: {MIN_SIZE}x{MIN_SIZE}")

        # Compute click center in screen coordinates
        if nr.get('_coord') == 'rel':
            # Window-relative coordinates: convert to absolute screen coordinates
            # If window size changed, scale coordinates proportionally
            try:
                if wi and 'width' in wi and 'height' in wi:
                    win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                    win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                    
                    # Check if we have reference window size (saved when region was created)
                    ref_w = nr.get('_ref_width', 0)
                    ref_h = nr.get('_ref_height', 0)
                    
                    if ref_w > 0 and ref_h > 0:
                        # Window size changed - scale coordinates proportionally
                        scale_x = win_w / float(ref_w)
                        scale_y = win_h / float(ref_h)
                        
                        # Warn if scaling is extreme (window much smaller or larger)
                        if scale_x < 0.5 or scale_x > 2.0 or scale_y < 0.5 or scale_y > 2.0:
                            print(f"⚠️ Extreme window size change detected for region '{name}': ref({ref_w}x{ref_h}) -> current({win_w}x{win_h}), scale=({scale_x:.3f},{scale_y:.3f})")
                            print(f"   Click accuracy may be reduced. Consider re-saving regions with current window size.")
                        
                        # Scale the region coordinates
                        rel_x = int(nr.get('x', 0) * scale_x)
                        rel_y = int(nr.get('y', 0) * scale_y)
                        rel_w = int(nr.get('w', 0) * scale_x)
                        rel_h = int(nr.get('h', 0) * scale_y)
                        
                        # Ensure scaled region doesn't exceed window bounds
                        rel_x = max(0, min(rel_x, win_w - 1))
                        rel_y = max(0, min(rel_y, win_h - 1))
                        rel_w = min(rel_w, win_w - rel_x)
                        rel_h = min(rel_h, win_h - rel_y)
                        
                        # Calculate center in scaled window-relative coordinates, then add window offset
                        cx = win_x + int(rel_x + rel_w / 2.0)
                        cy = win_y + int(rel_y + rel_h / 2.0)
                        # Debug logging for scaling
                        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
                            print(f"📐 Scaled region '{name}': ref({ref_w}x{ref_h}) -> current({win_w}x{win_h}), scale=({scale_x:.3f},{scale_y:.3f}), rel({nr.get('x', 0)},{nr.get('y', 0)}) -> ({rel_x},{rel_y})")
                    else:
                        # No reference size - use coordinates as-is (assumes window size hasn't changed)
                        # This is for backward compatibility with old regions
                        rel_x = nr.get('x', 0)
                        rel_y = nr.get('y', 0)
                        rel_w = nr.get('w', 0)
                        rel_h = nr.get('h', 0)
                        # Calculate center in window-relative coordinates, then add window offset
                        cx = win_x + int(rel_x + rel_w / 2.0)
                        cy = win_y + int(rel_y + rel_h / 2.0)
                        # Warn if region might be out of bounds
                        if rel_x + rel_w > win_w or rel_y + rel_h > win_h:
                            print(f"⚠️ Region '{name}' coordinates ({rel_x},{rel_y},{rel_w},{rel_h}) exceed window size ({win_w}x{win_h}) - region may need re-saving with current window size")
                else:
                    # Fallback if window info not available
                    cx = int(nr.get('x', 0) + nr.get('w', 0) // 2)
                    cy = int(nr.get('y', 0) + nr.get('h', 0) // 2)
            except Exception as e:
                print(f"⚠️ Error converting window-relative coords: {e}")
                cx = int(nr.get('x', 0) + nr.get('w', 0) // 2)
                cy = int(nr.get('y', 0) + nr.get('h', 0) // 2)
        elif nr.get('_coord') == 'ref':
            # Map reference coords to actual screen/window coordinates
            try:
                if wi and 'width' in wi and 'height' in wi:
                    win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                    win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                    cx = win_x + int((nr.get('x', 0) + nr.get('w', 0) / 2.0) * win_w / float(REF_WIDTH))
                    cy = win_y + int((nr.get('y', 0) + nr.get('h', 0) / 2.0) * win_h / float(REF_HEIGHT))
                else:
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)
                    cx = int((nr.get('x', 0) + nr.get('w', 0) / 2.0) * sw / float(REF_WIDTH))
                    cy = int((nr.get('y', 0) + nr.get('h', 0) / 2.0) * sh / float(REF_HEIGHT))
            except Exception:
                cx = int(nr.get('x', 0) + nr.get('w', 0) // 2)
                cy = int(nr.get('y', 0) + nr.get('h', 0) // 2)
        else:
            # nr contains absolute screen coords (legacy format)
            # Try to convert to window-relative if the region appears to be within the current window
            abs_x = int(nr.get('x', 0))
            abs_y = int(nr.get('y', 0))
            abs_w = int(nr.get('w', 0))
            abs_h = int(nr.get('h', 0))
            abs_cx = abs_x + abs_w // 2
            abs_cy = abs_y + abs_h // 2
            
            # Check if this absolute coordinate region is within the current window bounds
            if wi and 'x' in wi and 'y' in wi and 'width' in wi and 'height' in wi:
                win_x = int(wi.get('x', 0))
                win_y = int(wi.get('y', 0))
                win_w = int(wi.get('width', 0))
                win_h = int(wi.get('height', 0))
                
                # For legacy absolute coordinates, try to detect if they were meant to be window-relative
                # Strategy: Check if the region coordinates are "reasonable" for the current window
                # If the region's relative position within a typical window size matches the current window,
                # assume it was meant to be window-relative
                
                region_right = abs_x + abs_w
                region_bottom = abs_y + abs_h
                win_right = win_x + win_w
                win_bottom = win_y + win_h
                
                # Check for overlap: region overlaps window if they intersect
                overlaps = not (region_right < win_x or abs_x > win_right or 
                               region_bottom < win_y or abs_y > win_bottom)
                
                # Also check if region is "close" to window (within reasonable distance)
                # This handles cases where window moved but region coordinates are still valid relative to window
                distance_threshold = 500  # pixels
                x_distance = min(abs(abs_x - win_x), abs(region_right - win_right), 
                               abs(abs_x - win_right), abs(region_right - win_x))
                y_distance = min(abs(abs_y - win_y), abs(region_bottom - win_bottom),
                               abs(abs_y - win_bottom), abs(region_bottom - win_y))
                is_close = x_distance < distance_threshold and y_distance < distance_threshold
                
                if overlaps or is_close:
                    # Region is related to current window - convert to window-relative
                    rel_x = abs_x - win_x
                    rel_y = abs_y - win_y
                    # Clamp to window bounds to handle edge cases
                    rel_x = max(0, min(rel_x, win_w - 1))
                    rel_y = max(0, min(rel_y, win_h - 1))
                    # Recalculate width/height if region extends beyond window
                    rel_w = min(abs_w, win_w - rel_x)
                    rel_h = min(abs_h, win_h - rel_y)
                    # Calculate center using window-relative coordinates
                    cx = win_x + int(rel_x + rel_w / 2.0)
                    cy = win_y + int(rel_y + rel_h / 2.0)
                    print(f"🔄 Converted absolute region '{name}' to window-relative: abs({abs_x},{abs_y}) -> rel({rel_x},{rel_y}) -> screen({cx},{cy})")
                else:
                    # Region is far from window - might be for a different window or screen area
                    # Use absolute coordinates as-is
                    cx = abs_cx
                    cy = abs_cy
                    print(f"⚠️ Region '{name}' at abs({abs_x},{abs_y}) is far from current window at ({win_x},{win_y}) size {win_w}x{win_h}, using absolute coords")
            else:
                # No window info, use absolute coordinates as-is
                cx = abs_cx
                cy = abs_cy

        print(f"🖱 Test clicking region '{name}' at ({cx},{cy}) using method={method}")
        if method == 'window':
            return send_window_click(hwnd, cx, cy)
        if method == 'sendinput':
            return send_input_click_abs(cx, cy, hwnd=hwnd)
        if method == 'real_restore':
            return send_real_click_restore(cx, cy)

        # auto: respect USE_REAL_MOUSE fallback logic
        if USE_REAL_MOUSE:
            ok = send_real_click_restore(cx, cy)
            if not ok and hwnd is not None:
                ok = send_window_click(hwnd, cx, cy)
            return ok
        else:
            if hwnd is not None:
                ok = send_window_click(hwnd, cx, cy)
                if not ok:
                    return send_input_click_abs(cx, cy, hwnd=hwnd)
            else:
                try:
                    pydirectinput.click(cx, cy)
                    return True
                except Exception:
                    pyautogui.click(cx, cy)
                    return True
        return False
    except Exception as e:
        print('⚠️ click_saved_region_center failed:', e)
        return False
 
def normalize_gray(img):
    """Normalize contrast and brightness to help template matching."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img

def detect_indi_button_optimized(template, filename, gray, scales, tm_threshold=0.85, region=None, do_click=False, debug=False, debug_out=None, x_offset=0, y_offset=0, hwnd=None):
    """Optimized version that accepts pre-loaded template and pre-converted grayscale frame."""
    global CLICK_OFFSET_X, CLICK_OFFSET_Y, USE_REAL_MOUSE
    if template is None:
        return None

    th, tw = template.shape[:2]
    best_val = -1.0
    best_loc = None
    best_size = (tw, th)
    
    # Handle region cropping if needed
    draw_base_x, draw_base_y = 0, 0
    working_gray = gray
    if region is not None:
        # For region handling, we need the original BGR frame, so this path is less optimized
        # But we'll keep it for compatibility
        return None  # Fall back to original function for region-based detection
    
    img_h, img_w = working_gray.shape[:2]
    
    # Optimized multi-scale template matching with fewer scales
    for s in scales:
        nw, nh = int(tw * s), int(th * s)
        
        # Quick validation
        if nw <= 0 or nh <= 0 or nw < 8 or nh < 8 or nw > img_w or nh > img_h:
            continue
            
        try:
            resized = cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(working_gray, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_size = (nw, nh)
                # Early exit if we find a very confident match
                if best_val >= 0.95:
                    break
        except Exception:
            continue

    if best_val >= tm_threshold and best_loc is not None:
        bx, by = best_loc
        bw, bh = best_size
        cx, cy = bx + bw // 2, by + bh // 2
        sx, sy = x_offset + cx, y_offset + cy
        
        # Convert frame-relative coordinates to absolute screen coordinates
        try:
            wi = get_window_info()
            if wi and 'x' in wi and 'y' in wi:
                sx = sx + int(wi['x'])
                sy = sy + int(wi['y'])
        except Exception:
            pass
        info = {"filename": filename, "method": "template", "confidence": float(best_val), "x": int(sx), "y": int(sy)}
        
        
        if do_click:
            sx = sx + CLICK_OFFSET_X
            sy = sy + CLICK_OFFSET_Y
            sx, sy, is_valid = validate_click_coordinates(sx, sy)
            if not is_valid:
                return info
            if USE_REAL_MOUSE:
                ok = send_real_click_restore(sx, sy)
                if not ok and hwnd is not None:
                    ok = send_window_click(hwnd, sx, sy)
                if not ok:
                    try:
                        pydirectinput.click(sx, sy)
                    except Exception:
                        pyautogui.click(sx, sy)
            else:
                if hwnd is not None:
                    ok = send_window_click(hwnd, sx, sy)
                    if not ok:
                        try:
                            pydirectinput.click(sx, sy)
                        except Exception:
                            pyautogui.click(sx, sy)
                else:
                    try:
                        pydirectinput.click(sx, sy)
                    except Exception:
                        pyautogui.click(sx, sy)
            time.sleep(0.3)
        
        return info

    return None

def detect_indi_button(tpl_path, gray_frame, scales, tm_threshold=0.85, region=None, do_click=False, debug=False, debug_out=None, x_offset=0, y_offset=0, hwnd=None):
    global CLICK_OFFSET_X, CLICK_OFFSET_Y, USE_REAL_MOUSE
    template = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
    filename = os.path.basename(tpl_path)
    if template is None:
        print(f"⚠️ Failed to load template: {tpl_path}")
        return None

    th, tw = template.shape[:2]

    best_val = -1.0
    best_loc = None
    best_size = (tw, th)
    if gray_frame is not None:
        if len(gray_frame.shape) == 3:
            gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
        else:
            print("⚠️'gray_frame' is None; cannot crop region.")
            return None

    # Multi-scale template matching
    # If a region was provided, crop the working images to that region first.
    # region may be a region-name (string), a dict with x,y,w,h, or a tuple (x,y,w,h).
    draw_base_x, draw_base_y = 0, 0
    if region is not None:
        # need a full gray_frame to crop from
        if gray_frame is None:
            print("⚠️ detect_indi_button: region specified but 'gray_frame' is None; cannot crop region.")
            return None
        # resolve region dict
        r = None
        if isinstance(region, str):
            r = SAVED_REGIONS.get(region)
        elif isinstance(region, (list, tuple)) and len(region) >= 4:
            r = {'x': int(region[0]), 'y': int(region[1]), 'w': int(region[2]), 'h': int(region[3])}
        elif isinstance(region, dict):
            r = region
        else:
            print(f"⚠️ detect_indi_button: unknown region format: {type(region)}")
            return None

        if not r:
            print(f"⚠️ detect_indi_button: region '{region}' not found in SAVED_REGIONS")
            return None

        # crop_image expects a region dict and a frame; it returns a BGR image
        gray = crop_image(r, gray_frame)

        # adjust offsets so returned coordinates are relative to original frame/screen
        # crop_image converts coordinates to frame-relative internally
        # For window-relative regions, coordinates are already frame-relative after crop_image
        # For absolute regions, crop_image converts them to frame-relative
        # So we can use the region coordinates directly as frame-relative offsets
        if r.get('_coord') == 'rel':
            # Window-relative: coordinates are already frame-relative after crop_image processing
            draw_base_x = int(r.get('x', 0))
            draw_base_y = int(r.get('y', 0))
        elif r.get('_coord') == 'ref':
            # Reference coordinates: crop_image converts to frame-relative, but we need to calculate
            # the frame-relative position from the reference coordinates
            wi = get_window_info()
            if wi:
                win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                # Convert reference coords to frame-relative
                draw_base_x = int(r.get('x', 0) * win_w / float(REF_WIDTH))
                draw_base_y = int(r.get('y', 0) * win_h / float(REF_HEIGHT))
            else:
                draw_base_x = int(r.get('x', 0))
                draw_base_y = int(r.get('y', 0))
        else:
            # Absolute coordinates: crop_image converts to frame-relative by subtracting window position
            wi = get_window_info()
            if wi:
                win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                draw_base_x = int(r.get('x', 0)) - win_x
                draw_base_y = int(r.get('y', 0)) - win_y
            else:
                draw_base_x = int(r.get('x', 0))
                draw_base_y = int(r.get('y', 0))
        x_offset = x_offset + draw_base_x
        y_offset = y_offset + draw_base_y
    # If the working image is small, upscale it to a target reference to improve matching
    # We'll upscale to at least 1920x1080 while preserving aspect ratio
    target_w, target_h = 1920, 1080
    orig_h, orig_w = gray.shape[:2]
    scale = 1.0
    if orig_w < target_w or orig_h < target_h:
        try:
            scale_w = float(target_w) / float(orig_w)
            scale_h = float(target_h) / float(orig_h)
            # choose the larger scale so both dimensions reach the target
            scale = max(scale_w, scale_h)
            # compute scaled size (keep integer sizes)
            scaled_w = max(1, int(round(orig_w * scale)))
            scaled_h = max(1, int(round(orig_h * scale)))
            scaled_gray = cv2.resize(gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            print(f"⚠️ Upscale failed: {e}")
            scaled_gray = gray.copy()
    else:
        scaled_gray = gray.copy()

    img_h, img_w = scaled_gray.shape[:2]
    
    for s in scales:
        # Calculate new dimensions
        nw, nh = int(tw * s), int(th * s)
        
        # Validate dimensions
        if nw <= 0 or nh <= 0:
            print(f"⚠️ Skipping invalid dimensions: {nw}x{nh}")
            continue
            
        # Skip if too small or too large
        if nw < 8 or nh < 8:
            continue
            
        # Convert image to grayscale if needed

        if nw > img_w or nh > img_h:
            continue
            
        try:
            # Additional safety check before resize
            if nw > 0 and nh > 0:
                resized = cv2.resize(template, (nw, nh), interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR)
            else:
                print("⚠️ Invalid resize dimensions")
                continue
        except Exception as e:
            print(f"⚠️ Resize error: {e}")
            continue

        res = cv2.matchTemplate(scaled_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_size = (nw, nh)

    if best_val >= tm_threshold and best_loc is not None:
        bx, by = best_loc
        bw, bh = best_size
        cx, cy = bx + bw // 2, by + bh // 2
        sx_scaled, sy_scaled = x_offset + cx, y_offset + cy

        # Map scaled image coordinates back to original frame coordinates
        try:
            sx_frame = int(round(sx_scaled / scale))
            sy_frame = int(round(sy_scaled / scale))
        except Exception:
            sx_frame = int(round(sx_scaled))
            sy_frame = int(round(sy_scaled))

        # Convert frame-relative coordinates to absolute screen coordinates
        try:
            wi = get_window_info()
            if wi and 'x' in wi and 'y' in wi:
                sx = sx_frame + int(wi['x'])
                sy = sy_frame + int(wi['y'])
            else:
                sx = sx_frame
                sy = sy_frame
        except Exception as e:
            print(f"⚠️ Error getting window info for button click: {e}")
            sx = sx_frame
            sy = sy_frame

        info = {"filename": filename, "method": "template", "confidence": float(best_val), "x": int(sx), "y": int(sy)}
        print(f"✅ Button '{filename}' found (template) confidence={best_val:.2f} at scaled({cx},{cy}) -> frame({sx_frame},{sy_frame}) -> screen ({sx},{sy})")
        # perform click if requested
        if do_click:
            # apply global click offset (tweak these constants above if clicks land off)
            sx = sx + CLICK_OFFSET_X
            sy = sy + CLICK_OFFSET_Y
            
            # Validate coordinates before clicking to prevent failsafe triggers
            sx, sy, is_valid = validate_click_coordinates(sx, sy)
            if not is_valid:
                print(f"⚠️ Skipping click due to invalid coordinates")
                return info
            if USE_REAL_MOUSE:
                print(sx, sy)
                ok = send_real_click_restore(sx, sy)
                # if real mouse click failed and we have a hwnd, try window message
                if not ok and hwnd is not None:
                    ok = send_window_click(hwnd, sx, sy)
                if not ok:
                    try:
                        pydirectinput.click(sx, sy)
                    except Exception:
                        pyautogui.click(sx, sy)
            else:
                if hwnd is not None:
                    ok = send_window_click(hwnd, sx, sy)
                    if not ok:
                        try:
                            pydirectinput.click(sx, sy)
                        except Exception:
                            pyautogui.click(sx, sy)
                else:
                    try:
                        pydirectinput.click(sx, sy)
                    except Exception:
                        pyautogui.click(sx, sy)
            time.sleep(0.3)

        # debug: save annotated image
        if debug:
            # draw on full frame if available, otherwise draw on the working gray image
            if gray_frame is not None:
                out = gray_frame.copy()
                # translate scaled detection box back to original coordinates for drawing
                draw_bx_scaled = bx + draw_base_x
                draw_by_scaled = by + draw_base_y
                draw_bx = int(round(draw_bx_scaled / scale))
                draw_by = int(round(draw_by_scaled / scale))
                bw_un = max(1, int(round(bw / scale)))
                bh_un = max(1, int(round(bh / scale)))
            else:
                out = cv2.cvtColor(scaled_gray, cv2.COLOR_GRAY2BGR)
                draw_bx = bx
                draw_by = by
                bw_un = bw
                bh_un = bh

            cv2.rectangle(out, (int(draw_bx), int(draw_by)), (int(draw_bx) + bw_un - 1, int(draw_by) + bh_un - 1), (0, 255, 0), 2)
            cv2.putText(out, f"{filename} {best_val:.2f}", (max(0, int(draw_bx)), max(0, int(draw_by) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            save_path = debug_out or os.path.join(BUTTONS_FOLDER, "detection_debug.png")
            print(f"🖼 Debug image saved to {save_path}")

        return info

    try:
        orb = cv2.ORB_create(600)
        kp_img, des_img = orb.detectAndCompute(scaled_gray, None)
        if des_img is None or len(kp_img) < 8:
            return

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for filename in os.listdir(BUTTONS_FOLDER):
            if not filename.lower().endswith('.png'):
                continue
            tpl_path = os.path.join(BUTTONS_FOLDER, filename)
            tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue
            kp_tpl, des_tpl = orb.detectAndCompute(tpl, None)
            if des_tpl is None or len(kp_tpl) < 4:
                continue

            # knn match template->image
            matches = bf.knnMatch(des_tpl, des_img, k=2)
            good = []
            for m_n in matches:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) >= 8:
                # compute centroid of matched keypoints in the image
                pts = [kp_img[m.trainIdx].pt for m in good]
                avg_x = int(sum(p[0] for p in pts) / len(pts))
                avg_y = int(sum(p[1] for p in pts) / len(pts))
                sx_scaled, sy_scaled = x_offset + avg_x, y_offset + avg_y

                try:
                    sx_frame = int(round(sx_scaled / scale))
                    sy_frame = int(round(sy_scaled / scale))
                except Exception:
                    sx_frame = int(round(sx_scaled))
                    sy_frame = int(round(sy_scaled))

                try:
                    wi = get_window_info()
                    if wi and 'x' in wi and 'y' in wi:
                        sx = sx_frame + int(wi['x'])
                        sy = sy_frame + int(wi['y'])
                    else:
                        sx = sx_frame
                        sy = sy_frame
                except Exception as e:
                    print(f"⚠️ Error getting window info for ORB button click: {e}")
                    sx = sx_frame
                    sy = sy_frame

                info = {"filename": filename, "method": "orb", "matches": len(good), "x": int(sx), "y": int(sy)}
                print(f"✅ Button '{filename}' found (ORB) matches={len(good)} at scaled({avg_x},{avg_y}) -> frame({sx_frame},{sy_frame}) -> screen ({sx},{sy})")
                # perform click if requested
                if do_click:
                    # apply global click offset
                    sx = sx + CLICK_OFFSET_X
                    sy = sy + CLICK_OFFSET_Y
                    
                    # Validate coordinates before clicking to prevent failsafe triggers
                    sx, sy, is_valid = validate_click_coordinates(sx, sy)
                    if not is_valid:
                        print(f"⚠️ Skipping click due to invalid coordinates")
                        return info
                    if USE_REAL_MOUSE:
                        ok = send_real_click_restore(sx, sy)
                        if not ok and hwnd is not None:
                            ok = send_window_click(hwnd, sx, sy)
                        if not ok:
                            try:
                                pydirectinput.click(sx, sy)
                            except Exception:
                                pyautogui.click(sx, sy)
                    else:
                        if hwnd is not None:
                            ok = send_window_click(hwnd, sx, sy)
                            if not ok:
                                try:
                                    pydirectinput.click(sx, sy)
                                except Exception:
                                    pyautogui.click(sx, sy)
                        else:
                            try:
                                pydirectinput.click(sx, sy)
                            except Exception:
                                pyautogui.click(sx, sy)
                    time.sleep(0.3)

                # debug: save annotated image
                if debug:
                    if gray_frame is not None:
                        out = gray_frame.copy()
                        draw_x = int(round((avg_x + draw_base_x) / scale))
                        draw_y = int(round((avg_y + draw_base_y) / scale))
                    else:
                        out = cv2.cvtColor(scaled_gray, cv2.COLOR_GRAY2BGR)
                        draw_x = int(avg_x)
                        draw_y = int(avg_y)

                    cv2.circle(out, (draw_x, draw_y), 10, (255, 0, 0), 2)
                    cv2.putText(out, f"{filename} orb:{len(good)}", (max(0, draw_x-30), max(0, draw_y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    save_path = debug_out or os.path.join(BUTTONS_FOLDER, "detection_debug.png")
                    print(f"🖼 Debug image saved to {save_path}")

                return info
    except Exception as e:
        # ORB may fail on some installations; treat as optional fallback
        print("⚠️ ORB fallback failed:", e)
        return
    except Exception as e:
        # ORB may fail on some installations; treat as optional fallback
        print("⚠️ ORB fallback failed:", e)
        return
def detect_ui_buttons(gray_frame, file_name="", do_click=True, debug=False, debug_out=None, x_offset=0, y_offset=0, hwnd=None):
    """Detect in-game UI buttons by multi-scale template matching (optimized version).

    This function uses cached templates and optimized detection for faster performance.
    On first confident match the function clicks the found center and returns.
    """
    global BUTTON_TEMPLATES_CACHE
    
    # Ensure templates are loaded
    if not BUTTON_TEMPLATES_LOADED:
        load_button_templates()
    
    if not BUTTON_TEMPLATES_CACHE:
        print(f"⚠️ No button templates available in '{BUTTONS_FOLDER}'.")
        return None

    # Convert frame to grayscale once (optimization)
    if gray_frame is None:
        return None
    if len(gray_frame.shape) == 3:
        gray = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_frame

    # Reduced scales for better performance (10 scales instead of 21)
    # Focus on common scale range: 0.7 to 1.3 with higher density around 1.0
    scales = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    threshold = 0.7

    if file_name == "":
        # Check all buttons, return on first match (early exit optimization)
        for filename, template in BUTTON_TEMPLATES_CACHE.items():
            info = detect_indi_button_optimized(
                template, filename, gray, scales, 
                tm_threshold=threshold, region=None, do_click=do_click, 
                debug=debug, debug_out=debug_out, x_offset=x_offset, 
                y_offset=y_offset, hwnd=hwnd
            )
            if info is not None:
                return info
        return None
    else:
        # Check specific button
        if not file_name.lower().endswith(".png"):
            return None
        template = BUTTON_TEMPLATES_CACHE.get(file_name)
        if template is None:
            # Fallback: try loading from disk if not in cache
            tpl_path = os.path.join(BUTTONS_FOLDER, file_name)
            info = detect_indi_button(tpl_path, gray_frame, scales, tm_threshold=threshold, region=None, do_click=do_click, debug=debug, debug_out=debug_out, x_offset=x_offset, y_offset=y_offset, hwnd=hwnd)
            return info
        info = detect_indi_button_optimized(
            template, file_name, gray, scales,
            tm_threshold=threshold, region=None, do_click=do_click,
            debug=debug, debug_out=debug_out, x_offset=x_offset,
            y_offset=y_offset, hwnd=hwnd
        )
        return info
def get_blood_percentage(image: np.ndarray) -> float:
    """
    Estimate the red (blood) region percentage in an image with high accuracy.
    Works well for progress/health bars even if numbers or UI elements overlay it.

    Args:
        image (np.ndarray): RGB or BGR image array (H x W x 3).

    Returns:
        float: Percentage of red (blood) pixels in the total visible area.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid NumPy image array")

    # Ensure BGR -> RGB if needed
    if image.shape[2] == 3:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image.copy()

    # Convert to multiple color spaces
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    # --- HSV red range (two segments because hue wraps around 180) ---
    mask1 = cv2.inRange(hsv, (0, 80, 50), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
    hsv_mask = cv2.bitwise_or(mask1, mask2)

    # --- LAB a-channel (detect red component intensity) ---
    a_channel = lab[:, :, 1]
    lab_mask = cv2.inRange(a_channel, 150, 255)

    # Combine both masks for high precision
    combined_mask = cv2.bitwise_or(hsv_mask, lab_mask)

    # Morphological filtering for smoothness and removing noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Compute visible (non-black / non-transparent) mask
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    visible_mask = gray > 5

    # Count pixels
    total_visible = np.count_nonzero(visible_mask)
    red_pixels = np.count_nonzero(combined_mask & visible_mask)

    if total_visible == 0:
        return 0.0

    percentage = (red_pixels / total_visible) * 100.0
    return float(percentage)


def extract_number_from_region(region_name, frame, window_info=None):
    """Crop saved region `region_name` from `frame` and OCR an integer number.

    Returns an int if a number was detected, otherwise None.

    - region_name: key in SAVED_REGIONS (e.g. 'check_bottle_amount', 'blood_buy_part_region')
    - frame: the captured frame (BGR numpy array)
    - window_info: optional window rect used for mapping reference coords
    """
    try:
        if region_name not in SAVED_REGIONS:
            print(f"⚠️ Region '{region_name}' not found in SAVED_REGIONS")
            return None
        r = SAVED_REGIONS[region_name]

        # Map reference coords to actual frame/window coords if needed
        try:
            if r.get('_coord') == 'ref':
                wi = window_info or get_window_info()
                if wi and 'width' in wi and 'height' in wi:
                    win_x, win_y = int(wi.get('x', 0)), int(wi.get('y', 0))
                    win_w, win_h = int(wi.get('width', 1)), int(wi.get('height', 1))
                    x = win_x + int(r.get('x', 0) * win_w / float(REF_WIDTH))
                    y = win_y + int(r.get('y', 0) * win_h / float(REF_HEIGHT))
                    w = int(r.get('w', 0) * win_w / float(REF_WIDTH))
                    h = int(r.get('h', 0) * win_h / float(REF_HEIGHT))
                else:
                    # fallback: map against full screen
                    user32 = ctypes.windll.user32
                    sw = user32.GetSystemMetrics(0)
                    sh = user32.GetSystemMetrics(1)
                    x = int(r.get('x', 0) * sw / float(REF_WIDTH))
                    y = int(r.get('y', 0) * sh / float(REF_HEIGHT))
                    w = int(r.get('w', 0) * sw / float(REF_WIDTH))
                    h = int(r.get('h', 0) * sh / float(REF_HEIGHT))
            else:
                x = int(r.get('x', 0))
                y = int(r.get('y', 0))
                w = int(r.get('w', 0))
                h = int(r.get('h', 0))
        except Exception:
            x = int(r.get('x', 0))
            y = int(r.get('y', 0))
            w = int(r.get('w', 0))
            h = int(r.get('h', 0))

        # clamp to frame
        fh, fw = frame.shape[:2]
        x0 = max(0, min(fw - 1, x))
        y0 = max(0, min(fh - 1, y))
        x1 = max(0, min(fw, x0 + max(1, w)))
        y1 = max(0, min(fh, y0 + max(1, h)))
        if x1 <= x0 or y1 <= y0:
            print(f"⚠️ Invalid crop for region '{region_name}': ({x0},{y0})-({x1},{y1})")
            return None

        crop = frame[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            print(f"⚠️ Empty crop for region '{region_name}'")
            return None

        # Preprocess for OCR: grayscale, upscale, denoise, adaptive threshold
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # upscale small regions to help OCR
        scale = 1
        if gray.shape[0] < 30 or gray.shape[1] < 60:
            scale = 3
        elif gray.shape[0] < 60 or gray.shape[1] < 120:
            scale = 2
        if scale != 1:
            gray = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # try adaptive thresholding for variable UI backgrounds
        try:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        except Exception:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

        # OCR with digits-only whitelist
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        try:
            text = pytesseract.image_to_string(th, config=config)
        except Exception as e:
            print(f"⚠️ pytesseract failed for region '{region_name}': {e}")
            return None

        # extract digits
        m = re.search(r"(\d+)", text)
        if not m:
            # debug output when OCR didn't find digits
            print(f"🔍 OCR result for '{region_name}': '{text.strip()}' (no digits)")
            return None
        digits = m.group(1)
        try:
            val = int(digits)
            print(f"🔢 Detected number in '{region_name}': {val} (raw='{text.strip()}')")
            return val
        except Exception:
            print(f"⚠️ Failed to parse digits '{digits}' for region '{region_name}'")
            return None
    except Exception as e:
        print(f"⚠️ extract_number_from_region error: {e}")
        return None

def extract_number_from_image(img_bgr, debug_save=False, region_label='img_crop', allow_decimals=True):
    """Extract a number from a BGR crop (numpy array).
    
    Returns a float or int if found, otherwise None. Saves debug images to
    `ocr_debug/` when debug_save=True.
    
    Args:
        img_bgr: BGR image array
        debug_save: Save debug preprocessing images
        region_label: Label for debugging
        allow_decimals: If True, matches decimal numbers (e.g., 37.794). If False, matches integers only.
    """

    print("Starting OCR processing...",img_bgr.shape)

    try:
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            if debug_save:
                print(f"[OCR] provided image is None or empty")
            return None
            
        # Check if image is valid and has proper dimensions
        if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
            if debug_save:
                print(f"[OCR] invalid image format: shape={img_bgr.shape}")
            return None

        gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        hsv_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_orig, lower_yellow, upper_yellow)
        
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv_orig, lower_green, upper_green)
        
        mask_colored = cv2.bitwise_or(mask_yellow, mask_green)
        # upscale small crops to improve OCR
        h, w = gray_orig.shape[:2]
        scale = 1
        aspect_ratio_val = max(h, w) / max(1, min(h, w))
        if min(h, w) < 20 or (aspect_ratio_val > 5 and max(h, w) < 200):
            scale = 4
        elif max(h, w) < 120:
            scale = 3
        elif max(h, w) < 200:
            scale = 2
        if scale != 1:
            gray_orig = cv2.resize(gray_orig, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            h, w = gray_orig.shape[:2]
        variants = []
        is_tiny = max(h, w) < 50
        is_thin = min(h, w) < 15
        aspect_ratio = max(h, w) / max(1, min(h, w))
        
        blur_kernel = (2, 2) if is_thin else ((2, 2) if is_tiny else (3, 3))
        b = cv2.GaussianBlur(gray_orig, blur_kernel, 0)
        
        _, th_otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((th_otsu, 'otsu'))
        variants.append((cv2.bitwise_not(th_otsu), 'otsu_inv'))
        
        adaptive_size = 5 if is_thin else (9 if is_tiny else 15)
        adaptive_const = 2 if is_thin else (3 if is_tiny else 8)
        th_adapt = cv2.adaptiveThreshold(gray_orig, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, adaptive_size, adaptive_const)
        variants.append((th_adapt, 'adaptive_mean'))
        variants.append((cv2.bitwise_not(th_adapt), 'adaptive_mean_inv'))
        
        gauss_size = 5 if is_thin else (9 if is_tiny else 11)
        th_adapt_gauss = cv2.adaptiveThreshold(gray_orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, gauss_size, 2)
        variants.append((th_adapt_gauss, 'adaptive_gauss'))
        variants.append((cv2.bitwise_not(th_adapt_gauss), 'adaptive_gauss_inv'))
        
        high_threshold = 150 if is_thin else 130
        _, th_high = cv2.threshold(gray_orig, high_threshold, 255, cv2.THRESH_BINARY)
        variants.append((th_high, 'high_threshold'))
        
        low_threshold = 100 if is_thin else 80
        _, th_low = cv2.threshold(gray_orig, low_threshold, 255, cv2.THRESH_BINARY_INV)
        variants.append((th_low, 'low_threshold_inv'))
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray_orig)
        _, th_clahe = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append((th_clahe, 'clahe_otsu'))
        variants.append((cv2.bitwise_not(th_clahe), 'clahe_otsu_inv'))
        
        _, th_colored = cv2.threshold(mask_colored, 127, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(th_colored) > 0:
            variants.append((th_colored, 'color_yellow_green'))
            variants.append((cv2.bitwise_not(th_colored), 'color_yellow_green_inv'))
        
        very_high_threshold = 200 if is_thin else 180
        _, th_very_high = cv2.threshold(gray_orig, very_high_threshold, 255, cv2.THRESH_BINARY)
        if np.count_nonzero(th_very_high) > 0:
            variants.append((th_very_high, 'very_high_threshold'))
        
        very_low_threshold = 50 if is_thin else 30
        _, th_very_low = cv2.threshold(gray_orig, very_low_threshold, 255, cv2.THRESH_BINARY_INV)
        if np.count_nonzero(th_very_low) > 0:
            variants.append((th_very_low, 'very_low_threshold_inv'))
        
        kernel_size = (1, 1) if is_thin else ((1, 1) if is_tiny else (2, 2))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        ts = int(time.time())
        
        for idx, (proc_img, vname) in enumerate(variants):
            try:
                proc = cv2.medianBlur(proc_img, 3)
                
                if not is_thin:
                    if not is_tiny:
                        proc = cv2.morphologyEx(proc, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size))
                
                proc = cv2.dilate(proc, dilate_kernel, iterations=1)
                
                if "high" in vname or "very_high" in vname or "very_low" in vname or "low" in vname:
                    proc = cv2.dilate(proc, dilate_kernel, iterations=2)

                # if debug_save:
                #     try:
                #         os.makedirs('ocr_debug', exist_ok=True)
                #         fname = os.path.join('ocr_debug', f'{region_label}_{vname}_{ts}.png')
                #     except Exception:
                #         pass
                
                whitelist = "0123456789/"
                if allow_decimals:
                    whitelist += "."
                config = f"--psm 7 --oem 3 -c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1 -c tessedit_pageseg_mode=7"
                text = pytesseract.image_to_string(proc, config=config)
                clean_text = text.strip().replace(" ", "").replace("|", "/").replace("—", "").replace("_", "")
                clean_text = clean_text.replace("l", "1").replace("I", "1").replace("O", "0").replace("o", "0")
                if not allow_decimals:
                    clean_text = clean_text.replace("S", "5")
                
                if clean_text and len(clean_text) > 0:
                    print(f"🔍 OCR [{vname}] for '{region_label}': '{clean_text}'")

                    m = re.search(r"(\d+)\s*/\s*(\d+)", clean_text)
                    if m:
                        current_val = int(m.group(1))
                        max_val = int(m.group(2))
                        if max_val > 0:
                            result = (current_val*100/max_val)
                            print(f"🔢 Detected '{region_label}': {current_val}/{max_val} = {result:.1f}%")
                            return result
                    
                    if allow_decimals:
                        m = re.search(r"(\d+\.?\d*)", clean_text)
                        if m:
                            try:
                                val = float(m.group(1))
                                if val > 0 or val < 0:
                                    print(f"🔢 Detected '{region_label}': {val}")
                                    return val
                            except ValueError:
                                pass
                    
                    m = re.search(r"(\d{2,}|\d+\.)", clean_text)
                    if m:
                        try:
                            val = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
                            print(f"🔢 Detected '{region_label}': {val}")
                            return val
                        except (ValueError, AttributeError):
                            pass
                    
                    m = re.search(r"(\d+)", clean_text)
                    if m:
                        val_str = m.group(1)
                        if len(val_str) > 1 or (len(val_str) == 1 and val_str != "0"):
                            try:
                                val = int(val_str)
                                print(f"🔢 Detected '{region_label}': {val}")
                                return val
                            except ValueError:
                                pass
                        
            except Exception as e:
                continue
        
        print(f"⚠️ No valid number found in '{region_label}' (tried all {len(variants)} variants, aspect={aspect_ratio:.1f}:1)")
        return None

    except Exception as e:
        if debug_save:
            print(f"[OCR] exception for {region_label}: {e}")
        return None

def calculate_three_images_similarity(img1, img2, img3):
    """Calculate overall similarity of three images using SSIM (Structural Similarity Index).
    
    This function computes pairwise similarities between all three images and returns
    a single value representing the total similarity of all three images together.
    
    Parameters:
    - img1, img2, img3: numpy arrays (BGR images) or None
    
    Returns:
    - float: average similarity of all three pairwise comparisons (0-1)
      - Returns 0.0 if any input image is None or invalid
    
    Similarity scores:
    - 1.0 = all three images are identical
    - 0.0 = images are completely different
    - Typically > 0.9 for very similar images
    - The returned value is the average of (img1-img2, img2-img3, img1-img3) similarities
    """
    # Input validation
    if img1 is None or img2 is None or img3 is None:
        print("⚠️ One or more images are None")
        return 0.0
    
    try:
        # Convert to numpy arrays if needed and validate
        images = [img1, img2, img3]
        for i, img in enumerate(images):
            if not isinstance(img, np.ndarray):
                print(f"⚠️ Image {i+1} is not a numpy array")
                return 0.0
            if img.size == 0:
                print(f"⚠️ Image {i+1} is empty")
                return 0.0
        
        # Helper function to normalize images to same size and convert to grayscale
        def normalize_image(img):
            """Convert to grayscale and return as-is (assumes same size or caller handles resizing)."""
            if len(img.shape) == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif len(img.shape) == 2:
                return img
            else:
                raise ValueError(f"Unexpected image shape: {img.shape}")
        
        # Get dimensions and determine target size (use minimum dimensions to avoid upscaling)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h3, w3 = img3.shape[:2]
        
        target_h = min(h1, h2, h3)
        target_w = min(w1, w2, w3)
        
        # Resize all images to target size if needed
        def resize_if_needed(img, target_h, target_w):
            h, w = img.shape[:2]
            if h != target_h or w != target_w:
                return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return img
        
        # Normalize all images
        gray1 = normalize_image(resize_if_needed(img1, target_h, target_w))
        gray2 = normalize_image(resize_if_needed(img2, target_h, target_w))
        gray3 = normalize_image(resize_if_needed(img3, target_h, target_w))
        
        # Calculate SSIM for each pair
        # SSIM window size should be odd and smaller than image dimensions
        win_size = min(7, min(target_w, target_h))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3
        
        try:
            # Calculate pairwise similarities
            sim_12, _ = ssim(gray1, gray2, win_size=win_size, full=True)
            sim_23, _ = ssim(gray2, gray3, win_size=win_size, full=True)
            sim_13, _ = ssim(gray1, gray3, win_size=win_size, full=True)
            
            # Convert to float to ensure proper numeric types
            sim_12 = float(sim_12)
            sim_23 = float(sim_23)
            sim_13 = float(sim_13)
            
            # Calculate overall similarity as average of all three pairwise comparisons
            similarities = [sim_12, sim_23, sim_13]
            overall_similarity = sum(similarities) / len(similarities)
            
            return float(overall_similarity)
            
        except Exception as e:
            print(f"⚠️ Error calculating SSIM: {e}")
            return 0.0
            
    except Exception as e:
        print(f"⚠️ Error in calculate_three_images_similarity: {e}")
        return 0.0


def safe_resize(img, target_size=None, fx=None, fy=None, interpolation=cv2.INTER_LINEAR):
    """Resize an image safely. Accepts either target_size=(w,h) or fx/fy scale factors.

    Guarantees integer positive sizes; returns None on invalid input.
    """
    try:
        if img is None:
            return None
        h, w = img.shape[:2]
        if target_size is not None:
            tw, th = target_size
            tw = max(1, int(tw))
            th = max(1, int(th))
            # If sizes equal, return original
            if tw == w and th == h:
                return img
            return cv2.resize(img, (tw, th), interpolation=interpolation)

        # fallback to fx/fy
        if fx is None and fy is None:
            return img
        if fx is None:
            fx = fy
        if fy is None:
            fy = fx
        nw = max(1, int(round(w * float(fx))))
        nh = max(1, int(round(h * float(fy))))
        if nw == w and nh == h:
            return img
        return cv2.resize(img, (nw, nh), interpolation=interpolation)
    except Exception as e:
        print(f"⚠️ safe_resize failed: {e}")
        return None
    
# --- Movement logging helpers ---
MOVEMENT_LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
MOVEMENT_LOG_CSV = os.path.join(MOVEMENT_LOG_DIR, 'movement_log.csv')
MOVEMENT_LOG_JSONL = os.path.join(MOVEMENT_LOG_DIR, 'movement_log.jsonl')

def _init_movement_log():
    try:
        os.makedirs(MOVEMENT_LOG_DIR, exist_ok=True)
        if not os.path.exists(MOVEMENT_LOG_CSV):
            with open(MOVEMENT_LOG_CSV, 'w', encoding='utf-8') as f:
                f.write('timestamp,score,frac,ema,streak,stable,moving\n')
    except Exception as e:
        print(f"⚠️ Could not initialize movement log: {e}")

def append_movement_log_entry(score, frac, ema, streak, stable, moving):
    """Append a movement entry to CSV and JSONL logs."""
    try:
        _init_movement_log()
        ts = datetime.now().isoformat()
        # CSV: simple numeric fields; moving stored as 0/1
        try:
            with open(MOVEMENT_LOG_CSV, 'a', encoding='utf-8') as f:
                f.write(f"{ts},{int(score)},{frac:.6f},{ema:.6f},{int(streak)},{int(stable)},{int(bool(moving))}\n")
        except Exception as e:
            print(f"⚠️ Failed to write CSV movement log: {e}")

        # JSONL: full structured entry
        entry = {
            'timestamp': ts,
            'score': int(score) if score is not None else None,
            'frac': float(frac) if frac is not None else None,
            'ema': float(ema) if ema is not None else None,
            'streak': int(streak),
            'stable': int(stable),
            'moving': bool(moving)
        }
        try:
            with open(MOVEMENT_LOG_JSONL, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Failed to write JSONL movement log: {e}")
    except Exception as e:
        print(f"⚠️ append_movement_log_entry failed: {e}")



def handle_mission_area(quest_img):
    print("handle_mission_area_testing_score")
    if quest_img is None:
        return None

    # Access global state
    global QUEST_HISTORY
    # ensure history and state exist
    try:
        QUEST_HISTORY
    except NameError:
        QUEST_HISTORY = deque(maxlen=8)
    # push current frame into history (make a safe copy)
    try:
        QUEST_HISTORY.append(quest_img.copy())
    except Exception:
        QUEST_HISTORY.append(quest_img)

    # need at least 3 frames
    if len(QUEST_HISTORY) < 3:
        print("character_analysis: need 3 frames (have", len(QUEST_HISTORY), ")")
        return 1.0

    f1, f2, f3 = list(QUEST_HISTORY)[-3:]
    similarity_score = calculate_three_images_similarity(f1, f2, f3)
    return similarity_score

# ========== MAIN LOOP ==========
def position_console_window():
    """Position the console window to not overlap with the game."""
    try:
        # Get console window handle
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if not console_hwnd:
            return

        # Get primary monitor resolution
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)  # SM_CXSCREEN
        screen_height = user32.GetSystemMetrics(1)  # SM_CYSCREEN

        # Position console in bottom-right corner
        console_width = 800
        console_height = 600
        console_x = screen_width - console_width  # Right side
        console_y = screen_height - console_height  # Bottom

        # Move console window
        win32gui.MoveWindow(console_hwnd, console_x, console_y, 
                           console_width, console_height, True)
        
        # Ensure console stays below other windows
        win32gui.SetWindowPos(console_hwnd, win32con.HWND_NOTOPMOST, 
                             console_x, console_y, console_width, console_height,
                             win32con.SWP_NOACTIVATE)

    except Exception as e:
        print("⚠️ Could not position console window:", e)

def go_to_village(frame, window_info=None):
    """Navigate to village. window_info is for multi-instance support."""
    print("==================go_to_village==================")
    click_saved_region_center("guide_button_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("guide_tab_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    if(info := search_region_in_frame("auto_trade_region", frame)):
        send_real_mouse_move(info["x"], info["y"])
    else:
        click_saved_region_center("out_button_region", method="auto", window_info=window_info)
        time.sleep(0.5)
        go_to_village_using_map(frame, window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("teleport_village_button_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("purchase_confirm_at_vil_region", method='auto', window_info=window_info)
    time.sleep(0.5)

def go_to_village_using_map(frame, window_info=None):
    """Navigate to village using map. window_info is for multi-instance support."""
    print("==================go_to_village_using_map==================")
    time.sleep(0.5)
    click_saved_region_center("map_region", method='auto', window_info=window_info)
    time.sleep(1)
    drag_mouse(SAVED_REGIONS["village_list_region"])
    time.sleep(0.5)
    drag_mouse(SAVED_REGIONS["village_list_region"])
    time.sleep(0.5)
    click_saved_region_center("gludin_village_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("teleport_village_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("purchase_confirm_at_vil_region", method='auto', window_info=window_info)
    time.sleep(0.5)


def buy_hp_bottle(frame, window_info=None):
    """Buy HP bottles. window_info is for multi-instance support."""
    print("==================buy_hp_bottle==================")
    time.sleep(15.0)
    click_saved_region_center("sales_address_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("general_mer_region", method='auto', window_info=window_info)
    time.sleep(10)
    
    click_saved_region_center("blood_buy_part_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("max_blood_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("blood_buy_confirm_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("purchase_confirm_at_vil_region", method='auto', window_info=window_info)
    time.sleep(0.5)
    click_saved_region_center("out_button_region", method='auto', window_info=window_info)
    time.sleep(1)
    click_saved_region_center("sales_address_region", method='auto', window_info=window_info)
    time.sleep(0.5)

def go_to_other_states(state_name, purpose_name="", frame=None, window_info=None):
    """Navigate to different game states.
    
    window_info: Optional window info dict for multi-instance support.
                 If provided, clicks will target this specific window.
    """
    if state_name == "hp_charge_state":
        if(purpose_name == "to_charge"):
            buy_hp_bottle(frame, window_info=window_info)
        elif(purpose_name == "to_village"):
            go_to_village(frame, window_info=window_info)
            buy_hp_bottle(frame, window_info=window_info)
        elif(purpose_name == "to_village_using_map"):
            go_to_village_using_map(frame, window_info=window_info)
            buy_hp_bottle(frame, window_info=window_info)
    elif(state_name == "jump_button_region"):
        click_saved_region_center('jump_button_region', method='auto', window_info=window_info)
        time.sleep(0.3)
        click_saved_region_center("teleport_confirm_region", method='auto', window_info=window_info)
    return False


def main():

    
    game_state = {
        "movement_flag": False,
        "teleport_ability": False,
        "current_hp_state": 100,
        "jump_village_count": 0,
        "blood_bottle_count": 0
    }




    game_start_flag  = False
    score = 1.0
    movement_flag_start_time = None
    movement_flag_end_time = None
    # Position console window before starting
    position_console_window()
    print("🚀 Starting smart bot... (Ctrl+C to stop)")
    
    try:
        while not STOP_EVENT:
            window_info = get_window_info()
            if not window_info:
                print("❌ Game window not found.")
                time.sleep(2)
                continue

            # Keep window in foreground
            if 'hwnd' in window_info:
                force_foreground(window_info['hwnd'])

            x1, y1 = window_info['x'], window_info['y']
            x2, y2 = x1 + window_info['width'], y1 + window_info['height']
            frame = capture_region(x1, y1, x2, y2)
            ("captured_frame.png", frame) # doesn't capture self size window.
            # Use safe crop helper instead of slicing the frame directly
            if frame is None:
                print("⚠️ capture_region returned None — skipping this loop iteration")
                continue          




            if(game_start_flag is False):
                if(search_region_in_frame('start_sign_region', frame)):
                    game_start_flag = True
                    temp_region = SAVED_REGIONS.get("start_sign_region", {})
                    if temp_region:
                        drag_mouse(temp_region)
                    print("start a game")
                    time.sleep(5.0)
                else:
                    pass


            try:
                # Detect character movement
                # character_area_image = crop_image(SAVED_REGIONS.get("character_region"), frame)
                # game_state["movement_flag"] = handle_character_area(frame,i)
                # print movement status
                
                #getting jump_village_count
                jump_village_count = crop_image(SAVED_REGIONS.get('jump_village_region'), frame, offset=[0,10,0,10])
                if jump_village_count is not None and jump_village_count.size > 0:
                    bottle_count = extract_number_from_image(jump_village_count, region_label="jump_village_region")
                    if bottle_count is not None and isinstance(bottle_count, (int, float)):
                        game_state["jump_village_count"] = bottle_count
                    else:
                        game_state["jump_village_count"] = 0
                else:
                    print("⚠️ Invalid jump village region or image")
                    game_state["jump_village_count"] = game_state.get("jump_village_count", 100)
                #end getting jump_village_count


                #getting blood_bottle_count
                img_crop_bottle = crop_image(SAVED_REGIONS.get('blood_bottle_region'), frame, offset=[0,10,0,10])
                if img_crop_bottle is not None and img_crop_bottle.size > 0:
                
                    blood_bottle_count = extract_number_from_image(img_crop_bottle, debug_save=True, region_label="blood_bottle_region", allow_decimals=False)
                    if blood_bottle_count is not None and isinstance(blood_bottle_count, (int, float)):
                        game_state["blood_bottle_count"] = blood_bottle_count
                    else:
                        game_state["blood_bottle_count"] = 0
                else:
                    print("⚠️ Invalid blood bottle region or image")
                    game_state["blood_bottle_count"] = game_state.get("blood_bottle_count", 100)
                game_state["current_hp_state"] = game_state.get("current_hp_state", 100)
                img_hp_crop = crop_image(SAVED_REGIONS.get('hp_bar_main_region'), frame)
                # Validate crop before attempting to show it. cv2.imshow requires a window name and a valid image.




                if img_hp_crop is not None and img_hp_crop.size > 0:
                    try:
                        current_hp_state = get_blood_percentage(img_hp_crop)
                        print(f"Current HP State: {current_hp_state}")
                        
                        # Validate HP state before using
                        if current_hp_state is not None and isinstance(current_hp_state, (int, float)):
                            if 0 < current_hp_state < 100:  # Validate HP is in valid range
                                game_state["current_hp_state"] = current_hp_state
                                print(f"⚠️ Invalid HP value: {current_hp_state}")
                        else:
                            print("⚠️ Invalid HP state type or None value")
                    except Exception as e:
                        print(f"⚠️ Error processing HP: {str(e)}")
                else:
                    print("⚠️ Invalid HP region or image")
                

                #Analyze the  quest

                #end getting estimating  to charge hp
                # if (search_region_in_frame("teleport_confirm_region", frame) and "hp_charge_state" not in game_state):
                #         game_state["movement_flag"] = True

                first_quest = crop_image(SAVED_REGIONS['first_quest_region'], frame)
                score = handle_mission_area(first_quest)
                print("Mission  score test:",score)

                current_movement_flag = game_state.get("movement_flag", False)
                if current_movement_flag:
                    # Movement flag is True - check if it just became True
                    movement_flag_end_time = None
                    if movement_flag_start_time is None:
                        movement_flag_start_time = time.time()
                        print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")
                    
                    # Check if movement_flag has been True for 5 minutes
                    elapsed_time = time.time() - movement_flag_start_time
                    if elapsed_time >= MOVEMENT_FLAG_TIMEOUT:
                        go_to_other_states("jump_button_region", frame)
                        movement_flag_start_time = None
                else:
                    # Movement flag is False - reset timer
                    movement_flag_start_time = None
                    if movement_flag_end_time is None:
                        movement_flag_end_time = time.time()
                        print(f"⏱️ Movement flag became False at {time.strftime('%H:%M:%S')}")
                    # Check if movement_flag has been True for 5 minutes
                    elapsed_time = time.time() - movement_flag_end_time
                    if elapsed_time >= MOVEMENT_FLAG_TIMEOUT:
                        go_to_other_states("jump_button_region", frame)
                        game_state.get["movement_flag"] = True
                        movement_flag_end_time = None

            except Exception as e:
                print(f"⚠️ Error Getting Game Status: {e}")


            try:
                # Initialize default HP state
                
                # Update game state and check if healing is needed
                updated_state = screen_state(frame, game_state)
                if updated_state is not None:
                    game_state.update(updated_state)
                    
            except Exception as e:
                print(f"⚠️ Error checking HP state: {e}")
            if ("region_name" in game_state):
                region_name = game_state["region_name"]
                print("region_name in screen state", region_name)
                # send_real_click_restore(result['screen_x'], result['screen_y'])
                if region_name == "sleep_show_region":
                    temp_region = SAVED_REGIONS.get(game_state["region_name"], {})
                    drag_mouse(temp_region)

                else:
                    print("Found a region")
                    click_saved_region_center(game_state["region_name"], method='auto')
                    time.sleep(1.0)
                del  game_state["region_name"]


            
            elif("button" in game_state):
                print("Found a button:",game_state["button"])
                if (game_state["button"] == "skip_button_region.png" or game_state["button"] == "enter_button_region.png"):
                    game_state["movement_flag"] = False
                    movement_flag_start_time = None  # Reset timer when flag becomes False

                # if (info["filename"] == "teleport_confirm_region.png"):
                #     game_state["movement_flag"] = True
                del game_state["button"]
            # Track when movement_flag becomes True
            if(game_state["current_hp_state"] < 70 and game_state["movement_flag"] == False):
                continue
            print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
            # charge hp logic
            current_hp = game_state["current_hp_state"]
            if isinstance(current_hp, (int, float)) and current_hp < HEAL_THRESHOLD:
                if (game_state.get("jump_village_count", 0) > 0):
                    click_saved_region_center('jump_village_region', method='auto')
                    game_state["movement_flag"] = go_to_other_states("hp_charge_state","to_charge", frame)
                    # Reset timer if movement_flag changed
                elif("teleport_ability" in game_state and game_state["teleport_ability"] is True and "hp_charge_state" not  in game_state):
                    game_state["movement_flag"] = go_to_other_states("hp_charge_state", "to_village", frame)
                    # Reset timer if movement_flag changed
                else:
                    game_state["movement_flag"] = go_to_other_states("hp_charge_state", "to_village_using_map", frame)
                    print("No teleport ability available and No jump_village_count.")


            # Safe score comparison with proper validationmovement_count
            elif isinstance(score, (int, float)) and score < 0.3 and game_state.get("movement_flag") is True:
                if(game_state["current_hp_state"] > HEAL_THRESHOLD):
                    print("score:",score,"movement_flag:",game_state.get("movement_flag")," Clicking jump button")
                    go_to_other_states("jump_button_region")
                    game_state["movement_flag"] = True
                    # Track when movement_flag becomes True
                    if movement_flag_start_time is None:
                        movement_flag_start_time = time.time()
                        print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")

            elif(game_state.get("movement_flag") is False):
                go_to_other_states("jump_button_region")
                game_state["movement_flag"] = True
                # Track when movement_flag becomes True
                if movement_flag_start_time is None:
                    movement_flag_start_time = time.time()
                    print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")


            # Show daily notification when bot starts


            try:


                if should_notify_today():
                    time.sleep(2.0)
                    # show_daily_notification(ctypes.windll.kernel32.GetConsoleWindow())
                    time.sleep(4.0)
                    click_saved_region_center("nav_button_region", method='auto')
                    time.sleep(1.0)
                    click_saved_region_center("daily_button_region", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("daily_claim_benefit_region", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("daily_bonus_confirm", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("out_button_region", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("nav_button_region", method='auto')
                    time.sleep(1.0)
                    click_saved_region_center("mail_button_region", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("claim_all_button_region", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("daily_bonus_confirm", method='auto')
                    time.sleep(2.0)
                    click_saved_region_center("out_button_region", method='auto')
            except Exception as e:
                print(f"Failed to show daily notification: {e}")

            print("===================================================================================================")
            print(game_state)
            print("MOTION_RATIO", MOTION_RATIO)
            print(f"⏳ Waiting {CAPTURE_INTERVAL}s...\n")
            time.sleep(CAPTURE_INTERVAL)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    
    


if __name__ == "__main__":
    main()