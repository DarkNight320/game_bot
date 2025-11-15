"""
Multi-Window Game Controller for Lineage2M Bot

This module manages multiple game windows simultaneously, cycling through them
sequentially with a 5-second interval between each window control cycle.
Each window maintains its own independent game state.
"""

import win32gui
import win32api
import win32con
import time
import ctypes
from stop_event import STOP_EVENT
from window_utils import force_foreground

# Import all necessary functions and variables from final_4
import final_4

# Re-export configuration from final_4
WINDOW_SUBSTRING = final_4.WINDOW_SUBSTRING
HEAL_THRESHOLD = final_4.HEAL_THRESHOLD
CAPTURE_INTERVAL = final_4.CAPTURE_INTERVAL
MOTION_RATIO = final_4.MOTION_RATIO
SAVED_REGIONS = final_4.SAVED_REGIONS

# Interval between window switches (1 second as specified)
WINDOW_SWITCH_INTERVAL = 5.0


def get_all_game_windows():
    """Find all Lineage2M game windows currently running.
    
    Returns:
        list: List of window info dictionaries, each containing:
            - hwnd: Window handle
            - x, y: Window position
            - width, height: Window dimensions
            - name: Window title
            - is_fullscreen: Boolean indicating if window is fullscreen
    """
    candidates = []
    
    # Get screen dimensions
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
            
            # Validate and clamp coordinates
            x = max(0, min(rect[0], screen_width - 1))
            y = max(0, min(rect[1], screen_height - 1))
            width = max(1, min(rect[2] - rect[0], screen_width - x))
            height = max(1, min(rect[3] - rect[1], screen_height - y))
            
            # Check if window is fullscreen
            is_fullscreen = (width >= screen_width * 0.95 and 
                           height >= screen_height * 0.95)
            
            # Calculate priority: prefer game windows over bot/controller windows
            priority = 0
            title_lower = title.lower()
            
            # Prefer windows that are NOT bot/controller windows
            if "bot" not in title_lower and "controller" not in title_lower:
                priority += 10
            
            # Prefer larger windows
            priority += min(5, width // 200) + min(5, height // 200)
            
            # Prefer non-fullscreen windows
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
            
        except Exception as e:
            print(f"⚠️ Error getting window info for {title}: {e}")
            
    win32gui.EnumWindows(callback, None)
    
    if not candidates:
        print("⚠️ No game windows found")
        return []
    
    # Sort by priority (highest first) and remove priority from result
    candidates.sort(key=lambda x: x.get("priority", 0), reverse=True)
    
    result = []
    for candidate in candidates:
        info = {
            "x": max(0, candidate["x"]),
            "y": max(0, candidate["y"]),
            "width": max(1, candidate["width"]),
            "height": max(1, candidate["height"]),
            "name": candidate["name"],
            "hwnd": candidate["hwnd"],
            "is_fullscreen": candidate["is_fullscreen"]
        }
        result.append(info)
    
    print(f"✅ Found {len(result)} game window(s)")
    for i, win in enumerate(result, 1):
        print(f"   Window {i}: {win['name']} at ({win['x']}, {win['y']}) size {win['width']}x{win['height']}")
    
    return result


def run_single_window_cycle(window_info, window_state):
    """
    Run one control cycle for a single game window.
    
    Args:
        window_info: Dictionary containing window information (x, y, width, height, hwnd, name)
        window_state: Dictionary containing per-window state:
            - game_state: Game state dictionary
            - state_estimation_i: State estimation counter
            - movement_count: Movement counter
            - game_start_flag: Whether game has started
            - score: Current quest score
            - before_quest_img: Previous quest image
    
    Returns:
        Updated window_state dictionary
    """
    hwnd = window_info['hwnd']
    window_name = window_info['name']
    
    print(f"\n{'='*80}")
    print(f"🎮 Controlling window: {window_name} (hwnd: {hwnd})")
    print(f"{'='*80}")
    
    # Activate window (retry a few times using robust foreground helper)
    activated = False
    for attempt in range(3):
        # increase timeout slightly on subsequent attempts
        try:
            if force_foreground(hwnd, timeout=1.0 + attempt * 0.5):
                activated = True
                break
        except Exception:
            pass
        time.sleep(0.25)

    if not activated:
        print(f"⚠️ Failed to activate window: {window_name}")
        return window_state

    # Small delay to ensure window is active
    time.sleep(0.5)

    # Capture frame
    x1, y1 = window_info['x'], window_info['y']
    x2, y2 = x1 + window_info['width'], y1 + window_info['height']
    frame = final_4.capture_region(x1, y1, x2, y2)
    
    if frame is None:
        print(f"⚠️ Failed to capture frame from window: {window_name}")
        return window_state
    
    # Save captured frame with window identifier
    # final_4.safe_imwrite(f"captured_frame_{hwnd}.png", frame)

    
    # Extract state variables
    game_state = window_state.get("game_state", {
        "movement_flag": False,
        "teleport_ability": False,
        "current_hp_state": 100,
        "jump_village_count": 0,
        "blood_bottle_count": 0
    })

    game_start_flag = window_state.get("game_start_flag", False)
    score = window_state.get("score", 1.0)
    before_quest_img = window_state.get("before_quest_img", None)
    
    # Extract per-window timer state (critical for multi-instance support)
    movement_flag_start_time = window_state.get("movement_flag_start_time", None)
    movement_flag_end_time = window_state.get("movement_flag_end_time", None)
    
    # Handle game start flag
    if(game_start_flag is False):
        if(final_4.search_region_in_frame('start_sign_region', frame)):
            game_start_flag = True
            temp_region = SAVED_REGIONS.get("start_sign_region", {})
            if temp_region:
                final_4.drag_mouse(temp_region)
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
        jump_village_count = final_4.crop_image(SAVED_REGIONS.get('jump_village_region'), frame, offset=[0,10,0,10])
        if jump_village_count is not None and jump_village_count.size > 0:
            bottle_count = final_4.extract_number_from_image(jump_village_count, region_label="jump_village_region")
            if bottle_count is not None and isinstance(bottle_count, (int, float)):
                game_state["jump_village_count"] = bottle_count
            else:
                game_state["jump_village_count"] = 0
        else:
            print("⚠️ Invalid jump village region or image")
            game_state["jump_village_count"] = game_state.get("jump_village_count", 100)
        #end getting jump_village_count


        #getting blood_bottle_count
        img_crop_bottle = final_4.crop_image(SAVED_REGIONS.get('blood_bottle_region'), frame, offset=[0,10,0,10])
        if img_crop_bottle is not None and img_crop_bottle.size > 0:
        
            blood_bottle_count = final_4.extract_number_from_image(img_crop_bottle, debug_save=True, region_label="blood_bottle_region", allow_decimals=False)
            if blood_bottle_count is not None and isinstance(blood_bottle_count, (int, float)):
                game_state["blood_bottle_count"] = blood_bottle_count
            else:
                game_state["blood_bottle_count"] = 0
        else:
            print("⚠️ Invalid blood bottle region or image")
            game_state["blood_bottle_count"] = game_state.get("blood_bottle_count", 100)
        game_state["current_hp_state"] = game_state.get("current_hp_state", 100)
        img_hp_crop = final_4.crop_image(SAVED_REGIONS.get('hp_bar_main_region'), frame)
        # Validate crop before attempting to show it. cv2.imshow requires a window name and a valid image.




        if img_hp_crop is not None and img_hp_crop.size > 0:
            try:
                current_hp_state = final_4.get_blood_percentage(img_hp_crop)
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

        first_quest = final_4.crop_image(SAVED_REGIONS['first_quest_region'], frame)
        score = final_4.handle_mission_area(first_quest)
        print("Mission  score test:",score)

        current_movement_flag = game_state.get("movement_flag", False)
        if current_movement_flag:
            # Movement flag is True - check if it just became True
            movement_flag_end_time = None
            if movement_flag_start_time is None:
                movement_flag_start_time = time.time()
                print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")
            
            # Check if movement_flag has been True for 5 minutes
            if movement_flag_start_time is not None:
                elapsed_time = time.time() - movement_flag_start_time
                if elapsed_time >= final_4.MOVEMENT_FLAG_TIMEOUT:
                    final_4.go_to_other_states("jump_button_region", frame, window_info=window_info)
                    movement_flag_start_time = None
        else:
            # Movement flag is False - reset timer
            movement_flag_start_time = None
            if movement_flag_end_time is None:
                movement_flag_end_time = time.time()
                print(f"⏱️ Movement flag became False at {time.strftime('%H:%M:%S')}")
            # Check if movement_flag has been False for 5 minutes
            if movement_flag_end_time is not None:
                elapsed_time = time.time() - movement_flag_end_time
                if elapsed_time >= final_4.MOVEMENT_FLAG_TIMEOUT:
                    final_4.go_to_other_states("jump_button_region", frame, window_info=window_info)
                    game_state["movement_flag"] = True
                    movement_flag_end_time = None

    except Exception as e:
        print(f"⚠️ Error Getting Game Status: {e}")


    try:
        # Initialize default HP state
        
        # Update game state and check if healing is needed
        updated_state = final_4.screen_state(frame, game_state)
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
            final_4.drag_mouse(temp_region)
        else:
            print("Found a region")
            final_4.click_saved_region_center(game_state["region_name"], method='auto', window_info=window_info)
            time.sleep(1.0)
        del game_state["region_name"]
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
        # Save timer state before returning
        window_state["movement_flag_start_time"] = movement_flag_start_time
        window_state["movement_flag_end_time"] = movement_flag_end_time
        window_state["game_state"] = game_state
        window_state["game_start_flag"] = game_start_flag
        window_state["score"] = score
        window_state["before_quest_img"] = before_quest_img
        return window_state
    
    # charge hp logic
    current_hp = game_state["current_hp_state"]
    if isinstance(current_hp, (int, float)) and current_hp < HEAL_THRESHOLD:
        if (game_state.get("jump_village_count", 0) > 0):
            final_4.click_saved_region_center('jump_village_region', method='auto', window_info=window_info)
            game_state["movement_flag"] = final_4.go_to_other_states("hp_charge_state","to_charge", frame, window_info=window_info)
            # Reset timer if movement_flag changed
            if game_state["movement_flag"]:
                movement_flag_start_time = None  # Will be set on next cycle if still True
        elif("teleport_ability" in game_state and game_state["teleport_ability"] is True and "hp_charge_state" not  in game_state):
            game_state["movement_flag"] = final_4.go_to_other_states("hp_charge_state", "to_village", frame, window_info=window_info)
            # Reset timer if movement_flag changed
            if game_state["movement_flag"]:
                movement_flag_start_time = None  # Will be set on next cycle if still True
        else:
            game_state["movement_flag"] = final_4.go_to_other_states("hp_charge_state", "to_village_using_map", frame, window_info=window_info)
            print("No teleport ability available and No jump_village_count.")
            # Reset timer if movement_flag changed
            if game_state["movement_flag"]:
                movement_flag_start_time = None  # Will be set on next cycle if still True

    # Safe score comparison with proper validation
    elif isinstance(score, (int, float)) and score < 0.3 and game_state.get("movement_flag") is True:
        if(game_state["current_hp_state"] > HEAL_THRESHOLD):
            print("score:",score,"movement_flag:",game_state.get("movement_flag")," Clicking jump button")
            final_4.go_to_other_states("jump_button_region", frame, window_info=window_info)
            game_state["movement_flag"] = True
            # Track when movement_flag becomes True
            if movement_flag_start_time is None:
                movement_flag_start_time = time.time()
                print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")

    elif(game_state.get("movement_flag") is False):
        final_4.go_to_other_states("jump_button_region", frame, window_info=window_info)
        game_state["movement_flag"] = True
        # Track when movement_flag becomes True
        if movement_flag_start_time is None:
            movement_flag_start_time = time.time()
            print(f"⏱️ Movement flag became True at {time.strftime('%H:%M:%S')}")
    
    # Print current state
    print(f"📊 Game State for {window_name}:")
    print(f"   HP: {game_state.get('current_hp_state', 'N/A')}")
    print(f"   Jump Village Count: {game_state.get('jump_village_count', 'N/A')}")
    print(f"   Blood Bottle Count: {game_state.get('blood_bottle_count', 'N/A')}")
    print(f"   Movement Flag: {game_state.get('movement_flag', 'N/A')}")
    print(f"   Score: {score}")
    
    # Update and return window state (including timer state)
    window_state["game_state"] = game_state
    window_state["game_start_flag"] = game_start_flag
    window_state["score"] = score
    window_state["before_quest_img"] = before_quest_img
    window_state["movement_flag_start_time"] = movement_flag_start_time
    window_state["movement_flag_end_time"] = movement_flag_end_time
    
    return window_state


def main():
    """
    Main multi-window control loop.
    
    Finds all game windows, maintains separate state for each,
    and cycles through them with 5-second intervals.
    """
    print("🚀 Starting Multi-Window Bot Controller...")
    print("=" * 80)
    
    # Position console window
    try:
        final_4.position_console_window()
    except Exception as e:
        print(f"⚠️ Could not position console window: {e}")
    
    # Find all game windows
    windows = get_all_game_windows()
    
    if not windows:
        print("❌ No game windows found. Please start at least one Lineage2M game.")
        return
    
    # Initialize per-window state storage
    # Key: hwnd (window handle), Value: window state dictionary
    window_states = {}
    
    for window_info in windows:
        hwnd = window_info['hwnd']
        window_states[hwnd] = {
            "game_state": {
                "movement_flag": False,
                "teleport_ability": False,
                "current_hp_state": 100,
                "jump_village_count": 0,
                "blood_bottle_count": 0
            },
            "game_start_flag": False,
            "score": 1.0,
            "before_quest_img": None,
            "movement_flag_start_time": None,
            "movement_flag_end_time": None
        }
    
    print(f"\n✅ Initialized {len(window_states)} window(s) for control")
    print(f"⏱️  Control interval: {WINDOW_SWITCH_INTERVAL} seconds between windows\n")
    
    # Daily notification flag - will be set per window if needed
    daily_notification_done = set()
    
    # Main control loop
    cycle_count = 0
    try:
        while not STOP_EVENT:
            cycle_count += 1
            print(f"\n{'#'*80}")
            print(f"🔄 Control Cycle #{cycle_count}")
            print(f"{'#'*80}\n")
            
            # Refresh window list periodically (every 10 cycles) to detect new windows
            if cycle_count % 10 == 1:
                new_windows = get_all_game_windows()
                # Add new windows to window_states if they don't exist
                for window_info in new_windows:
                    hwnd = window_info['hwnd']
                    if hwnd not in window_states:
                        print(f"✅ New window detected: {window_info['name']}")
                        window_states[hwnd] = {
                            "game_state": {
                                "movement_flag": False,
                                "teleport_ability": False,
                                "current_hp_state": 100,
                                "jump_village_count": 0,
                                "blood_bottle_count": 0
                            },
                            "game_start_flag": False,
                            "score": 1.0,
                            "before_quest_img": None
                        }
                # Update windows list
                windows = [w for w in new_windows if w['hwnd'] in window_states]
            
            if not windows:
                print("⚠️ No active windows. Waiting 5 seconds before retry...")
                time.sleep(5.0)
                windows = get_all_game_windows()
                continue
            
            # Cycle through all windows
            for idx, window_info in enumerate(windows, 1):
                if STOP_EVENT:
                    break
                
                hwnd = window_info['hwnd']
                
                # Skip if window no longer exists
                if not win32gui.IsWindow(hwnd):
                    print(f"⚠️ Window {window_info['name']} no longer exists. Removing from control list.")
                    if hwnd in window_states:
                        del window_states[hwnd]
                    continue
                
                # Skip if window state doesn't exist (shouldn't happen, but safety check)
                if hwnd not in window_states:
                    print(f"⚠️ Window state missing for {window_info['name']}. Initializing...")
                    window_states[hwnd] = {
                        "game_state": {
                            "movement_flag": False,
                            "teleport_ability": False,
                            "current_hp_state": 100,
                            "jump_village_count": 0,
                            "blood_bottle_count": 0
                        },
                        "game_start_flag": False,
                        "score": 1.0,
                        "before_quest_img": None,
                        "movement_flag_start_time": None,
                        "movement_flag_end_time": None
                    }
                
                # Handle daily notification per window (only once per window)
                if hwnd not in daily_notification_done:
                    try:
                        if final_4.should_notify_today():
                            print(f"📅 Processing daily rewards for window: {window_info['name']}")
                            # Activate window for daily tasks
                            if force_foreground(hwnd, timeout=2.0):
                                time.sleep(1.0)
                                final_4.click_saved_region_center("nav_button_region", method='auto', window_info=window_info)
                                time.sleep(1.0)
                                final_4.click_saved_region_center("daily_button_region", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("daily_claim_benefit_region", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("daily_bonus_confirm", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("out_button_region", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("nav_button_region", method='auto', window_info=window_info)
                                time.sleep(1.0)
                                final_4.click_saved_region_center("mail_button_region", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("claim_all_button_region", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("daily_bonus_confirm", method='auto', window_info=window_info)
                                time.sleep(2.0)
                                final_4.click_saved_region_center("out_button_region", method='auto', window_info=window_info)
                            daily_notification_done.add(hwnd)
                    except Exception as e:
                        print(f"⚠️ Failed to process daily notification for {window_info['name']}: {e}")
                
                # Run control cycle for this window
                try:
                    window_states[hwnd] = run_single_window_cycle(window_info, window_states[hwnd])
                except Exception as e:
                    print(f"⚠️ Error controlling window {window_info['name']}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Wait 5 seconds before next window (except after last window)
                if idx < len(windows) and not STOP_EVENT:
                    print(f"\n⏳ Waiting {WINDOW_SWITCH_INTERVAL} seconds before next window...\n")
                    time.sleep(WINDOW_SWITCH_INTERVAL)
            
            # After all windows, wait before starting next cycle
            if not STOP_EVENT:
                print(f"\n✅ Completed cycle for all {len(windows)} window(s)")
                print(f"⏳ Waiting {WINDOW_SWITCH_INTERVAL} seconds before next cycle...\n")
                time.sleep(WINDOW_SWITCH_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n🛑 Multi-window bot stopped by user.")
    except Exception as e:
        print(f"\n❌ Fatal error in multi-window controller: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Multi-window controller shutdown complete.")


if __name__ == "__main__":
    main()

