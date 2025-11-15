import json
import os
import subprocess
import sys
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# optional imports for displaying/saving images
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import final_4
except Exception:
    final_4 = None

CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'bot_config.json')

# Default values (kept in sync with final_4.py defaults)
DEFAULTS = {
    "WINDOW_SUBSTRING": "Lineage2M",
    "HEAL_KEY": "1",
    "HEAL_THRESHOLD": 40.0,
    "CAPTURE_INTERVAL": 3.0,
    "MOTION_THRESHOLD": 10000,
    "MISSION_DIFF_THRESHOLD": 20,
    "CLICK_OFFSET_X": 0,
    "CLICK_OFFSET_Y": 5,
    "USE_REAL_MOUSE": True
    ,"MOTION_RATIO": 0.3
}


def load_config():
    # Robust loader: if the config file is missing, return defaults.
    # If the file exists but is invalid JSON, move the corrupt file aside
    # and attempt to use a .bak file if present to avoid losing SAVED_REGIONS.
    if not os.path.exists(CONFIG_FILE):
        return DEFAULTS.copy()
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        try:
            # backup corrupt config
            ts = int(time.time())
            corrupt_path = CONFIG_FILE + f'.corrupt.{ts}'
            try:
                os.replace(CONFIG_FILE, corrupt_path)
                print(f"⚠️ Config JSON parse failed — moved corrupt config to {corrupt_path}")
            except Exception:
                # if replace fails, try copy
                try:
                    import shutil
                    shutil.copy2(CONFIG_FILE, corrupt_path)
                    print(f"⚠️ Config JSON parse failed — copied corrupt config to {corrupt_path}")
                except Exception:
                    print(f"⚠️ Failed to back up corrupt config: {e}")
            # try to load bak if available
            bak = CONFIG_FILE + '.bak'
            if os.path.exists(bak):
                try:
                    with open(bak, 'r', encoding='utf-8') as f2:
                        return json.load(f2)
                except Exception:
                    pass
        except Exception:
            pass
        return DEFAULTS.copy()


def _safe_write_config(cfg):
    """Write config to disk atomically, keeping a .bak copy of the previous file.

    This reduces the chance of corrupting the main config and losing keys
    such as SAVED_REGIONS if multiple writers run concurrently.
    """
    try:
        import shutil
        # create backup of existing file
        if os.path.exists(CONFIG_FILE):
            try:
                bak = CONFIG_FILE + '.bak'
                shutil.copy2(CONFIG_FILE, bak)
            except Exception:
                pass
        tmp_path = CONFIG_FILE + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        # atomic replace
        try:
            os.replace(tmp_path, CONFIG_FILE)
        except Exception:
            # final fallback: copy
            shutil.copy2(tmp_path, CONFIG_FILE)
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ _safe_write_config failed: {e}")


class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Bot Configuration')
        self.resizable(False, False)

        self.cfg = load_config()

        frm = ttk.Frame(self, padding=12)
        frm.grid()

        row = 0
        def add_row(label, var):
            nonlocal row
            ttk.Label(frm, text=label).grid(column=0, row=row, sticky='w')
            ent = ttk.Entry(frm, textvariable=var, width=30)
            ent.grid(column=1, row=row, sticky='w')
            row += 1
            return ent

        self.window_sub = tk.StringVar(value=self.cfg.get('WINDOW_SUBSTRING', DEFAULTS['WINDOW_SUBSTRING']))
        add_row('Window substring', self.window_sub)

        self.heal_key = tk.StringVar(value=self.cfg.get('HEAL_KEY', DEFAULTS['HEAL_KEY']))
        add_row('Heal key', self.heal_key)

        self.heal_thr = tk.StringVar(value=str(self.cfg.get('HEAL_THRESHOLD', DEFAULTS['HEAL_THRESHOLD'])))
        add_row('Heal threshold (%)', self.heal_thr)

        self.cap_interval = tk.StringVar(value=str(self.cfg.get('CAPTURE_INTERVAL', DEFAULTS['CAPTURE_INTERVAL'])))
        add_row('Capture interval (s)', self.cap_interval)

    # (capture scale removed from UI) display will show the original captured image

        # Motion threshold UI removed; use MOTION_RATIO instead.
        # New: motion ratio (fraction changed pixels / area)
        self.motion_ratio = tk.StringVar(value=str(self.cfg.get('MOTION_RATIO', DEFAULTS.get('MOTION_RATIO', 0.3))))
        add_row('Motion ratio (0..1)', self.motion_ratio)

        self.mission_diff = tk.StringVar(value=str(self.cfg.get('MISSION_DIFF_THRESHOLD', DEFAULTS['MISSION_DIFF_THRESHOLD'])))
        add_row('Mission diff threshold', self.mission_diff)

        self.click_x = tk.StringVar(value=str(self.cfg.get('CLICK_OFFSET_X', DEFAULTS['CLICK_OFFSET_X'])))
        add_row('Click offset X', self.click_x)

        self.click_y = tk.StringVar(value=str(self.cfg.get('CLICK_OFFSET_Y', DEFAULTS['CLICK_OFFSET_Y'])))
        add_row('Click offset Y', self.click_y)

        self.use_real = tk.BooleanVar(value=bool(self.cfg.get('USE_REAL_MOUSE', DEFAULTS['USE_REAL_MOUSE'])))
        ttk.Checkbutton(frm, text='Use real OS mouse (move->click->restore)', variable=self.use_real).grid(column=0, row=row, columnspan=2, sticky='w')
        row += 1

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(column=0, row=row, columnspan=2, pady=(8,0))
        ttk.Button(btn_frame, text='Save', command=self.save).grid(column=0, row=0, padx=6)
        ttk.Button(btn_frame, text='Load', command=self.reload).grid(column=1, row=0, padx=6)
        # Start/Stop controls for the bot process
        self.start_btn = ttk.Button(btn_frame, text='Start bot', command=self.run_bot)
        self.start_btn.grid(column=2, row=0, padx=6)
        self.stop_btn = ttk.Button(btn_frame, text='Stop bot', command=self.stop_bot, state='disabled')
        self.stop_btn.grid(column=3, row=0, padx=6)
        ttk.Button(btn_frame, text='Capture Frame', command=self.capture_frame).grid(column=4, row=0, padx=6)
        ttk.Button(btn_frame, text='Quit', command=self.on_close).grid(column=5, row=0, padx=6)

        # Saved regions viewer / manager
        row += 1
        regs_frame = ttk.LabelFrame(frm, text='Saved Regions')
        regs_frame.grid(column=0, row=row, columnspan=2, pady=(8,0), sticky='we')

        self.reg_listbox = tk.Listbox(regs_frame, height=6, width=50)
        self.reg_listbox.grid(column=0, row=0, padx=(4,0), pady=6, sticky='nsew')
        regs_frame.grid_columnconfigure(0, weight=1)

        scr = ttk.Scrollbar(regs_frame, orient='vertical', command=self.reg_listbox.yview)
        scr.grid(column=1, row=0, sticky='ns', padx=(0,4), pady=6)
        self.reg_listbox.config(yscrollcommand=scr.set)

        reg_btn_fr = ttk.Frame(regs_frame)
        reg_btn_fr.grid(column=0, row=1, columnspan=2, pady=(0,6))
        ttk.Button(reg_btn_fr, text='Rename', command=self.rename_selected_region).pack(side='left', padx=6)
        ttk.Button(reg_btn_fr, text='Delete', command=self.delete_selected_region).pack(side='left', padx=6)
        ttk.Button(reg_btn_fr, text='Refresh', command=self.reload).pack(side='left', padx=6)

        # populate list initially
        self.reload()
        # process handle for launched bot (if any)
        self.bot_proc = None
        # start a periodic poll to update button states if bot was launched externally
        try:
            self.after(1000, self._poll_bot_process)
        except Exception:
            pass
        # ensure we save regions on close and handle window close button
        try:
            self.protocol('WM_DELETE_WINDOW', self.on_close)
        except Exception:
            pass

    def capture_frame(self):
        """Capture current game window frame (or full screen) and show a popup with size info."""
        if final_4 is None:
            messagebox.showerror('Unavailable', 'final_4 module not available — capture not possible.')
            return

        # get window rect if possible
        win_info = None
        try:
            win_info = final_4.get_window_info()
        except Exception:
            win_info = None

        try:
            if win_info:
                x, y = win_info['x'], win_info['y']
                w, h = win_info['width'], win_info['height']
                region = (x, y, w, h)
            else:
                user32 = final_4.ctypes.windll.user32
                sw = user32.GetSystemMetrics(0)
                sh = user32.GetSystemMetrics(1)
                region = (0, 0, sw, sh)

            # grab frame using dxcam instance from final_4
            frame = None
            try:
                frame = final_4.camera.grab(region=region)
            except Exception as e:
                print('⚠️ capture_frame: camera.grab failed:', e)

            if frame is None:
                messagebox.showerror('Capture failed', 'Could not capture frame (dxcam returned None).')
                return

            # convert BGR (dxcam returns BGR) to RGB for PIL
            try:
                import cv2
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = frame

            # save to screenshots folder
            screenshots = os.path.join(os.path.dirname(__file__), 'screenshots')
            os.makedirs(screenshots, exist_ok=True)

            # create a popup window to display the captured image and controls
            popup = tk.Toplevel(self)
            popup.title('Captured Frame')
            try:
                popup.transient(self)
                popup.grab_set()
            except Exception:
                pass
            ts = int(time.time())

            out_path = os.path.join(screenshots, f'capture_{ts}.png')

            if Image is not None:
                img = Image.fromarray(rgb)
                img.save(out_path)
            else:
                # fallback: try cv2.imwrite
                try:
                    import cv2
                    cv2.imwrite(out_path, frame)
                except Exception as e:
                    messagebox.showerror('Save failed', f'Could not save image: {e}')
            if Image is not None:
                # resize image if too large for screen and display on a Canvas to allow selection
                try:
                    # load original image
                    img = Image.open(out_path)
                    ow, oh = img.size
                    # display the original captured image (no UI scale control)
                    total_scale = 1.0
                    # draw width/height text on a copy of the image so original file remains unchanged
                    try:
                        from PIL import ImageDraw, ImageFont
                        disp_img = img.copy()
                        draw = ImageDraw.Draw(disp_img)
                        txt = f"{ow}×{oh}"
                        try:
                            font = ImageFont.truetype("arial.ttf", 20)
                        except Exception:
                            font = None
                        tw, th = draw.textsize(txt, font=font)
                        pad = 6
                        # semi-opaque background box (solid black for simplicity)
                        draw.rectangle((10 - pad, 10 - pad, 10 + tw + pad, 10 + th + pad), fill=(0, 0, 0))
                        # white text with simple outline for readability
                        draw.text((10, 10), txt, fill=(255, 255, 255), font=font)
                    except Exception:
                        disp_img = img
                    photo = ImageTk.PhotoImage(disp_img)

                    # Fit the image into the popup so the whole image is visible
                    # without scrollbars. Compute a fit scale so the displayed image
                    # fits within the screen minus margins.
                    screen_w = self.winfo_screenwidth()
                    screen_h = self.winfo_screenheight()
                    max_w = screen_w - 100
                    max_h = screen_h - 150
                    fit_scale = min(max_w / ow, max_h / oh, 1.0)
                    disp_w, disp_h = int(ow * fit_scale), int(oh * fit_scale)
                    disp_img = img.resize((disp_w, disp_h), Image.LANCZOS) if fit_scale != 1.0 else img
                    photo = ImageTk.PhotoImage(disp_img)

                    # layout: left = controls, right = image canvas
                    content_fr = ttk.Frame(popup)
                    content_fr.pack(fill='both', expand=True, padx=6, pady=6)
                    # left column with fixed width so it doesn't stretch too wide
                    left_fr = ttk.Frame(content_fr, width=150)
                    left_fr.pack(side='left', fill='y', padx=(0,8))
                    try:
                        left_fr.pack_propagate(False)
                    except Exception:
                        pass
                    right_fr = ttk.Frame(content_fr)
                    right_fr.pack(side='right', fill='both', expand=True)

                    canvas = tk.Canvas(right_fr, width=disp_w, height=disp_h, cursor='cross')
                    canvas.photo = photo
                    canvas.create_image(0, 0, anchor='nw', image=photo)
                    # draw saved regions (if any) on top of the displayed capture
                    try:
                        cfg_now = load_config() or {}
                        regs = cfg_now.get('SAVED_REGIONS', {})
                    except Exception:
                        regs = {}

                    # simple palette of readable colors
                    palette = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']

                    # Container for per-region visibility flags (default True)
                    visible_vars = {}

                    def draw_overlays():
                        # remove previous overlay items
                        try:
                            canvas.delete('overlay')
                        except Exception:
                            pass
                        try:
                            for i, (rname, r) in enumerate(regs.items()):
                                var = visible_vars.get(rname)
                                if var is None or not var.get():
                                    continue
                                try:
                                    rx = int(r.get('x', 0))
                                    ry = int(r.get('y', 0))
                                    rw = int(r.get('w', 0))
                                    rh = int(r.get('h', 0))
                                except Exception:
                                    continue
                                # Support multiple coordinate formats in config:
                                # - window-relative coords with '_coord': 'rel' (new format)
                                # - reference coords saved with '_coord': 'ref'
                                # - legacy absolute screen coords (no marker)
                                try:
                                    if r.get('_coord') == 'rel':
                                        # Window-relative coordinates: need to convert to display coordinates
                                        # region[0], region[1] is the window position (win_x, win_y)
                                        # region[2], region[3] is the window size (win_w, win_h)
                                        win_x = region[0]
                                        win_y = region[1]
                                        win_w = region[2]
                                        win_h = region[3]
                                        
                                        # Check if we have reference window size for scaling
                                        ref_w = r.get('_ref_width', 0)
                                        ref_h = r.get('_ref_height', 0)
                                        
                                        if ref_w > 0 and ref_h > 0 and (win_w != ref_w or win_h != ref_h):
                                            # Window size changed - scale coordinates
                                            scale_x = win_w / float(ref_w)
                                            scale_y = win_h / float(ref_h)
                                            rx_scaled = int(rx * scale_x)
                                            ry_scaled = int(ry * scale_y)
                                            rw_scaled = int(rw * scale_x)
                                            rh_scaled = int(rh * scale_y)
                                        else:
                                            # No scaling needed
                                            rx_scaled = rx
                                            ry_scaled = ry
                                            rw_scaled = rw
                                            rh_scaled = rh
                                        
                                        # Window-relative coords are already relative to window, so use directly
                                        rel_x = rx_scaled
                                        rel_y = ry_scaled
                                        if rel_x + rw_scaled <= 0 or rel_y + rh_scaled <= 0 or rel_x >= ow or rel_y >= oh:
                                            continue
                                        x0 = max(0, rel_x)
                                        y0 = max(0, rel_y)
                                        x1 = min(ow, rel_x + rw_scaled)
                                        y1 = min(oh, rel_y + rh_scaled)
                                        dx0 = int(x0 * fit_scale)
                                        dy0 = int(y0 * fit_scale)
                                        dx1 = int(x1 * fit_scale)
                                        dy1 = int(y1 * fit_scale)
                                    elif r.get('_coord') == 'ref':
                                        REF_W, REF_H = 1920.0, 1030.0
                                        dx0 = int(rx * disp_w / REF_W)
                                        dy0 = int(ry * disp_h / REF_H)
                                        dx1 = int((rx + rw) * disp_w / REF_W)
                                        dy1 = int((ry + rh) * disp_h / REF_H)
                                        if dx1 <= 0 or dy1 <= 0 or dx0 >= disp_w or dy0 >= disp_h:
                                            continue
                                    else:
                                        # Legacy absolute screen coordinates
                                        sx = rx
                                        sy = ry
                                        sw = rw
                                        sh = rh
                                        rel_x = sx - region[0]
                                        rel_y = sy - region[1]
                                        if rel_x + sw <= 0 or rel_y + sh <= 0 or rel_x >= ow or rel_y >= oh:
                                            continue
                                        x0 = max(0, rel_x)
                                        y0 = max(0, rel_y)
                                        x1 = min(ow, rel_x + sw)
                                        y1 = min(oh, rel_y + sh)
                                        dx0 = int(x0 * fit_scale)
                                        dy0 = int(y0 * fit_scale)
                                        dx1 = int(x1 * fit_scale)
                                        dy1 = int(y1 * fit_scale)
                                except Exception as e:
                                    print(f"⚠️ Error drawing overlay for region '{rname}': {e}")
                                    continue
                                color = palette[i % len(palette)]
                                # tag overlay items so they can be cleared/redrawn
                                canvas.create_rectangle(dx0, dy0, dx1, dy1, outline=color, width=2, tags=('overlay',))
                                try:
                                    canvas.create_text(dx0 + 4, dy0 + 10, text=rname, anchor='nw', fill=color, font=('Arial', 10, 'bold'), tags=('overlay',))
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # initial visible flags (all on)
                    for rname in regs.keys():
                        visible_vars[rname] = tk.BooleanVar(value=True)

                    # pack canvas now so layout has correct size before drawing overlays

                    # Create a small toolbar of buttons above the image so they remain
                    # visible even when the captured image is large. The commands are
                    # provided as lambdas so the save handlers (defined later) may be
                    # bound when invoked.
                    # left column: controls grouped together to save horizontal space
                    controls_fr = ttk.Frame(left_fr)
                    controls_fr.pack(fill='x', padx=4, pady=(6,4), anchor='nw')

                    top_btns = ttk.Frame(controls_fr)
                    ttk.Button(top_btns, text='Save Region', command=lambda: save_region()).pack(fill='x', pady=1)
                    ttk.Button(top_btns, text='Save Scaled', command=lambda: save_scaled()).pack(fill='x', pady=1)
                    ttk.Button(top_btns, text='Close', command=popup.destroy).pack(fill='x', pady=1)
                    top_btns.pack(side='top', fill='x', padx=2)

                    # zoom controls
                    zoom_btns = ttk.Frame(controls_fr)
                    def zoom_in():
                        nonlocal disp_w, disp_h, photo, fit_scale
                        new_scale = fit_scale * 1.2
                        disp_w = int(ow * new_scale)
                        disp_h = int(oh * new_scale)
                        disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(disp_img)
                        canvas.config(width=disp_w, height=disp_h)
                        canvas.delete('all')
                        canvas.create_image(0, 0, anchor='nw', image=photo)
                        fit_scale = new_scale
                        draw_overlays()
                    def zoom_out():
                        nonlocal disp_w, disp_h, photo, fit_scale
                        new_scale = max(0.1, fit_scale / 1.2)
                        disp_w = int(ow * new_scale)
                        disp_h = int(oh * new_scale)
                        disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(disp_img)
                        canvas.config(width=disp_w, height=disp_h)
                        canvas.delete('all')
                        canvas.create_image(0, 0, anchor='nw', image=photo)
                        fit_scale = new_scale
                        draw_overlays()
                    ttk.Button(zoom_btns, text='Zoom In (+20%)', command=zoom_in).pack(fill='x', pady=1)
                    ttk.Button(zoom_btns, text='Zoom Out (-20%)', command=zoom_out).pack(fill='x', pady=1)
                    zoom_btns.pack(side='top', fill='x', padx=2, pady=(1,0))

                    sel_label = ttk.Label(controls_fr, text='No selection')
                    sel_label.pack(padx=4, pady=(6,4), anchor='w')

                    name_var = tk.StringVar(value='region1')
                    name_fr = ttk.Frame(controls_fr)
                    ttk.Label(name_fr, text='Name:').pack(side='left')
                    ttk.Entry(name_fr, textvariable=name_var, width=18).pack(side='left', padx=(4,0))
                    name_fr.pack(padx=4, pady=(0,4), anchor='w')

                    # controls for toggling each region overlay visibility (on left column)
                    overlay_ctrl = ttk.LabelFrame(left_fr, text='Overlays')
                    overlay_ctrl.pack(fill='both', padx=2, pady=(4,6), expand=True)

                    # make overlays scrollable and responsive
                    overlay_canvas = tk.Canvas(overlay_ctrl, borderwidth=0, highlightthickness=0)
                    overlay_scroll = ttk.Scrollbar(overlay_ctrl, orient='vertical', command=overlay_canvas.yview)
                    overlay_inner = ttk.Frame(overlay_canvas)
                    
                    # configure canvas and scrollbar
                    overlay_canvas.configure(yscrollcommand=overlay_scroll.set)
                    overlay_canvas.pack(side='left', fill='both', expand=True, padx=1, pady=1)
                    overlay_scroll.pack(side='right', fill='y')
                    
                    # create window for inner frame
                    overlay_inner_id = overlay_canvas.create_window((0, 0), window=overlay_inner, anchor='nw', width=overlay_canvas.winfo_width())
                    
                    def _on_inner_config(ev):
                        # update scroll region when inner frame changes
                        try:
                            overlay_canvas.configure(scrollregion=overlay_canvas.bbox('all'))
                            # ensure inner frame matches canvas width
                            overlay_canvas.itemconfig(overlay_inner_id, width=overlay_canvas.winfo_width())
                        except Exception:
                            pass
                            
                    def _on_canvas_config(ev):
                        # update inner frame width when canvas resizes
                        try:
                            overlay_canvas.itemconfig(overlay_inner_id, width=overlay_canvas.winfo_width())
                        except Exception:
                            pass
                            
                    overlay_inner.bind('<Configure>', _on_inner_config)
                    overlay_canvas.bind('<Configure>', _on_canvas_config)

                    # create checkbuttons in a flow layout
                    flow_frame = ttk.Frame(overlay_inner)
                    flow_frame.pack(fill='both', expand=True, padx=1, pady=1)
                    
                    for rname in regs.keys():
                        cb = ttk.Checkbutton(flow_frame, text=rname, variable=visible_vars[rname], command=draw_overlays)
                        cb.pack(side='top', anchor='w', padx=2, pady=1, fill='x')

                    # draw overlays initially
                    draw_overlays()

                    canvas.pack(padx=4, pady=4, fill='both', expand=True)

                    rect_id = None
                    start = {'x': 0, 'y': 0}
                    sel = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}

                    def to_image_coords(cx, cy):
                        # convert display (canvas) coords to original image coords
                        img_x = int(cx / fit_scale)
                        img_y = int(cy / fit_scale)
                        return img_x, img_y

                    def on_button_press(ev):
                        # store canvas-space coordinates
                        start['x'], start['y'] = int(ev.x), int(ev.y)
                        nonlocal rect_id
                        if rect_id:
                            canvas.delete(rect_id)
                            rect_id = None

                    def on_move(ev):
                        nonlocal rect_id
                        x0, y0 = start['x'], start['y']
                        x1, y1 = int(ev.x), int(ev.y)
                        # clamp to image dims
                        x1 = max(0, min(x1, disp_w))
                        y1 = max(0, min(y1, disp_h))
                        if rect_id:
                            canvas.coords(rect_id, x0, y0, x1, y1)
                        else:
                            rect_id = canvas.create_rectangle(x0, y0, x1, y1, outline='red', width=2)
                        w_sel = abs(x1 - x0)
                        h_sel = abs(y1 - y0)
                        sel_label.config(text=f'Selection: {w_sel}×{h_sel} (display coords)')

                    def on_button_release(ev):
                        x0, y0 = start['x'], start['y']
                        x1, y1 = int(ev.x), int(ev.y)
                        x1 = max(0, min(x1, disp_w))
                        y1 = max(0, min(y1, disp_h))
                        sel['x1'], sel['y1'], sel['x2'], sel['y2'] = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
                        w_sel = sel['x2'] - sel['x1']
                        h_sel = sel['y2'] - sel['y1']
                        sel_label.config(text=f'Selection: {w_sel}×{h_sel} (display coords)')

                    canvas.bind('<ButtonPress-1>', on_button_press)
                    canvas.bind('<B1-Motion>', on_move)
                    canvas.bind('<ButtonRelease-1>', on_button_release)

                    # show effective display scale (always 100% since UI scale removed)
                    display_pct = 100
                    ttk.Label(popup, text=f'Display scale: {display_pct}% (of original)').pack(padx=8, pady=(0,6))

                    def save_scaled():
                        try:
                            scaled_out = out_path.replace('.png', f'_scaled_{display_pct}.png')
                            # disp_img is a PIL.Image instance
                            disp_img.save(scaled_out)
                            messagebox.showinfo('Saved', f'Scaled image saved to {scaled_out}')
                        except Exception as e:
                            messagebox.showerror('Save failed', f'Could not save scaled image: {e}')

                    def save_region():
                        # compute absolute screen coords of selection
                        if sel['x2'] - sel['x1'] <= 0 or sel['y2'] - sel['y1'] <= 0:
                            messagebox.showerror('No selection', 'Please select a region first.')
                            return
                        # convert selection (display coords) to original image coords
                        sx = region[0] + int(sel['x1'] / fit_scale)
                        sy = region[1] + int(sel['y1'] / fit_scale)
                        sw = int((sel['x2'] - sel['x1']) / fit_scale)
                        sh = int((sel['y2'] - sel['y1']) / fit_scale)
                        # Calculate window-relative coordinates
                        # region[0], region[1] is the window position (win_x, win_y)
                        # region[2], region[3] is the window size (win_w, win_h)
                        win_x = region[0]
                        win_y = region[1]
                        win_w = region[2]
                        win_h = region[3]
                        # Convert absolute screen coords to window-relative coords
                        rel_x = sx - win_x  # X position relative to window
                        rel_y = sy - win_y  # Y position relative to window
                        # Save as window-relative coordinates so regions work regardless of window position/size
                        try:
                            rel_x = int(rel_x)
                            rel_y = int(rel_y)
                            rel_w = max(1, int(sw))
                            rel_h = max(1, int(sh))
                        except Exception:
                            rel_x, rel_y, rel_w, rel_h = 0, 0, 1, 1
                        # write into config under SAVED_REGIONS using window-relative coords
                        cfg = load_config()
                        regs = cfg.get('SAVED_REGIONS', {})
                        name = name_var.get() or f'region_{int(time.time())}'
                        # store window-relative coords with '_coord': 'rel' marker
                        # Also store the reference window size for scaling when window is resized
                        # This allows regions to work regardless of window position or size
                        regs[name] = {
                            'x': rel_x, 
                            'y': rel_y, 
                            'w': rel_w, 
                            'h': rel_h,
                            '_coord': 'rel',  # Window-relative coordinates
                            '_ref_width': win_w,  # Reference window width when saved
                            '_ref_height': win_h  # Reference window height when saved
                        }
                        cfg['SAVED_REGIONS'] = regs
                        try:
                            _safe_write_config(cfg)
                            messagebox.showinfo('Saved', f'Region "{name}" saved to config ({sw}×{sh}).')
                            # Also save cropped region image into ./region/<name>.png
                            try:
                                region_dir = os.path.join(os.path.dirname(__file__), 'region')
                                os.makedirs(region_dir, exist_ok=True)
                                # img is the PIL image opened earlier (original size ow x oh)
                                img_x = sx - region[0]
                                img_y = sy - region[1]
                                crop_box = (img_x, img_y, img_x + sw, img_y + sh)
                                try:
                                    region_img = img.crop(crop_box)
                                    save_path = os.path.join(region_dir, f"{name}.png")
                                    region_img.save(save_path)
                                except Exception:
                                    # fallback to cv2 if PIL crop/save fails
                                    try:
                                        import cv2
                                        arr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        crop = arr[img_y:img_y+sh, img_x:img_x+sw]
                                        save_path = os.path.join(region_dir, f"{name}.png")
                                        cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                                    except Exception:
                                        save_path = None
                                if save_path:
                                    print(f"🖼 Saved region image to {save_path}")
                            except Exception:
                                pass
                        except Exception as e:
                            messagebox.showerror('Save failed', str(e))

                    # Note: top toolbar contains the same action buttons so they remain visible.
                except Exception as e:
                    ttk.Label(popup, text=f'Could not open image: {e}').pack(padx=8, pady=6)
                    ttk.Button(popup, text='Close', command=popup.destroy).pack(pady=6)
        except Exception as e:
            messagebox.showerror('Capture failed', f'Error capturing frame: {e}')
            return
        self.click_x.set(str(self.cfg.get('CLICK_OFFSET_X', DEFAULTS['CLICK_OFFSET_X'])))
        self.click_y.set(str(self.cfg.get('CLICK_OFFSET_Y', DEFAULTS['CLICK_OFFSET_Y'])))
        self.use_real.set(bool(self.cfg.get('USE_REAL_MOUSE', DEFAULTS['USE_REAL_MOUSE'])))

    def reload(self):
        """Reload config from disk and update UI fields and saved regions list."""
        self.cfg = load_config()
        self.window_sub.set(self.cfg.get('WINDOW_SUBSTRING', DEFAULTS['WINDOW_SUBSTRING']))
        self.heal_key.set(self.cfg.get('HEAL_KEY', DEFAULTS['HEAL_KEY']))
        self.heal_thr.set(str(self.cfg.get('HEAL_THRESHOLD', DEFAULTS['HEAL_THRESHOLD'])))
        self.cap_interval.set(str(self.cfg.get('CAPTURE_INTERVAL', DEFAULTS['CAPTURE_INTERVAL'])))
        # motion threshold removed from GUI; only motion_ratio is shown
        self.motion_ratio.set(str(self.cfg.get('MOTION_RATIO', DEFAULTS.get('MOTION_RATIO', 0.3))))
        self.mission_diff.set(str(self.cfg.get('MISSION_DIFF_THRESHOLD', DEFAULTS['MISSION_DIFF_THRESHOLD'])))
        self.click_x.set(str(self.cfg.get('CLICK_OFFSET_X', DEFAULTS['CLICK_OFFSET_X'])))
        self.click_y.set(str(self.cfg.get('CLICK_OFFSET_Y', DEFAULTS['CLICK_OFFSET_Y'])))
        self.use_real.set(bool(self.cfg.get('USE_REAL_MOUSE', DEFAULTS['USE_REAL_MOUSE'])))
        # refresh saved regions listbox
        self.refresh_saved_regions_list()

    def save_saved_regions(self):
        """Write current saved regions to the config file immediately."""
        try:
            cfg = load_config() or {}
            regs = getattr(self, 'saved_regions', None)
            if regs is None:
                regs = cfg.get('SAVED_REGIONS', {})
            cfg['SAVED_REGIONS'] = regs
            _safe_write_config(cfg)
        except Exception as e:
            try:
                messagebox.showerror('Save failed', f'Could not save regions: {e}')
            except Exception:
                print('Could not save regions:', e)

    def on_close(self):
        """Handler for quitting the GUI: persist config (including regions) then exit."""
        try:
            # save other settings first
            try:
                self.save()
            except Exception:
                pass
            # always persist saved regions
            try:
                self.save_saved_regions()
            except Exception:
                pass
        finally:
            try:
                self.destroy()
            except Exception:
                try:
                    self.quit()
                except Exception:
                    pass

    def refresh_saved_regions_list(self):
        cfg = load_config()
        regs = cfg.get('SAVED_REGIONS', {}) if cfg else {}
        self.saved_regions = regs
        self.reg_listbox.delete(0, tk.END)
        if not regs:
            self.reg_listbox.insert(tk.END, '(no saved regions)')
            return
        for name, r in regs.items():
            try:
                txt = f"{name}: {r.get('x')},{r.get('y')} {r.get('w')}x{r.get('h')}"
            except Exception:
                txt = f"{name}: {r}"
            self.reg_listbox.insert(tk.END, txt)

    def rename_selected_region(self):
        sel = self.reg_listbox.curselection()
        if not sel:
            messagebox.showerror('No selection', 'Please select a region to rename.')
            return
        idx = sel[0]
        names = list(self.saved_regions.keys())
        if idx >= len(names):
            messagebox.showerror('Invalid selection', 'Selected index out of range.')
            return
        old = names[idx]
        new = simpledialog.askstring('Rename region', 'New name for region:', initialvalue=old)
        if not new or not new.strip():
            return
        new = new.strip()
        if new in self.saved_regions:
            messagebox.showerror('Exists', 'A region with this name already exists.')
            return
        cfg = load_config()
        regs = cfg.get('SAVED_REGIONS', {})
        regs[new] = regs.pop(old)
        cfg['SAVED_REGIONS'] = regs
        try:
            # attempt to rename region image file if present
            region_dir = os.path.join(os.path.dirname(__file__), 'region')
            old_path = os.path.join(region_dir, f"{old}.png")
            new_path = os.path.join(region_dir, f"{new}.png")
            if os.path.exists(old_path):
                try:
                    os.makedirs(region_dir, exist_ok=True)
                    # use replace so it overwrites existing new_path if any
                    os.replace(old_path, new_path)
                except Exception as efile:
                    # non-fatal: continue but inform the user
                    messagebox.showwarning('File rename', f'Renamed config but could not rename image file: {efile}')

            _safe_write_config(cfg)
            messagebox.showinfo('Renamed', f'Region "{old}" renamed to "{new}"')
        except Exception as e:
            messagebox.showerror('Rename failed', str(e))
        self.refresh_saved_regions_list()

    def delete_selected_region(self):
        sel = self.reg_listbox.curselection()
        if not sel:
            messagebox.showerror('No selection', 'Please select a region to delete.')
            return
        idx = sel[0]
        names = list(self.saved_regions.keys())
        if idx >= len(names):
            messagebox.showerror('Invalid selection', 'Selected index out of range.')
            return
        name = names[idx]
        if not messagebox.askyesno('Confirm', f'Delete region "{name}"?'):
            return
        cfg = load_config()
        regs = cfg.get('SAVED_REGIONS', {})
        if name in regs:
            regs.pop(name, None)
            cfg['SAVED_REGIONS'] = regs
            try:
                _safe_write_config(cfg)
                # attempt to remove the saved region image if it exists
                try:
                    region_dir = os.path.join(os.path.dirname(__file__), 'region')
                    img_path = os.path.join(region_dir, f"{name}.png")
                    if os.path.exists(img_path):
                        os.remove(img_path)
                except Exception as efile:
                    messagebox.showwarning('File delete', f'Region removed from config but failed to delete image file: {efile}')
                messagebox.showinfo('Deleted', f'Region "{name}" removed from config.')
            except Exception as e:
                messagebox.showerror('Delete failed', str(e))
        self.refresh_saved_regions_list()

    def save(self):
        # Load existing config and update only the editable keys so we do not
        # accidentally wipe out SAVED_REGIONS or other fields maintained by
        # the GUI or other tools.
        try:
            cfg = load_config() or {}
            cfg.update({
                'WINDOW_SUBSTRING': self.window_sub.get(),
                'HEAL_KEY': self.heal_key.get(),
                'HEAL_THRESHOLD': float(self.heal_thr.get()),
                'CAPTURE_INTERVAL': float(self.cap_interval.get()),
                'MOTION_RATIO': float(self.motion_ratio.get()),
                'MISSION_DIFF_THRESHOLD': int(self.mission_diff.get()),
                'CLICK_OFFSET_X': int(self.click_x.get()),
                'CLICK_OFFSET_Y': int(self.click_y.get()),
                'USE_REAL_MOUSE': bool(self.use_real.get())
            })
        except Exception as e:
            messagebox.showerror('Invalid value', f'Please check entered values: {e}')
            return

        try:
            _safe_write_config(cfg)
            messagebox.showinfo('Saved', f'Configuration saved to {CONFIG_FILE}')
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def run_bot(self):
        """Save current config and launch final_4.py in a new console."""
        # save configuration first (preserve saved regions)
        self.save()
        final_py = os.path.join(os.path.dirname(__file__), 'final_4.py')
        if not os.path.exists(final_py):
            messagebox.showerror('Not found', f'final_4.py not found at {final_py}')
            return
        if self.bot_proc is not None and self.bot_proc.poll() is None:
            messagebox.showwarning('Already running', 'Bot appears to be already running.')
            return
        try:
            # Launch in a new console so the GUI stays responsive and keep the Popen
            self.bot_proc = subprocess.Popen([sys.executable, final_py], creationflags=subprocess.CREATE_NEW_CONSOLE)
            # Write control file to signal start to any running bot process
            try:
                control = {'timestamp': datetime.now().isoformat(), 'game_state': {'bot_running': True, 'launched_from_gui': True}}
                ctrl_path = os.path.join(os.path.dirname(__file__), 'bot_control.json')
                with open(ctrl_path, 'w', encoding='utf-8') as cf:
                    json.dump(control, cf)
            except Exception as e:
                print(f"⚠️ Could not write control file after launch: {e}")
            # update buttons
            try:
                self.start_btn.config(state='disabled')
                self.stop_btn.config(state='normal')
            except Exception:
                pass
            messagebox.showinfo('Launched', 'Bot launched in a new console.')
        except Exception as e:
            messagebox.showerror('Launch failed', str(e))

    def stop_bot(self):
        """Attempt to gracefully terminate the launched bot process."""
        if not getattr(self, 'bot_proc', None):
            messagebox.showinfo('Not running', 'No bot process is currently tracked by the GUI.')
            return
        try:
            # signal intent to stop via control file before terminating process
            try:
                control = {'timestamp': datetime.now().isoformat(), 'game_state': {'bot_running': False, 'stopped_from_gui': True}}
                ctrl_path = os.path.join(os.path.dirname(__file__), 'bot_control.json')
                with open(ctrl_path, 'w', encoding='utf-8') as cf:
                    json.dump(control, cf)
            except Exception as e:
                print(f"⚠️ Could not write control file before stop: {e}")
            if self.bot_proc.poll() is None:
                # try a gentle terminate first
                self.bot_proc.terminate()
                # wait briefly
                try:
                    self.bot_proc.wait(timeout=3)
                except Exception:
                    # force kill if still alive
                    try:
                        self.bot_proc.kill()
                    except Exception:
                        pass
            messagebox.showinfo('Stopped', 'Bot process has been stopped.')
        except Exception as e:
            messagebox.showerror('Stop failed', str(e))
        finally:
            self.bot_proc = None
            try:
                self.start_btn.config(state='normal')
                self.stop_btn.config(state='disabled')
            except Exception:
                pass

    def _poll_bot_process(self):
        """Periodic check to update Start/Stop button states if bot exits externally."""
        try:
            proc = getattr(self, 'bot_proc', None)
            if proc is not None:
                if proc.poll() is not None:
                    # process ended
                    self.bot_proc = None
                    try:
                        self.start_btn.config(state='normal')
                        self.stop_btn.config(state='disabled')
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.after(1000, self._poll_bot_process)
        except Exception:
            pass

if __name__ == '__main__':
    app = ConfigGUI()
    app.mainloop()
