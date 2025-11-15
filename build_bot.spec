# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.datastruct import Tree

datas = [
    ('buttons', 'buttons'),
    ('region', 'region'),
    ('bot_config.json', '.'),
    ('bot_control.json', '.'),
    ('icon.ico', '.'),
] + collect_data_files('PyQt5') + collect_data_files('cv2')

a = Analysis(
    ['bot_gui.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'win32',
        'win32gui',
        'win32api',
        'win32con',
        'win32com',
        'win32com.client',
        'pywintypes',
        'dxcam',
        'cv2',
        'numpy',
        'pytesseract',
        'pyautogui',
        'pydirectinput',
        'skimage',
        'skimage.metrics',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.sip',
        'final_4',
        'multi_window_controller',
        'window_utils',
        'launcher',
        'launch_game',
        'stop_event',
        'resource_path',
    ] + collect_submodules('win32'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='LineageBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
)
