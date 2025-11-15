from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys
import subprocess
import time
from launch_game import is_game_running, launch_lineage2m
import final_4
from stop_event import STOP_EVENT
from launcher import launch_game, is_game_running
import json
from datetime import datetime
import os
import multi_window_controller

# --------------------------
# Worker Thread for Bot Logic
# --------------------------
class BotThread(QThread):
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def run(self):
        global STOP_EVENT
        try:
            STOP_EVENT = False
            self.status.emit("Searching for game windows...")
            time.sleep(1)
            multi_window_controller.main()
        except KeyboardInterrupt:
            self.status.emit("Bot stopped")
        except Exception as e:
            self.error.emit(str(e))


# --------------------------
# Main Application Window
# --------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("⚔️Bot Control Panel ⚙️")
        self.setGeometry(100, 100, 500, 300)
        self.setStyleSheet(self._load_dark_theme())

        # --- Central Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # --- Title ---
        self.title_label = QLabel("BOT")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("TitleLabel")
        layout.addWidget(self.title_label)

        # --- Status Panel ---
        self.status_frame = QFrame()
        self.status_frame.setObjectName("StatusFrame")
        status_layout = QVBoxLayout(self.status_frame)
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setObjectName("StatusLabel")
        status_layout.addWidget(self.status_label)
        layout.addWidget(self.status_frame)

        # --- Buttons ---
        self.start_button = QPushButton("▶ Start Bot")
        self.start_button.setObjectName("StartButton")
        self.start_button.clicked.connect(self.start_bot)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("⏹ Stop Bot")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)

        # self.config_button = QPushButton("⚙ Open Config")
        # self.config_button.setObjectName("ConfigButton")
        # self.config_button.clicked.connect(self.open_config)
        # layout.addWidget(self.config_button)

        self.bot_thread = None

    # --------------------------
    # Bot Control Logic
    # --------------------------
    def start_bot(self):
        self.status_label.setText("Starting bot...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        # self.config_button.setEnabled(False)

        try:
            ctrl_path = os.path.join(os.path.dirname(__file__), 'bot_control.json')
            control = {
                'timestamp': datetime.now().isoformat(),
                'game_state': {'bot_running': True, 'launched_from_gui': True}
            }
            with open(ctrl_path, 'w', encoding='utf-8') as cf:
                json.dump(control, cf)
        except Exception:
            pass

        self.bot_thread = BotThread()
        self.bot_thread.error.connect(self.on_bot_error)
        self.bot_thread.status.connect(self.status_label.setText)
        self.bot_thread.finished.connect(self.on_bot_finished)
        self.bot_thread.start()

    def stop_bot(self):
        if self.bot_thread and self.bot_thread.isRunning():
            self.status_label.setText("Stopping bot...")
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(True)
            # self.config_button.setEnabled(True)

            # Signal the thread to stop safely
            global STOP_EVENT
            STOP_EVENT = True

            # Wait briefly for the bot thread to finish
            self.bot_thread.wait(2000)  # 2 seconds timeout

            if self.bot_thread.isRunning():
                # Force terminate if not stopping gracefully
                try:
                    self.bot_thread.terminate()
                    self.bot_thread.wait()
                except Exception as e:
                    print(f"Force stop error: {e}")

            self.on_bot_finished()

            # --- Update bot_control.json state ---
            try:
                ctrl_path = os.path.join(os.path.dirname(__file__), 'bot_control.json')
                control = {
                    'timestamp': datetime.now().isoformat(),
                    'game_state': {'bot_running': False, 'stopped_from_gui': True}
                }
                with open(ctrl_path, 'w', encoding='utf-8') as cf:
                    json.dump(control, cf)
            except Exception as e:
                print(f"Failed to write control file: {e}")


    def on_bot_error(self, error_msg):
        QMessageBox.critical(self, "Bot Error", f"Bot encountered an error:\n{error_msg}")
        self.on_bot_finished()

    def on_bot_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        # self.config_button.setEnabled(True)
        self.status_label.setText("Ready")
        global STOP_EVENT
        STOP_EVENT = False

    def open_config(self):
        try:
            subprocess.Popen(['python', 'config_gui.py'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open config:\n{str(e)}")

    # --------------------------
    # Neon Dark Theme Stylesheet
    # --------------------------
    def _load_dark_theme(self):
        return """
        QMainWindow {
            background-color: #0a0f1a;
        }
        QLabel#TitleLabel {
            color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                   stop:0 #00b4ff, stop:1 #0077ff);
            font-size: 26px;
            font-weight: bold;
            letter-spacing: 2px;
            text-shadow: 0 0 10px #00b4ff;
        }
        QFrame#StatusFrame {
            background-color: #101826;
            border: 1px solid #0077ff;
            border-radius: 10px;
            padding: 15px;
        }
        QLabel#StatusLabel {
            color: #00ffff;
            font-size: 16px;
            font-weight: bold;
        }
        QPushButton {
            background-color: #0d1a26;
            color: #00b4ff;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #0077ff;
            border-radius: 12px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #112a3f;
            border: 2px solid #00ffff;
            color: #00ffff;
            box-shadow: 0 0 15px #00ffff;
        }
        QPushButton:disabled {
            background-color: #1a1a1a;
            color: #555;
            border: 2px solid #333;
        }
        """

# --------------------------
# Application Entry Point
# --------------------------
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
