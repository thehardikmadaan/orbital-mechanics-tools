# ui/main_window.py
import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFrame, QGridLayout)
from PySide6.QtCore import Qt

# Import our math engine
from core.astrodynamics import hohmann_transfer
from core.rocket_math import calculate_initial_mass


class OrbitalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Orbital Transfer System")
        self.setGeometry(100, 100, 600, 500)

        # --- QSS STYLING (Bringing your web CSS into PySide6) ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020408;
            }
            QLabel {
                color: #e8edf5;
                font-family: 'Outfit', sans-serif;
                font-size: 14px;
            }
            QLabel#Header {
                color: #00d4ff;
                font-family: 'Syne', sans-serif;
                font-size: 28px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#Eyebrow {
                color: #7b61ff;
                font-family: 'DM Mono', monospace;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 2px;
            }
            QFrame#Panel {
                background-color: #080d14;
                border: 1px solid rgba(0, 212, 255, 0.12);
                border-radius: 8px;
            }
            QLineEdit {
                background-color: #020408;
                color: #00d4ff;
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 4px;
                padding: 10px;
                font-family: 'DM Mono', monospace;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
                background-color: rgba(0, 212, 255, 0.05);
            }
            QPushButton {
                background-color: transparent;
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 4px;
                padding: 12px;
                font-family: 'Syne', sans-serif;
                font-weight: 700;
                font-size: 14px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #00d4ff;
                color: #020408;
            }
            QLabel#ResultLabel {
                color: #ff6b35;
                font-family: 'DM Mono', monospace;
                font-size: 16px;
                margin-top: 20px;
            }
        """)

        # --- LAYOUT SETUP ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # 1. Headers
        eyebrow = QLabel("Mission Control // Trajectory Planning")
        eyebrow.setObjectName("Eyebrow")
        header = QLabel("HOHMANN TRANSFER")
        header.setObjectName("Header")
        main_layout.addWidget(eyebrow)
        main_layout.addWidget(header)

        # 2. Input Panel (The dark surface frame)
        panel = QFrame()
        panel.setObjectName("Panel")
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)

        # Add input fields to the grid
        panel_layout.addWidget(QLabel("Initial LEO Altitude (km):"), 0, 0)
        self.input_r1 = QLineEdit("300")
        panel_layout.addWidget(self.input_r1, 0, 1)

        panel_layout.addWidget(QLabel("Target GEO Altitude (km):"), 1, 0)
        self.input_r2 = QLineEdit("35786")
        panel_layout.addWidget(self.input_r2, 1, 1)

        panel_layout.addWidget(QLabel("Payload Mass (kg):"), 2, 0)
        self.input_mass = QLineEdit("2000")
        panel_layout.addWidget(self.input_mass, 2, 1)

        main_layout.addWidget(panel)

        # 3. Calculate Button & Results
        self.calc_button = QPushButton("INITIATE CALCULATION")
        main_layout.addWidget(self.calc_button)

        self.result_label = QLabel("SYSTEM STANDBY. AWAITING INPUT.")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        main_layout.addStretch()

        # --- SIGNALS & SLOTS (The logic bridge) ---
        # When the button is clicked (Signal), run the calculate_mission function (Slot)
        self.calc_button.clicked.connect(self.calculate_mission)

    def calculate_mission(self):
        """This function runs when the button is clicked."""
        try:
            # 1. Grab the text from the UI boxes and turn them into numbers (floats)
            alt1_km = float(self.input_r1.text())
            alt2_km = float(self.input_r2.text())
            final_mass = float(self.input_mass.text())

            # 2. Constants
            mu = 3.986e14
            r_earth = 6371000
            isp = 310

            # 3. Math (Converting km to meters and adding Earth radius)
            r1 = (alt1_km * 1000) + r_earth
            r2 = (alt2_km * 1000) + r_earth

            # 4. Call our core engine
            delta_v = hohmann_transfer(mu, r1, r2)
            wet_mass = calculate_initial_mass(delta_v, isp, final_mass)
            propellant = wet_mass - final_mass

            # 5. Update the UI with the result
            result_text = f"SUCCESS: Δv Required: {delta_v:.2f} m/s | Propellant: {propellant:.2f} kg"
            self.result_label.setStyleSheet("color: #00d4ff;")  # Turn text cyan on success
            self.result_label.setText(result_text)

        except ValueError:
            # HMI Safety Feature: If the user types a letter instead of a number, catch the crash!
            self.result_label.setStyleSheet("color: #ff6b35;")  # Turn text orange/red on error
            self.result_label.setText("ERROR: INVALID INPUT DETECTED. NUMBERS ONLY.")


def run_app():
    app = QApplication(sys.argv)
    window = OrbitalDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()