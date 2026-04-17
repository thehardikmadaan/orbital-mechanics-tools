# ui/main_window.py
import sys
import math
import pandas as pd
import joblib
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFrame, QGridLayout,
                               QSlider, QComboBox, QSizePolicy)
from PySide6.QtCore import Qt, QTimer

from core.astrodynamics import hohmann_transfer, bi_elliptic_transfer, phasing_maneuver
from core.rocket_math import calculate_initial_mass

from visualization.plot_orbit import OrbitPlotter
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
class OrbitalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Orbital Transfer System")
        self.setMinimumSize(1000, 750)
        self.setGeometry(100, 100, 1000, 750)

        # STYLING
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
                font-family: 'DM Mono', Courier New;
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
                font-family: 'DM Mono', Courier New;
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
                font-family: 'DM Mono', Courier New;
                font-size: 16px;
                margin-top: 20px;
            }
        """)

        # LAYOUT SETUP
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # 1. Headers
        eyebrow = QLabel("Mission Control // Trajectory Planning")
        eyebrow.setObjectName("Eyebrow")
        self.header = QLabel("HOHMANN TRANSFER")
        self.header.setObjectName("Header")
        main_layout.addWidget(eyebrow)
        main_layout.addWidget(self.header)

        # 2. Input Panel (Building it top-to-bottom)
        panel = QFrame()
        panel.setObjectName("Panel")
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)

        panel_layout.addWidget(QLabel("Parking Orbit (h_park, km):"), 0, 0)
        self.input_r1 = QLineEdit("300")
        panel_layout.addWidget(self.input_r1, 0, 1)

        panel_layout.addWidget(QLabel("Target Orbit (h_tgt, km):"), 1, 0)
        self.input_r2 = QLineEdit("35786")
        panel_layout.addWidget(self.input_r2, 1, 1)

        panel_layout.addWidget(QLabel("Dry Mass (m_dry, kg):"), 2, 0)
        self.input_mass = QLineEdit("2000")
        panel_layout.addWidget(self.input_mass, 2, 1)

        panel_layout.addWidget(QLabel("Maneuver Profile:"), 3, 0)
        self.maneuver_box = QComboBox()
        self.maneuver_box.addItems(["Hohmann Transfer", "Bi-Elliptic Transfer", "Phasing Orbit"])
        self.maneuver_box.setStyleSheet("""
            QComboBox {
                background-color: #020408;
                color: #00d4ff;
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 4px;
                padding: 5px;
            }
        """)
        panel_layout.addWidget(self.maneuver_box, 3, 1)

        panel_layout.addWidget(QLabel("Simulation Speed:"), 4, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #5a6a80;
                height: 8px;
                background: #020408;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff6b35;
                width: 18px;
                margin: -5px 0; 
                border-radius: 9px;
            }
        """)
        panel_layout.addWidget(self.speed_slider, 4, 1)

        self.label_rb = QLabel("Apoapsis Radius (r_a, km):")
        self.input_rb = QLineEdit("100000")
        panel_layout.addWidget(self.label_rb, 5, 0)
        panel_layout.addWidget(self.input_rb, 5, 1)
        self.label_rb.hide()
        self.input_rb.hide()

        # Add the completed panel to the main layout ONCE
        main_layout.addWidget(panel)

        # 3. Calculate Button & Results
        self.calc_button = QPushButton("INITIATE CALCULATION")
        main_layout.addWidget(self.calc_button)

        self.result_label = QLabel("SYSTEM STANDBY. AWAITING INPUT.")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        # 4. Matplotlib Graph (With expanding policy)
        self.plotter = OrbitPlotter(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plotter)

        # Push everything up from the bottom
        main_layout.addStretch()

        # 5. SIGNALS & SLOTS
        self.calc_button.clicked.connect(self.calculate_mission)
        self.maneuver_box.currentTextChanged.connect(self.update_header)

        # 6. ANIMATION ENGINE
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)
        self.current_frame = 0

        # 7. AI SURROGATE INTEGRATION
        try:
            # Locate the model files dynamically
            current_folder = os.path.dirname(__file__)
            project_root = os.path.abspath(os.path.join(current_folder, '..'))

            model_path = os.path.join(project_root, 'ml', 'surrogate_model.pkl')
            columns_path = os.path.join(project_root, 'ml', 'model_columns.pkl')

            self.ai_model = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
            self.ai_status = "ONLINE"
        except FileNotFoundError:
            # Failsafe: If the model isn't found, the dashboard still works on pure physics!
            self.ai_model = None
            self.ai_status = "OFFLINE"

        # 8. Phasing Orbit Phase Angle (Starts Hidden)
        self.label_phase = QLabel("Phase Angle (degrees):")
        self.input_phase = QLineEdit("10")  # Default 10 degree lead

        panel_layout.addWidget(self.label_phase, 6, 0)
        panel_layout.addWidget(self.input_phase, 6, 1)

        self.label_phase.hide()
        self.input_phase.hide()

    def calculate_mission(self):
        """This function runs when the button is clicked."""
        try:
            # take input and turn them into numbers (floats)
            alt1_km = float(self.input_r1.text())
            alt2_km = float(self.input_r2.text())
            final_mass = float(self.input_mass.text())

            # Constants
            mu = 3.986e14
            r_earth = 6371000
            isp = 310

            # Math (Converting km to meters and adding Earth radius)
            r1 = (alt1_km * 1000) + r_earth
            r2 = (alt2_km * 1000) + r_earth

            # Call our core engine
            # Find out which maneuver the user selected
            maneuver_type = self.maneuver_box.currentText()

            # 4. Route to the correct core engine math
            if maneuver_type == "Hohmann Transfer":
                delta_v = hohmann_transfer(mu, r1, r2)


            elif maneuver_type == "Bi-Elliptic Transfer":

                # Get the rb value from the UI

                rb_km = float(self.input_rb.text())

                rb = (rb_km * 1000) + r_earth

                # SAFETY CHECK: rb must be larger than r1 and r2

                if rb <= r1 or rb <= r2:
                    self.result_label.setStyleSheet("color: #ff6b35;")

                    self.result_label.setText("ERROR: DEEP SPACE RADIUS MUST BE LARGER THAN ORBITS.")

                    return
                delta_v = bi_elliptic_transfer(mu, r1, r2, rb)



            elif maneuver_type == "Phasing Orbit":

                phase_angle = float(self.input_phase.text())

                # Note: r2 is not used here, we stay in r1

                delta_v = phasing_maneuver(mu, r1, phase_angle)

                # Update graph for phasing (drawing a closed loop)

                self.plotter.draw_orbits(alt1_km, alt1_km, maneuver=maneuver_type)
            wet_mass = calculate_initial_mass(delta_v, isp, final_mass)
            propellant = wet_mass - final_mass

            # --- AI SURROGATE PREDICTION ---
            ai_propellant_text = "N/A"
            if self.ai_model is not None:
                # Format the inputs for the AI (Handling the One-Hot Encoding manually)
                phase_angle = float(self.input_phase.text()) if maneuver_type == "Phasing Orbit" else 0.0
                rb_safe = rb_km if maneuver_type == "Bi-Elliptic Transfer" else 0.0
                r2_safe = alt2_km if maneuver_type != "Phasing Orbit" else 0.0

                input_data = {
                    'R1_km': [alt1_km],
                    'R2_km': [r2_safe],
                    'Rb_km': [rb_safe],
                    'Phase_Angle': [phase_angle],
                    'Payload_kg': [final_mass],
                    'Maneuver_Type_Bi-Elliptic': [1 if maneuver_type == "Bi-Elliptic Transfer" else 0],
                    'Maneuver_Type_Hohmann': [1 if maneuver_type == "Hohmann Transfer" else 0],
                    'Maneuver_Type_Phasing': [1 if maneuver_type == "Phasing Orbit" else 0]
                }

                # Convert to DataFrame and align columns with the trained model
                input_df = pd.DataFrame(input_data)
                input_df = input_df.reindex(columns=self.model_columns, fill_value=0)

                # Get the prediction!
                ai_prediction = self.ai_model.predict(input_df)[0]

                # Safety Check: AI should never predict negative fuel
                if ai_prediction < 0: ai_prediction = 0.0

                ai_propellant_text = f"{ai_prediction:.2f} kg"


            # 5. Update the Graph
            if maneuver_type == "Bi-Elliptic Transfer":
                self.plotter.draw_orbits(alt1_km, alt2_km, maneuver=maneuver_type, rb_km=rb_km)
            else:
                self.plotter.draw_orbits(alt1_km, alt2_km, maneuver=maneuver_type)

            # 6. Start the Animation
            self.current_frame = 0
            # Speed slider controls the delay (1 to 100). Higher slider = lower delay = faster!
            delay_ms = int(1000 / self.speed_slider.value())
            self.animation_timer.start(delay_ms)

            # Update the UI Result Label (HMI Side-by-Side Verification)
            result_text = (f"MISSION SUCCESS | Required Δv: {delta_v:.2f} m/s\n"
                           f"PHYSICS Propellant: {propellant:.2f} kg  //  AI PREDICTED: {ai_propellant_text}")

            self.result_label.setStyleSheet("color: #00d4ff;")
            self.result_label.setText(result_text)

        except ValueError:
            # HMI Safety Feature: If the user types a letter instead of a number, catch the crash!
            self.result_label.setStyleSheet("color: #ff6b35;")  # Turn text orange/red on error
            self.result_label.setText("ERROR: INVALID INPUT DETECTED. NUMBERS ONLY.")

    def animation_step(self):
        """Fires every few milliseconds to move the rocket one step forward."""
        if len(self.plotter.flight_path_x) == 0:
            self.animation_timer.stop()
            return

        x = self.plotter.flight_path_x[self.current_frame]
        y = self.plotter.flight_path_y[self.current_frame]

        # 1. Calculate the flight direction (Tangent Vector)
        if self.current_frame < len(self.plotter.flight_path_x) - 1:
            next_x = self.plotter.flight_path_x[self.current_frame + 1]
            next_y = self.plotter.flight_path_y[self.current_frame + 1]
            dx = next_x - x
            dy = next_y - y
        else:
            dx, dy = 0, 0

        # 2. Pass the calculated vector to the plotter (CRITICAL LINE!)
        self.plotter.update_rocket_position(x, y, dx, dy, "Prograde")

        self.current_frame += 1

        if self.current_frame >= len(self.plotter.flight_path_x):
            self.animation_timer.stop()
            # Engine cutoff: send 0, 0 to hide the vector arrow when done
            self.plotter.update_rocket_position(x, y, 0, 0)

    def update_header(self, text):
        """Updates UI Title based on selected maneuver."""
        self.header.setText(text.upper())

        # Reset all dynamic visibilities
        self.label_rb.hide()
        self.input_rb.hide()
        self.label_phase.hide()
        self.input_phase.hide()

        # Show Target Orbit by default, hide it for Phasing
        self.input_r2.show()

        if text == "Bi-Elliptic Transfer":
            self.label_rb.show()
            self.input_rb.show()
        elif text == "Phasing Orbit":
            self.input_r2.hide()  # Pilot stays in r1
            self.label_phase.show()
            self.input_phase.show()

def run_app():
    app = QApplication(sys.argv)
    window = OrbitalDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()